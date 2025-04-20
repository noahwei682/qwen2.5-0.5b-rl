from datasets import load_dataset
from datasets import load_dataset
import re
import wandb  # Import wandb here
import time
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from math_verify import LatexExtractionConfig, parse, verify

dataset_id = "Fancy-MLLM/R1-Onevision"
train_dataset = load_dataset(dataset_id, split="train[:5%]")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

model_id = "lmms-lab/llava-onevision-qwen2-7b-ov"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Set padding side

# Format as standard text for language model training
def make_plain_conversation(example):
    # Extract problem from the conversation structure
    problem = ""
    solution = ""
    
    if "conversations" in example:
        for item in example["conversations"]:
            if item.get("from") == "human":
                problem = item.get("value", "")
            elif item.get("from") == "assistant":
                solution = item.get("value", "")
    
    # Format the prompt with the problem
    formatted_prompt = f"{SYSTEM_PROMPT}\n\nUser: {problem}\n\nAssistant:"
    
    return {
        "input": problem,        # Simple problem text
        "query": problem,        # Also include as query field which some trainers look for
        "prompt": formatted_prompt,   # Full formatted prompt with system message
        "solution": solution     # Keep solution for reward functions
    }

# Print the keys of the first example to understand the available columns
print("Available keys in the dataset:", train_dataset.column_names)
print("First item sample:", train_dataset[0])

# Apply the transformation and ensure dataset has only our specified fields
train_dataset = train_dataset.map(make_plain_conversation, remove_columns=train_dataset.column_names)
print("Sample from training dataset:", train_dataset[0])
print("Keys in sample:", list(train_dataset[0].keys()))  # Debug: print all keys

# 初始化分布式环境
def init_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"Distributed initialization complete: rank {dist.get_rank()} of {dist.get_world_size()}")
        return True
    return False

# 调用初始化函数
is_distributed = init_distributed()

# If using PyTorch 2.0+, disable automatic DTensor conversion
if hasattr(torch, "_C"):
    if hasattr(torch._C, "_set_print_stack_traces_on_fatal"):
        torch._C._set_print_stack_traces_on_fatal(True)
    if hasattr(torch._C, "_set_dtensor_auto_to_dtensor"):
        torch._C._set_dtensor_auto_to_dtensor(False)

model_id = "Qwen/Qwen2-0.5B-Instruct"
if is_distributed:
    # For distributed training, don't use device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=None,  # Don't use automatic device mapping
    )
    # Manually move to the correct device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
else:
    # For non-distributed environment, can use auto
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )

# Create the wandb callback reference first (needed by reward functions)
wandb_callback = None

# Global counter for tracking reward function calls
reward_call_counts = {
    "format": 0,
    "accuracy": 0,
    "bm25": 0,
    "f1": 0,
    "recall": 0,
    "precision": 0,
    "sbert_cosine": 0
}

# Load the Sentence-BERT model (do this once at module level to avoid reloading)
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded Sentence-BERT model successfully")
except Exception as e:
    print(f"Error loading Sentence-BERT model: {e}")
    sbert_model = None

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    global reward_call_counts
    reward_call_counts["format"] += 1
    step_count = reward_call_counts["format"]
    
    print(f"Format reward call #{step_count}, kwargs keys: {list(kwargs.keys())}")  # Debug
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = []
    
    for i, completion in enumerate(completions):
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
            print(f"Completion {i}: {content[:50]}...")  # Debug first 50 chars
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            print(f"Completion type: {type(completion)}")  # Debug type
            completion_contents.append("")
    
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    print(f"Format rewards: {rewards}")  # Debug rewards
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "format_reward_direct": avg_reward,
            "format_reward_call": step_count,
            "format_rewards_raw": rewards,
            "format_reward_min": min(rewards) if rewards else 0,
            "format_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: format_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_format_rewards = rewards
    
    return rewards

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    global reward_call_counts
    reward_call_counts["accuracy"] += 1
    step_count = reward_call_counts["accuracy"]
    
    print(f"Accuracy reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate rewards
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            
            if len(gold_parsed) != 0:
                try:
                    reward = float(verify(answer_parsed, gold_parsed))
                    rewards.append(reward)
                except Exception as e:
                    print(f"Verification error: {e}")
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "accuracy_reward_direct": avg_reward,
            "accuracy_reward_call": step_count,
            "accuracy_rewards_raw": rewards,
            "accuracy_reward_min": min(rewards) if rewards else 0,
            "accuracy_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: accuracy_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_accuracy_rewards = rewards
    
    return rewards

# Define BM25 reward function based on BM25 implementation
def bm25_reward(completions, **kwargs):
    """Reward function that computes BM25 scores between model completion and solution"""
    global reward_call_counts
    reward_call_counts["bm25"] += 1
    step_count = reward_call_counts["bm25"]
    
    print(f"BM25 reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Create corpus for BM25 calculation
    corpus = solutions + completion_contents
    
    # Calculate BM25 scores
    rewards = []
    k1 = 1.2  # Parameter for BM25
    b = 0.75  # Parameter for BM25
    
    # Calculate average document length for BM25
    N = len(corpus)
    avgdl = sum(len(doc.split()) for doc in corpus) / N if N > 0 else 0
    
    # Helper function to calculate IDF
    def idf(term):
        n_t = sum(1 for doc in corpus if term in doc.split())
        return math.log((N - n_t + 0.5) / (n_t + 0.5) + 1) if n_t > 0 else 0
    
    # Calculate BM25 for each completion-solution pair
    for content, solution in zip(completion_contents, solutions):
        try:
            # Use solution as query and content as document
            query_terms = solution.split()
            doc_terms = content.split()
            doc_len = len(doc_terms)
            
            # Calculate BM25 score
            bm25_score = 0
            for term in query_terms:
                f_td = doc_terms.count(term)  # Term frequency
                idf_t = idf(term)
                if f_td > 0 and idf_t > 0:
                    numerator = f_td * (k1 + 1)
                    denominator = f_td + k1 * (1 - b + b * (doc_len / avgdl)) if avgdl > 0 else 1
                    bm25_score += idf_t * (numerator / denominator)
            
            # Normalize BM25 score to a 0-1 range
            # Assuming a maximum possible score of 10 for normalization
            normalized_score = min(bm25_score / 10.0, 1.0)
            rewards.append(normalized_score)
            
            print(f"BM25 score for pair: {normalized_score:.4f}")
        except Exception as e:
            print(f"Error in BM25 calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "bm25_reward_direct": avg_reward,
            "bm25_reward_call": step_count,
            "bm25_rewards_raw": rewards,
            "bm25_reward_min": min(rewards) if rewards else 0,
            "bm25_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: bm25_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_bm25_rewards = rewards
    
    return rewards

# Define F1 score reward function
def f1_reward(completions, **kwargs):
    """Reward function that computes F1 scores between model completion and solution"""
    global reward_call_counts
    reward_call_counts["f1"] += 1
    step_count = reward_call_counts["f1"]
    
    print(f"F1 reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate F1 scores
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            # Tokenize content and solution (simple splitting by whitespace)
            content_tokens = set(content.lower().split())
            solution_tokens = set(solution.lower().split())
            
            # Calculate precision, recall, and F1 score
            if not solution_tokens or not content_tokens:
                # If either set is empty, we can't compute a meaningful F1 score
                rewards.append(0.0)
                continue
                
            # Find common tokens (intersection)
            common_tokens = content_tokens.intersection(solution_tokens)
            
            # Calculate precision: common / generated
            precision = len(common_tokens) / len(content_tokens) if content_tokens else 0
            
            # Calculate recall: common / reference
            recall = len(common_tokens) / len(solution_tokens) if solution_tokens else 0
            
            # Calculate F1 score: harmonic mean of precision and recall
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            rewards.append(f1)
            print(f"F1 score for pair: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f}, common: {len(common_tokens)})")
            
        except Exception as e:
            print(f"Error in F1 calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "f1_reward_direct": avg_reward,
            "f1_reward_call": step_count,
            "f1_rewards_raw": rewards,
            "f1_reward_min": min(rewards) if rewards else 0,
            "f1_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: f1_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_f1_rewards = rewards
    
    return rewards

# Define Recall reward function
def recall_reward(completions, **kwargs):
    """Reward function that computes recall score between model completion and solution"""
    global reward_call_counts
    reward_call_counts["recall"] += 1
    step_count = reward_call_counts["recall"]
    
    print(f"Recall reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate Recall scores
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            # Tokenize content and solution (simple splitting by whitespace)
            content_tokens = set(content.lower().split())
            solution_tokens = set(solution.lower().split())
            
            # Calculate recall score
            if not solution_tokens:
                # If solution is empty, recall is meaningless
                rewards.append(0.0)
                continue
                
            # Find common tokens (intersection)
            common_tokens = content_tokens.intersection(solution_tokens)
            
            # Calculate recall: common / reference
            recall = len(common_tokens) / len(solution_tokens)
            
            rewards.append(recall)
            print(f"Recall score for pair: {recall:.4f} (common: {len(common_tokens)}, solution: {len(solution_tokens)})")
            
        except Exception as e:
            print(f"Error in Recall calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "recall_reward_direct": avg_reward,
            "recall_reward_call": step_count,
            "recall_rewards_raw": rewards,
            "recall_reward_min": min(rewards) if rewards else 0,
            "recall_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: recall_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_recall_rewards = rewards
    
    return rewards

# Define Precision reward function
def precision_reward(completions, **kwargs):
    """Reward function that computes precision score between model completion and solution"""
    global reward_call_counts
    reward_call_counts["precision"] += 1
    step_count = reward_call_counts["precision"]
    
    print(f"Precision reward call #{step_count}")
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate Precision scores
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            # Tokenize content and solution (simple splitting by whitespace)
            content_tokens = set(content.lower().split())
            solution_tokens = set(solution.lower().split())
            
            # Calculate precision score
            if not content_tokens:
                # If completion is empty, precision is meaningless
                rewards.append(0.0)
                continue
                
            # Find common tokens (intersection)
            common_tokens = content_tokens.intersection(solution_tokens)
            
            # Calculate precision: common / generated
            precision = len(common_tokens) / len(content_tokens)
            
            rewards.append(precision)
            print(f"Precision score for pair: {precision:.4f} (common: {len(common_tokens)}, generated: {len(content_tokens)})")
            
        except Exception as e:
            print(f"Error in Precision calculation: {e}")
            rewards.append(0.0)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "precision_reward_direct": avg_reward,
            "precision_reward_call": step_count,
            "precision_rewards_raw": rewards,
            "precision_reward_min": min(rewards) if rewards else 0,
            "precision_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: precision_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_precision_rewards = rewards
    
    return rewards

# Define Sentence-BERT Cosine Similarity reward function
def sbert_cosine_reward(completions, **kwargs):
    """Reward function that computes cosine similarity between sentence embeddings of completions and solutions"""
    global reward_call_counts, sbert_model
    
    reward_call_counts["sbert_cosine"] += 1
    step_count = reward_call_counts["sbert_cosine"]
    
    print(f"SBERT Cosine reward call #{step_count}")
    
    # Check if SBERT model is available
    if sbert_model is None:
        print("Sentence-BERT model not available. Returning zero rewards.")
        return [0.0] * len(completions)
    
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print(f"Warning: No solutions found in kwargs: {list(kwargs.keys())}")
        return [0.0] * len(completions)
    
    # Robust content extraction
    completion_contents = []
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                content = completion.get("content", "")
            else:
                content = str(completion)
            completion_contents.append(content)
        except Exception as e:
            print(f"Error extracting content from completion: {e}")
            completion_contents.append("")
    
    # Ensure solutions and completions have matching lengths
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # Truncate the longer list to match the shorter list
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # Calculate SBERT Cosine Similarity scores
    rewards = []
    
    try:
        # Generate embeddings for solutions and completions
        solution_embeddings = sbert_model.encode(solutions, convert_to_tensor=True)
        completion_embeddings = sbert_model.encode(completion_contents, convert_to_tensor=True)
        
        # Convert to numpy arrays for sklearn cosine_similarity
        if torch.is_tensor(solution_embeddings):
            solution_embeddings_np = solution_embeddings.cpu().numpy()
        else:
            solution_embeddings_np = solution_embeddings
            
        if torch.is_tensor(completion_embeddings):
            completion_embeddings_np = completion_embeddings.cpu().numpy()
        else:
            completion_embeddings_np = completion_embeddings
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(solution_embeddings_np, completion_embeddings_np)
        
        # Extract diagonal (pairwise similarities between corresponding items)
        for i in range(min(len(solutions), len(completion_contents))):
            sim_score = float(similarity_matrix[i, i])
            rewards.append(sim_score)
            print(f"SBERT Cosine similarity for pair {i}: {sim_score:.4f}")
    
    except Exception as e:
        print(f"Error in SBERT Cosine calculation: {e}")
        # If there's an error, return zeros
        rewards = [0.0] * len(completion_contents)
    
    # Direct wandb logging with explicit step counter
    if wandb.run is not None:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        wandb.log({
            "sbert_cosine_reward_direct": avg_reward,
            "sbert_cosine_reward_call": step_count,
            "sbert_cosine_rewards_raw": rewards,
            "sbert_cosine_reward_min": min(rewards) if rewards else 0,
            "sbert_cosine_reward_max": max(rewards) if rewards else 0,
        }, step=step_count)
        print(f"DIRECT LOG to wandb: sbert_cosine_reward={avg_reward} at step {step_count}")
    
    # Store in callback if possible
    if wandb_callback and hasattr(wandb_callback, 'trainer') and wandb_callback.trainer:
        wandb_callback.trainer._last_sbert_cosine_rewards = rewards
    
    return rewards

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="llava-onevision-qwen2-7b-ov-test",
    learning_rate=1e-5,
    remove_unused_columns=False,  # Keep all columns including solution as needed by the reward functions
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=["wandb"],  # Use wandb for logging
    logging_steps=1,  # Log at every step
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
    # Set KL coefficient to 0 to disable KL penalty
)

# Initialize wandb
import wandb
wandb.init(
    project="llava-onevision-qwen2-7b-ov-test",
    config={
        "model": model_id,
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "max_completion_length": training_args.max_completion_length,
        "max_prompt_length": training_args.max_prompt_length,
        "reward_functions": ["format", "accuracy", "bm25", "f1", "recall", "precision", "sbert_cosine"],
        "bm25_params": {"k1": 1.2, "b": 0.75},
        "sbert_model": "all-MiniLM-L6-v2"
    }
)

# Create a simple callback to log rewards at every step
from transformers import TrainerCallback

class StepRewardCallback(TrainerCallback):
    """Simple callback to log rewards at every step"""
    
    def __init__(self):
        self.step = 0
        self.format_rewards = []
        self.accuracy_rewards = []
        self.bm25_rewards = []
        self.f1_rewards = []
        self.recall_rewards = []
        self.precision_rewards = []
        self.sbert_cosine_rewards = []
        self.kl_values = []
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step += 1
        print(f"Starting step {self.step}")
    
    def on_step_end(self, args, state, control, **kwargs):
        # Log step completion with any available metrics
        if wandb.run is not None:
            metrics = {"step": self.step}
            
            # Add rewards if available
            if self.format_rewards:
                avg_format = sum(self.format_rewards) / len(self.format_rewards)
                metrics["format_reward"] = avg_format
                
                # Log individual format rewards
                for i, reward in enumerate(self.format_rewards):
                    metrics[f"format_reward_{i}"] = reward
                    
                print(f"Format rewards at step {self.step}: {self.format_rewards}")
                # Reset for next step
                self.format_rewards = []
            
            if self.accuracy_rewards:
                avg_accuracy = sum(self.accuracy_rewards) / len(self.accuracy_rewards)
                metrics["accuracy_reward"] = avg_accuracy
                
                # Log individual accuracy rewards
                for i, reward in enumerate(self.accuracy_rewards):
                    metrics[f"accuracy_reward_{i}"] = reward
                
                print(f"Accuracy rewards at step {self.step}: {self.accuracy_rewards}")
                # Reset for next step
                self.accuracy_rewards = []
            
            # Add BM25 rewards logging
            if self.bm25_rewards:
                avg_bm25 = sum(self.bm25_rewards) / len(self.bm25_rewards)
                metrics["bm25_reward"] = avg_bm25
                
                # Log individual BM25 rewards
                for i, reward in enumerate(self.bm25_rewards):
                    metrics[f"bm25_reward_{i}"] = reward
                
                print(f"BM25 rewards at step {self.step}: {self.bm25_rewards}")
                # Reset for next step
                self.bm25_rewards = []
            
            # Add F1 rewards logging
            if self.f1_rewards:
                avg_f1 = sum(self.f1_rewards) / len(self.f1_rewards)
                metrics["f1_reward"] = avg_f1
                
                # Log individual F1 rewards
                for i, reward in enumerate(self.f1_rewards):
                    metrics[f"f1_reward_{i}"] = reward
                
                print(f"F1 rewards at step {self.step}: {self.f1_rewards}")
                # Reset for next step
                self.f1_rewards = []
            
            # Add Recall rewards logging
            if self.recall_rewards:
                avg_recall = sum(self.recall_rewards) / len(self.recall_rewards)
                metrics["recall_reward"] = avg_recall
                
                # Log individual Recall rewards
                for i, reward in enumerate(self.recall_rewards):
                    metrics[f"recall_reward_{i}"] = reward
                
                print(f"Recall rewards at step {self.step}: {self.recall_rewards}")
                # Reset for next step
                self.recall_rewards = []
            
            # Add Precision rewards logging
            if self.precision_rewards:
                avg_precision = sum(self.precision_rewards) / len(self.precision_rewards)
                metrics["precision_reward"] = avg_precision
                
                # Log individual Precision rewards
                for i, reward in enumerate(self.precision_rewards):
                    metrics[f"precision_reward_{i}"] = reward
                
                print(f"Precision rewards at step {self.step}: {self.precision_rewards}")
                # Reset for next step
                self.precision_rewards = []
            
            # Add SBERT Cosine rewards logging
            if self.sbert_cosine_rewards:
                avg_sbert_cosine = sum(self.sbert_cosine_rewards) / len(self.sbert_cosine_rewards)
                metrics["sbert_cosine_reward"] = avg_sbert_cosine
                
                # Log individual SBERT Cosine rewards
                for i, reward in enumerate(self.sbert_cosine_rewards):
                    metrics[f"sbert_cosine_reward_{i}"] = reward
                
                print(f"SBERT Cosine rewards at step {self.step}: {self.sbert_cosine_rewards}")
                # Reset for next step
                self.sbert_cosine_rewards = []
            
            # Track KL value
            if hasattr(control, '_trainer') and hasattr(control._trainer, 'state'):
                if hasattr(control._trainer.state, 'log_history') and control._trainer.state.log_history:
                    # Try to extract KL value from logs
                    for log_entry in control._trainer.state.log_history[-5:]:  # Look at recent logs
                        if 'kl' in log_entry:
                            metrics["kl_divergence"] = log_entry['kl']
                            print(f"KL divergence at step {self.step}: {log_entry['kl']}")
                            break
            
            # Log explicitly that KL is disabled
            # metrics["kl_coef"] = training_args.kl_coef
            
            wandb.log(metrics)
            print(f"Logged metrics for step {self.step}: {list(metrics.keys())}")

# Create a global callback instance
step_callback = StepRewardCallback()

# Modify reward functions to track rewards and print them for logging
original_format_reward = format_reward
def format_reward_with_logging(completions, **kwargs):
    rewards = original_format_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW format rewards: {rewards}")
    # Store for logging
    step_callback.format_rewards.extend(rewards)
    return rewards

original_accuracy_reward = accuracy_reward
def accuracy_reward_with_logging(completions, **kwargs):
    rewards = original_accuracy_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW accuracy rewards: {rewards}")
    
    # Print the parameters that go into the calculation
    if "solution" in kwargs:
        for i, (completion, solution) in enumerate(zip(completions, kwargs["solution"])):
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            print(f"Completion {i}:")
            print(f"  Solution: {solution}")
            print(f"  Content: {content[:100]}...")
            print(f"  Reward: {rewards[i]}")
    
    # Store for logging
    step_callback.accuracy_rewards.extend(rewards)
    return rewards

original_bm25_reward = bm25_reward
def bm25_reward_with_logging(completions, **kwargs):
    rewards = original_bm25_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW BM25 rewards: {rewards}")
    
    # Store for logging
    step_callback.bm25_rewards.extend(rewards)
    return rewards

original_f1_reward = f1_reward
def f1_reward_with_logging(completions, **kwargs):
    rewards = original_f1_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW F1 rewards: {rewards}")
    
    # Store for logging
    step_callback.f1_rewards.extend(rewards)
    return rewards

original_recall_reward = recall_reward
def recall_reward_with_logging(completions, **kwargs):
    rewards = original_recall_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW Recall rewards: {rewards}")
    
    # Store for logging
    step_callback.recall_rewards.extend(rewards)
    return rewards

original_precision_reward = precision_reward
def precision_reward_with_logging(completions, **kwargs):
    rewards = original_precision_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW Precision rewards: {rewards}")
    
    # Store for logging
    step_callback.precision_rewards.extend(rewards)
    return rewards

original_sbert_cosine_reward = sbert_cosine_reward
def sbert_cosine_reward_with_logging(completions, **kwargs):
    rewards = original_sbert_cosine_reward(completions, **kwargs)
    # Print rewards for visibility
    print(f"RAW SBERT Cosine rewards: {rewards}")
    
    # Store for logging
    step_callback.sbert_cosine_rewards.extend(rewards)
    return rewards

# Create trainer with logging
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_with_logging, 
        accuracy_reward_with_logging, 
        bm25_reward_with_logging, 
        f1_reward_with_logging, 
        recall_reward_with_logging, 
        precision_reward_with_logging,
        sbert_cosine_reward_with_logging
    ],
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[step_callback],  # Add our step logging callback
)

# Train the model and track with wandb
trainer.train()

# Close wandb run when done
wandb.finish()

# Implement a high-frequency metrics logger for wandb
class RealTimeMonitor:
    """A class to monitor and log metrics in real-time during training"""
    
    def __init__(self):
        self.iteration = 0
        self.batch_count = 0
        self.reward_freq = 0
        self.loss_freq = 0
        self.param_update_freq = 0
    
    def increment(self):
        """Increment internal counter for unique logging"""
        self.iteration += 1
        return self.iteration
    
    def log_batch(self, batch_data):
        """Log batch information"""
        self.batch_count += 1
        if wandb.run is not None:
            wandb.log({
                "batch_counter": self.batch_count,
                "batch_timestamp": time.time(),
                "batch_size": len(batch_data) if isinstance(batch_data, list) else 1
            }, step=self.increment())
    
    def log_reward(self, reward_values, reward_type="unknown"):
        """Log reward values immediately"""
        self.reward_freq += 1
        if wandb.run is not None and reward_values:
            # Calculate statistics
            try:
                avg_reward = sum(reward_values) / len(reward_values)
                min_reward = min(reward_values)
                max_reward = max(reward_values)
                
                # Log detailed metrics
                wandb.log({
                    f"rt_{reward_type}_reward_mean": avg_reward,
                    f"rt_{reward_type}_reward_min": min_reward,
                    f"rt_{reward_type}_reward_max": max_reward,
                    f"rt_{reward_type}_reward_freq": self.reward_freq,
                    f"rt_reward_timestamp": time.time(),
                }, step=self.increment())
                
                # Log each individual reward value
                for i, val in enumerate(reward_values):
                    wandb.log({f"rt_{reward_type}_reward_{i}": val}, step=self.iteration)
                
                print(f"RT Monitor: Logged {reward_type} rewards (avg={avg_reward:.4f})")
            except Exception as e:
                print(f"Error logging rewards: {e}")
    
    def log_loss(self, loss_value):
        """Log loss value immediately"""
        self.loss_freq += 1
        if wandb.run is not None:
            try:
                # Convert to float if needed
                if hasattr(loss_value, 'item'):
                    loss_value = loss_value.item()
                
                wandb.log({
                    "rt_loss": float(loss_value),
                    "rt_loss_freq": self.loss_freq,
                    "rt_loss_timestamp": time.time()
                }, step=self.increment())
                
                print(f"RT Monitor: Logged loss value: {float(loss_value):.6f}")
            except Exception as e:
                print(f"Error logging loss: {e}")
    
    def log_param_update(self, model):
        """Log parameter updates"""
        self.param_update_freq += 1
        if wandb.run is not None and model is not None:
            try:
                # Sample a few parameters to track their changes
                sample_params = {}
                for i, (name, param) in enumerate(model.named_parameters()):
                    if i % 1000 == 0:  # Sample every 1000th parameter
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad).item()
                            param_norm = torch.norm(param).item()
                            sample_params[f"param_{name}_norm"] = param_norm
                            sample_params[f"grad_{name}_norm"] = grad_norm
                
                # Log parameter statistics
                wandb.log({
                    "rt_param_update_freq": self.param_update_freq,
                    "rt_param_update_timestamp": time.time(),
                    **sample_params
                }, step=self.increment())
                
                print(f"RT Monitor: Logged parameter update #{self.param_update_freq}")
            except Exception as e:
                print(f"Error logging parameter update: {e}")

# Create an instance of the real-time monitor
rt_monitor = RealTimeMonitor()

# Patch the GRPOTrainer methods to log metrics in real-time
def patched_training_step(original_fn):
    def wrapper(self, *args, **kwargs):
        # Log batch information
        if len(args) > 1:
            rt_monitor.log_batch(args[1])  # args[1] is typically inputs
        
        # Call original function
        result = original_fn(self, *args, **kwargs)
        
        # Log loss
        if isinstance(result, torch.Tensor):
            rt_monitor.log_loss(result)
        
        return result
    return wrapper

def patched_format_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "format")
        
        return rewards
    return wrapper

def patched_accuracy_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "accuracy")
        
        return rewards
    return wrapper

def patched_bm25_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "bm25")
        
        return rewards
    return wrapper

def patched_f1_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "f1")
        
        return rewards
    return wrapper

def patched_recall_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "recall")
        
        return rewards
    return wrapper

def patched_precision_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "precision")
        
        return rewards
    return wrapper

def patched_sbert_cosine_reward(original_fn):
    def wrapper(*args, **kwargs):
        # Call original function
        rewards = original_fn(*args, **kwargs)
        
        # Log rewards
        rt_monitor.log_reward(rewards, "sbert_cosine")
        
        return rewards
    return wrapper

# Apply patches to monitor in real-time
original_training_step = GRPOTrainer.training_step
GRPOTrainer.training_step = patched_training_step(original_training_step)
format_reward = patched_format_reward(format_reward)
accuracy_reward = patched_accuracy_reward(accuracy_reward)
bm25_reward = patched_bm25_reward(bm25_reward)
f1_reward = patched_f1_reward(f1_reward)
recall_reward = patched_recall_reward(recall_reward)
precision_reward = patched_precision_reward(precision_reward)
sbert_cosine_reward = patched_sbert_cosine_reward(sbert_cosine_reward)

# Add a hook to track parameter gradients in real-time
def add_gradient_tracking_hooks(model):
    """Add hooks to track gradients for all parameters"""
    if model is None:
        return
    
    hooks = []
    
    def make_hook(name, param_id):
        def hook(grad):
            # Log gradient statistics immediately
            if wandb.run is not None and grad is not None:
                try:
                    grad_norm = torch.norm(grad).item()
                    wandb.log({
                        f"grad_{param_id}_norm": grad_norm,
                        "hook_timestamp": time.time()
                    })
                    if param_id % 100 == 0:  # Print only occasionally to avoid spam
                        print(f"Gradient norm for {name} (id:{param_id}): {grad_norm:.6f}")
                except Exception as e:
                    print(f"Error in gradient hook: {e}")
            return grad
        return hook
    
    # Register hooks for all parameters
    for param_id, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            # Only track a subset of parameters to avoid overwhelming wandb
            if param_id % 1000 == 0:
                hook = param.register_hook(make_hook(name, param_id))
                hooks.append(hook)
    
    print(f"Registered {len(hooks)} gradient tracking hooks")
    return hooks

# Patch the model's forward method to log inputs and outputs
def patch_model_forward(model):
    """Patch model's forward method to log info about inputs and outputs"""
    if model is None:
        return model
    
    original_forward = model.forward
    
    def patched_forward(*args, **kwargs):
        # Log information about inputs
        if wandb.run is not None:
            try:
                # Extract input_ids if available
                input_ids = None
                if 'input_ids' in kwargs:
                    input_ids = kwargs['input_ids']
                elif len(args) > 0 and hasattr(args[0], 'shape'):
                    input_ids = args[0]
                
                if input_ids is not None:
                    wandb.log({
                        "forward_input_shape": str(input_ids.shape),
                        "forward_timestamp": time.time()
                    })
            except Exception as e:
                print(f"Error logging forward inputs: {e}")
        
        # Call original forward method
        outputs = original_forward(*args, **kwargs)
        
        # Log information about outputs
        if wandb.run is not None:
            try:
                # Extract loss if available in outputs
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss_value = outputs.loss.item()
                    wandb.log({
                        "forward_loss": loss_value,
                        "forward_output_timestamp": time.time()
                    })
                    print(f"Forward pass completed with loss: {loss_value:.6f}")
            except Exception as e:
                print(f"Error logging forward outputs: {e}")
        
        return outputs
    
    # Replace the forward method
    model.forward = patched_forward
    print("Model forward method patched for real-time logging")
    
    return model

# Create trainer with patched model
def create_trainer_with_patched_model():
    """Create the GRPOTrainer with real-time monitoring enabled"""
    global model, wandb_callback, step_callback
    
    # Add gradient tracking hooks
    gradient_hooks = add_gradient_tracking_hooks(model)
    
    # Patch the model's forward method
    patched_model = patch_model_forward(model)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=patched_model,
        reward_funcs=[
            format_reward, 
            accuracy_reward, 
            bm25_reward, 
            f1_reward, 
            recall_reward, 
            precision_reward,
            sbert_cosine_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[wandb_callback, step_callback],  # Add both callbacks
    )
    
    # Log trainer creation
    if wandb.run is not None:
        wandb.log({
            "trainer_created": True,
            "trainer_timestamp": time.time(),
            "gradient_hooks_added": len(gradient_hooks) if gradient_hooks else 0
        })
    
    return trainer
