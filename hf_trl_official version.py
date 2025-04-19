from datasets import load_dataset

dataset_id = "AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

model_id = "Qwen/Qwen2-0.5B-Instruct"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Set padding side

# Format as standard text for language model training
def make_plain_conversation(example):
    # Include multiple fields that the trainer might look for
    formatted_prompt = f"{SYSTEM_PROMPT}\n\nUser: {example['problem']}\n\nAssistant:"
    return {
        "input": example["problem"],  # Simple problem text
        "query": example["problem"],  # Also include as query field which some trainers look for
        "prompt": formatted_prompt,   # Full formatted prompt with system message
        "solution": example["solution"]  # Keep solution for reward functions
    }

# Apply the transformation and ensure dataset has only our specified fields
train_dataset = train_dataset.map(make_plain_conversation, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(make_plain_conversation, remove_columns=test_dataset.column_names)
print("Sample from training dataset:", train_dataset[0])
print("Keys in sample:", list(train_dataset[0].keys()))  # Debug: print all keys

import torch
from transformers import AutoModelForCausalLM
import torch.distributed as dist
import os

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

# 如果是PyTorch 2.0+，禁用自动DTensor转换
if hasattr(torch, "_C"):
    if hasattr(torch._C, "_set_print_stack_traces_on_fatal"):
        torch._C._set_print_stack_traces_on_fatal(True)
    if hasattr(torch._C, "_set_dtensor_auto_to_dtensor"):
        torch._C._set_dtensor_auto_to_dtensor(False)

model_id = "Qwen/Qwen2-0.5B-Instruct"
if is_distributed:
    # 分布式训练时，不要使用device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=None,  # 不使用自动设备映射
    )
    # 手动移动到正确的设备
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
else:
    # 非分布式环境可以使用auto
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )

import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    print("Format reward received kwargs:", list(kwargs.keys()))  # Debug
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
    return rewards


from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Get solutions from solution key directly
    if "solution" in kwargs:
        solutions = kwargs["solution"]
    else:
        print("Warning: No solutions found in kwargs:", list(kwargs.keys()))
        return [0.0] * len(completions)
    
    # 健壮的内容提取
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
    
    # 确保solutions和completions长度匹配
    if len(solutions) != len(completion_contents):
        print(f"Warning: Number of solutions ({len(solutions)}) does not match completions ({len(completion_contents)})")
        # 截断较长的列表以匹配较短的列表
        min_len = min(len(solutions), len(completion_contents))
        solutions = solutions[:min_len]
        completion_contents = completion_contents[:min_len]
    
    # 计算rewards
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
    
    return rewards



from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
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
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)


from trl import GRPOTrainer

# Create a custom data collator for GRPO training
class GRPODataCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length or 512
        
    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "solution": []}
        
        for feature in features:
            # Format with system prompt - we do this dynamically here
            text = feature["prompt"] if "prompt" in feature else f"{SYSTEM_PROMPT}\n\nUser: {feature['input']}\n\nAssistant:"
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Add to batch
            batch["input_ids"].append(tokenized["input_ids"][0])
            batch["attention_mask"].append(tokenized["attention_mask"][0])
            batch["solution"].append(feature.get("solution", ""))
        
        # Stack tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        
        return batch

# 创建trainer with custom collator
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
