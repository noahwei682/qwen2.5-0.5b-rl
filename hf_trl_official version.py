# export HF_HOME=/home/aiscuser/qwen2.5-0.5b-rl
# export HF_TOKEN=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
# huggingface-cli login --token $HF_TOKEN
# !pip install  -U -q trl peft math_verify
# # Tested with transformers==4.47.1, trl==0.14.0, datasets==3.2.0, peft==0.14.0, accelerate==1.2.1, math_verify==0.3.3

from datasets import load_dataset
from transformers import DataCollatorWithPadding

dataset_id = "AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])

# Print detailed dataset information
print("\nDataset Information:")
print("Features:", train_dataset.features)
print("\nColumn names:", train_dataset.column_names)
print("\nFirst example:")
for key, value in train_dataset[0].items():
    print(f"{key}: {value}")

# # Stop here to inspect the output
# import sys
# sys.exit(0)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="left",
    truncation_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
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

def make_conversation(example):
    # Get the problem text
    problem_text = example['problem']
    if isinstance(problem_text, list):
        problem_text = problem_text[0]  # Take the first element if it's a list
    
    # Ensure problem_text is a string
    if not isinstance(problem_text, str):
        problem_text = str(problem_text)
    
    # Convert the conversation to a single text string
    text = f"system: {SYSTEM_PROMPT}\nuser: {problem_text}\n"
    return {
        "text": text,
        "solution": example["solution"]
    }

def tokenize_function(examples):
    # Get the texts
    texts = examples["text"]
    if not isinstance(texts, list):
        texts = [texts]
    
    # Ensure all texts are strings
    texts = [str(text) if not isinstance(text, str) else text for text in texts]
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=training_args.max_prompt_length,
        return_tensors=None
    )
    
    # Convert to lists for dataset compatibility
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "solution": examples["solution"]
    }

# First convert to text format
train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

# Then tokenize
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Print dataset info
print("\nDataset features:", train_dataset.features)
print("\nFirst example:")
for key, value in train_dataset[0].items():
    print(f"{key}: {value}")
    print(f"Type: {type(value)}")
    if isinstance(value, list):
        print(f"Length: {len(value)}")
        if len(value) > 0:
            print(f"First element type: {type(value[0])}")

import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]



# pip install math_verify
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs.get("solutions", [])  # Get solutions from kwargs
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


from trl import GRPOTrainer

# Subclass the GRPOTrainer to handle the solutions field
class CustomGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs):
        # Save solutions for the reward functions but don't pass them to the model
        solutions = inputs.pop("solutions", None)
        
        # Call the model with the filtered inputs
        outputs = model(**inputs)
        
        # Debugging: Print more detailed information about outputs
        print("Type of outputs:", type(outputs))
        print("Dir of outputs:", dir(outputs))
        
        # Check if 'loss' is an attribute
        if hasattr(outputs, 'loss'):
            print("Type of outputs.loss:", type(outputs.loss))
            
            # If outputs.loss is a dictionary, print its keys
            if isinstance(outputs.loss, dict):
                print("Keys in outputs.loss:", outputs.loss.keys())
                # Try to extract the first value from the dictionary as loss
                if len(outputs.loss) > 0:
                    loss = next(iter(outputs.loss.values()))
                else:
                    raise ValueError("outputs.loss is an empty dictionary")
            else:
                loss = outputs.loss
        else:
            raise ValueError("outputs does not have a 'loss' attribute")
            
        # Put back solutions for later use
        if solutions is not None:
            inputs["solutions"] = solutions
            
        # Ensure loss is a scalar tensor
        if torch.is_tensor(loss) and loss.numel() > 1:
            print(f"Converting loss tensor of shape {loss.shape} to scalar with mean()")
            loss = loss.mean()
            
        # Return the scalar loss value
        return loss

# Create a custom data collator
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Create a batch with model inputs
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "solutions": [f["solution"] for f in features]  # Store solutions as a list in the batch
        }
        
        return batch

# Create the trainer
trainer = CustomGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_functions=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    data_collator=CustomDataCollator(tokenizer)
)

# Train the model
trainer.train()
