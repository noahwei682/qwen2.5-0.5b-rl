import os
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
from trl import GRPOConfig, GRPOTrainer





# 系统提示和格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

import re
from datasets import load_dataset, Dataset

# 提取 XML 格式答案
def extract_xml_answer(text: str) -> str:
    match = re.search('<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = ''
    return answer.strip()

# 定义奖励函数
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [2 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [4 if match else 0.0 for match in matches]

from modelscope.msdatasets import MsDataset

# 提取 GSM8K 数据集答案
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# 加载 GSM8K 数据集
def get_gsm8k_questions(split="train") -> Dataset:
    data = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': '数字10203040里面有几个0?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(reasoning='可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0', answer='4')},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data




def main():
    # init_distributed_mode()
    # 下载并加载模型
    # model_name = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型并迁移到 GPU
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 torch_dtype=torch.bfloat16,)
    

    dataset = get_gsm8k_questions()
    print(dataset[0])
    # 配置训练参数
    training_args = GRPOConfig(
        use_vllm=False,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=300,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        vllm_gpu_memory_utilization=0.2,
        report_to="tensorboard",
        ddp_find_unused_parameters=False, 
        output_dir="outputs/Qwen2.5-0.5B-Instruct-GRPO",
    )

    # 初始化 Trainer
    trainer = GRPOTrainer(
        # model=ddp_model.module,  # 传递原始模型
        model=model,  
        processing_class=tokenizer,
        reward_funcs=[
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
