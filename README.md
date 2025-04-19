# qwen2.5-0.5b-grpo

GRPO Training based on Qwen2.5 0.5B 

# usage

```
source activate base
pip install modelscope
pip install datasets
pip install addict
python3 -m pip install trl
python3 -m pip install tensorboard
pip install -U vllm
pip install typing_extensions
pip install modelscope==1.9.5
pip uninstall -y datasets fsspec
pip install datasets==2.12.0 fsspec==2023.9.2
```

# run
```
torchrun --nproc_per_node=2 --num_processors=number of cards-1  --master_port 1234 grpo_multi_gpu_revise.py
```


## hardware

* NVIDIA A10 24GB x 1

## dataset

openai gsm8k

## time usage

45 minutes

## codes

* QwenGRPO.ipynb: training code
* QwenTest.ipynb: test code

## results

![](./tensorboard.png)

**"aha moment" occurred at step 500.**

---- 

query

```
树上7个鸟，又飞来1个鸟，一共几个鸟？
```

completion

```
<reasoning>
初始时，树上有 7 只鸟。后来又有 1 只鸟飞进来，所以总共的鸟的数量是 \(7 + 1 = 8\)。
</reasoning>
<answer>
8
</answer>
```

## ref

* [GRPO paper](refs/grpo/GRPO：Group%20Relative%20Policy%20Optimization.pdf)
* [Huggingface GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
* [Huggingface GRPOTrainer source code](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
* [Unsloth R1](https://unsloth.ai/blog/r1-reasoning)
