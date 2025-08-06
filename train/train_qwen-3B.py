from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os

# export HF_ENDPOINT=https://hf-mirror.com && export HF_HUB_DISABLE_TELEMETRY=1 && sudo /home/panding/miniconda3/envs/torch/bin/python train/train_qwen-3B.py

# 必须在导入其他库之前设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

model_path = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True, # 允许执行远程代码（Qwen 模型需要）
    use_fast=False, # 使用慢速但更稳定的 tokenizer
    model_max_length=1024 # 设置最大序列长度为 1024 tokens
)
tokenizer.pad_token = tokenizer.eos_token # End of Sequence Token（序列结束标记），告诉模型"这里是文本的结束"
# 生成任务: 模型从左到右生成，最关心的是最后几个 token
# 注意力机制: 左侧填充让模型把注意力集中在真实内容的末尾，这对于继续生成很重要
# 因果语言模型: 下一个词的预测主要依赖前面的真实内容，而不是填充符
tokenizer.padding_side = "left" # 左侧填充（对生成任务更友好）

# 加载 JSONL 格式的训练和验证数据
train_raw = load_dataset("json", data_files="JSON/data_train.jsonl", split="train")
val_raw = load_dataset("json", data_files="JSON/data_val.jsonl", split="train")

print(f"Train raw count: {len(train_raw)}")
print(f"Val raw count:   {len(val_raw)}")

def count_length(example):
    # 数据格式化: 使用指令-响应格式，类似 ChatML 风格
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}{tokenizer.eos_token}"
    # 重要：在计算长度时也要设置 max_length，确保与后续处理一致
    tokens = tokenizer(prompt, truncation=True, max_length=1024, padding=False)
    example["length"] = len(tokens["input_ids"])
    return example

# 映射操作，对数据集中的每一条数据都执行相同的函数
train_with_length = train_raw.map(count_length)
val_with_length = val_raw.map(count_length)

# 筛选操作，只保留满足条件的数据 (现在这一步实际上保留所有数据，因为已经在 count_length 中截断)
train_filtered = train_with_length.filter(lambda x: x["length"] <= 1024)
val_filtered = val_with_length.filter(lambda x: x["length"] <= 1024)
print(f"Train filtered count: {len(train_filtered)}")
print(f"Val filtered count:   {len(val_filtered)}")

def preprocess(example):
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}{tokenizer.eos_token}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)

# 移除原始文本列，只保留 token 数据
train_dataset = train_filtered.map(preprocess, remove_columns=train_filtered.column_names)
val_dataset = val_filtered.map(preprocess, remove_columns=val_filtered.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 精度节省显存
    device_map="auto" # 自动分配设备（支持多GPU）
)

training_args = TrainingArguments(
    output_dir="/mnt/hzx/Text-to-Cadquery/checkpoints_qwen3b",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=2, # 有效批次大小: 6 × 2 = 12 （考虑梯度累积）
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    tf32=True,
    logging_steps=50,
    logging_dir="./logs_qwen3b",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
