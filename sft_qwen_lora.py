import json
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
# 1. 加载数据（假设数据保存在train.json中，每行一个样本）
def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            # 构造对话格式
            conversation = [
                {"role": "user", "content": f"{sample['instruction']}\n{sample['input']}"},
                {"role": "assistant", "content": sample["output"]}
            ]
            data.append({"conversations": conversation})
    return Dataset.from_list(data)

# 加载训练集
train_dataset = load_data("train.json")



# 模型名称
model_name = "Qwen/Qwen2.5-7b-Instruct"

# 1. 配置4bit量化（可选，根据GPU显存调整）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"  # 右侧padding（Qwen推荐）
)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配设备
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.enable_input_require_grads()  # 启用梯度计算（LoRA需要）


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # LoRA秩（越大能力越强，显存消耗越高）
    lora_alpha=32,  # 缩放因子
    target_modules=[  # Qwen2.5-7b的注意力层和MLP层（关键！需匹配模型结构）
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",  # 不训练偏置
    task_type="CAUSAL_LM"
)

# 转换模型为LoRA模型（仅训练LoRA参数）
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例（通常<1%）

from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./qwen2.5-7b-ner-lora",  # 模型保存路径
    per_device_train_batch_size=2,  # 单卡batch size（根据显存调整）
    gradient_accumulation_steps=4,  # 梯度累积（等效增大batch size）
    learning_rate=2e-4,  # LoRA推荐学习率（高于全量微调）
    num_train_epochs=3,  # 训练轮次
    logging_steps=10,  # 日志打印间隔
    save_steps=100,  # 模型保存间隔
    warmup_ratio=0.05,  # 学习率热身比例
    lr_scheduler_type="cosine",  # 学习率调度器
    fp16=True,  # 启用fp16混合精度（需GPU支持）
    report_to="none",  # 不使用wandb等报告工具
    optim="paged_adamw_8bit"  # 8bit优化器（节省显存）
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    dataset_text_field="conversations",  # 数据集中对话字段名称
    max_seq_length=1024,  # 最大序列长度（根据数据调整）
    tokenizer=tokenizer,
    args=training_args,
    packing=False,  # 不打包样本（对话数据通常不打包）
)

# 开始训练
trainer.train()

# 保存LoRA权重
trainer.save_model("./qwen2.5-7b-ner-lora-final")