#!/bin/bash
export #!/bin/bash
sleep 18000  # 等待 1 小时（3600 秒）
model_type='qwen'
mode='sft_train'
sft_type='adalora'
python main_v2.py \
    --mode ${mode} \
    --model_type ${model_type} \
    --model_path /root/LLM/Qwen2.5-7B-Instruct \
    --train_file_dir /root/code/LLM/sft/CHIP2023-PromptCBLUE/small/train \
    --validation_file_dir /root/code/LLM/sft/CHIP2023-PromptCBLUE/small/dev \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --fine_tuning_type ${sft_type} \
    --lora_target q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
    --output_dir checkpoint/${model_type}_${sft_type}_${mode} \
    --learning_rate 1e-3 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 20 \
    --prompt_template qwen
MY_VAR  # 确保脚本执行完毕后，该变量仍可被访问