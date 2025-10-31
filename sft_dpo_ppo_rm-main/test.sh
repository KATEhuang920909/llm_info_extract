#!/bin/bash
# export #!/bin/bash
# sleep 18000  # 等待 1 小时（3600 秒）
model_type='qwen'
mode='sft_train'
sft_type='lora'
python main.py \
    --mode ${mode} \
    --model_type ${model_type} \
    --model_path /root/auto-tmp/Qwen2.5-7B-Instruct \
    --train_file_dir /root/code/sft_dpo_ppo_rm-main/datasets/compet_v1/ner_finetuning_train.json \
    --validation_file_dir /root/code/sft_dpo_ppo_rm-main/datasets/compet_v1/ner_finetuning_dev.json  \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5\
    --fine_tuning_type ${sft_type} \
    --lora_target q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
    --output_dir checkpoint/${model_type}_${sft_type}_${mode} \
    --learning_rate 1e-4 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --logging_steps 20 \
    --prompt_template qwen
