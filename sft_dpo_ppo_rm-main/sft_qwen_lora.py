# 1.加载数据
# 2.初始化模型
# 3.初始化peft
# 4.模型拼接
# 5.定义训练函数
# 6.定义评估指标
from transformers import BitsAndBytesConfig
import os
from types import MethodType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationConfig
from peft import LoraConfig, AdaLoraConfig,
from peft import TaskType, get_peft_model


def preprocess_train_supervised_fine_tuning_dataset(examples):
    # ChatGLM1: https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py#L323
    # ChatGLM2: https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L171
    # Baichuan: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/tokenization_baichuan.py#L152
    # internlm: https://huggingface.co/internlm/internlm-chat-7b/blob/main/tokenization_internlm.py#L179
    # moss: https://huggingface.co/fnlp/moss-moon-003-sft/blob/main/tokenization_moss.py#L226
    # Llama: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L296
    inputs_list = []
    attention_mask_list = []
    labels_list = []
    if self.training_args.use_firefly_loss:
        for prompt, answer in self.format_example(examples, False):
            source_ids = []
            labels = []
            for i, sentence in enumerate(prompt):
                if i % 2 == 0:
                    sentence_ids = self.tokenizer.encode(text=sentence, add_special_tokens=False)
                    source_ids.extend(sentence_ids)
                    labels.extend([self.label_pad_token_id] * (len(sentence_ids)))
                else:
                    sentence_ids = self.tokenizer.encode(text=sentence, add_special_tokens=False)
                    sentence_ids = sentence_ids + [self.tokenizer.eos_token_id]
                    source_ids.extend(sentence_ids)
                    labels.extend(sentence_ids)
            target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
                labels = self.tokenizer.build_inputs_with_special_tokens(labels)
                context_length = len(labels)
                labels = self.transfer_front_tail_to_label_pad_token_id(labels)
                labels = labels + input_ids[context_length:]
            else:
                input_ids = source_ids + target_ids + [self.tokenizer.eos_token_id]
                if self.tokenizer.bos_token_id is not None:
                    input_ids = [self.tokenizer.bos_token_id] + input_ids
                    labels = [self.label_pad_token_id] + labels
                labels = labels + target_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                input_ids = input_ids[:self.data_args.max_input_token]
                labels = labels[:self.data_args.max_input_token]
                attention_mask = attention_mask[:self.data_args.max_input_token]
            inputs_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
    else:
        for prompt, answer in self.format_example(examples):
            source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
                context_length = len(self.tokenizer.build_inputs_with_special_tokens(source_ids))
                labels = [self.label_pad_token_id] * context_length + input_ids[context_length:]
            else:
                input_ids = source_ids + target_ids + [self.tokenizer.eos_token_id]
                context_length = len(source_ids)
                if self.tokenizer.bos_token_id is not None:
                    input_ids = [self.tokenizer.bos_token_id] + input_ids
                    context_length = context_length + 1
                labels = [self.label_pad_token_id] * context_length + target_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                input_ids = input_ids[:self.data_args.max_input_token]
                labels = labels[:self.data_args.max_input_token]
                attention_mask = attention_mask[:self.data_args.max_input_token]
            inputs_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
    return {'input_ids': inputs_list, 'attention_mask': attention_mask_list, 'labels': labels_list}


def main():
    # load_base_model
    config_kwargs = {'cache_dir': cache_dir,
                     'torch_dtype': torch_dtype}
    config_kwargs['device_map'] = 'auto'
    config_kwargs['use_flash_attention_2'] = True
    config_kwargs['fp16'] = True

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, allowed_special='all',
                                              padding_side=padding_side, trust_remote_code=True)
    model.generate = MethodType(PreTrainedModel.generate, model)
    model.config.use_dynamic_ntk = True
    if os.path.exists(model_path + '/generation_config.json'):
        model.generation_config = GenerationConfig.from_pretrained(model_path)

    logger.info(f'Target liners for lora: {target_modules}')
    # load loramodel
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # set train environments
    model.config.use_cache = False # 训练时关闭，推理时打开，目的是加速推理

    if gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        # turn off when gradient checkpointing is enabled
    logger.info(f'Model struct:\n{model}')



    # load_data
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        return_tensors='pt',
        label_pad_token_id=label_pad_token_id,
    )

    train_dataset, eval_dataset = self.data_manager.prepare_dataset()
