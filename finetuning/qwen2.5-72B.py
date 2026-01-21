import os

os.environ["CUDA_VISIBLE_DEVICES"] = "30"

from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

from datasets import load_dataset

ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
dataset = ds["train"]

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen2.5",
)

def formatting_prompts_func(row):
    
    # instruction = {
    #     "content": row["instruction"],
    #     "role": "user"
    # }
    
    # output = {
    #     "content": row["output"],
    #     "role": "assistant"
    # }

    # convos = [instruction, output]
    # texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]

    instruction = row["instruction"]
    output = row["output"]
    texts = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>system\n{output}<|im_end|>\n"
    
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func)

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10,
        # max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        dataset_num_proc=2,
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>system\n",
)

trainer_stats = trainer.train()

model.save_pretrained_gguf("saved-models", tokenizer, quantization_method = "q4_k_m")
