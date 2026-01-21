import os

os.environ["CUDA_VISIBLE_DEVICES"] = "31"

from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = 1024,
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

ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
dataset = ds["train"]


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
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
    texts = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>\n"
    
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
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

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()


model.save_pretrained_gguf("saved-models", tokenizer, quantization_method = "q4_k_m")
