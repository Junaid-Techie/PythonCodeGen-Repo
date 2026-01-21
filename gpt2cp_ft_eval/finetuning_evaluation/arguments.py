import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning a GPT-2 model on Python code instructions dataset")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="iamtarun/python_code_instructions_18k_alpaca",
                        help="The name of the dataset to use")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="The configuration name of the dataset to use")
    parser.add_argument("--validation_split_percentage", type=int, default=5,
                        help="The percentage of the train set used as validation set")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="The maximum total input sequence length after tokenization")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, # change it needed 
                        help="The number of processes for preprocessing")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="Use slow tokenizer (not backed by the 🤗 Tokenizers library)")
    parser.add_argument("--checkpoint_path", type=str, default="/home/students/jkatama/CodeParrot2.0/scripts/step_51200/model.safetensors",
                        help="Path to model checkpoint in safetensors format")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Whether to use the checkpoint specified")
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Batch size per GPU for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate to use")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay to use")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Total number of training epochs to perform")
    parser.add_argument("--max_train_steps", type=int, default=15000, # 8840 batches
                        help="Total number of training steps. If provided, overrides num_train_epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, #effective batch size  = 16
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="The scheduler type to use")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for the learning rate scheduler")
    
    # FP16 training arguments
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use fp16 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level")
    
    # Checkpointing arguments
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_gpt2",
                        help="Where to store the final model")
    parser.add_argument("--save_steps", type=int, default=2048,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=16,
                        help="Limit the total amount of checkpoints. Deletes the older checkpoints.")
    
    # Logging arguments
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="Tensorboard log directory")
    parser.add_argument("--logging_steps", type=int, default=128,
                        help="Log every X updates steps")
    
    # Evaluation arguments
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="The evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=512,
                        help="Run evaluation every X steps")
    
    # Dataset formatting
    parser.add_argument("--prompt_column", type=str, default="prompt", #can be removed
                        help="The name of the column containing the prompt")
    parser.add_argument("--response_column", type=str, default="output",
                        help="The name of the column containing the response")
    parser.add_argument("--input_column", type=str, default="input",
                        help="The name of the column containing the input context")
    
    # Additional arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="The name of the repository to keep in sync with the local `output_dir`")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    
    args = parser.parse_args()
    return args