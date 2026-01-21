#!/usr/bin/env python
# coding=utf-8
import os
import math
import logging
import torch
import datasets
import transformers
import numpy as np
import sys
import signal
import time
import atexit
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
import arguments

# Global file handler for immediate flushing
file_handler = None

# Set up logging to file with immediate flushing
def setup_logging(log_file_path):
    global file_handler
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a file handler with immediate flushing
    file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode in case of restart
    file_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
                                 datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    
    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    
    # Add a stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    # Set up signal handlers for graceful shutdown
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, handle_exit_signal)
    
    # Register exit handler
    atexit.register(cleanup_logging)
    
    return root_logger

# Handle exit signals
def handle_exit_signal(signum, frame):
    sig_name = signal.Signals(signum).name
    root_logger = logging.getLogger()
    root_logger.warning(f"Received signal {sig_name} ({signum}). Shutting down gracefully.")
    # Force flush logs
    flush_logs()
    # Exit with non-zero code to indicate abnormal termination
    sys.exit(1)

# Flush logs immediately
def flush_logs():
    global file_handler
    if file_handler:
        file_handler.flush()
        os.fsync(file_handler.stream.fileno())

# Clean up logging on exit
def cleanup_logging():
    root_logger = logging.getLogger()
    root_logger.info("Cleaning up logging before exit")
    flush_logs()

# Function to periodically flush logs during training
def periodic_flush():
    flush_logs()

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "24,25,26,27,28,29,30,31" #15,16,17,18,19,21,22,23

def main():
    # Start time measurement
    start_time = time.time()
    
    # Parse arguments
    args = arguments.parse_args()
    
    # Set up log file path
    log_file = os.path.join(args.output_dir, "training_log.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging before we use accelerate
    root_logger = setup_logging(log_file)
    root_logger.info("=" * 50)
    root_logger.info(f"Starting training run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    root_logger.info(f"Log file: {log_file}")
    
    try:
        # Initialize the accelerator
        root_logger.info("Initializing accelerator")
        accelerator = Accelerator(
            mixed_precision="fp16" if args.fp16 else "no",
            log_with="tensorboard",
            project_dir=args.logging_dir
        )
        
        # Now get accelerate logger AFTER accelerator is initialized
        from accelerate.logging import get_logger
        logger = get_logger(__name__)
        
        # Log important info
        logger.info(accelerator.state, main_process_only=True)
        logger.info(f"Arguments: {vars(args)}", main_process_only=True)
        flush_logs()  # Flush immediately after config logging

        # Set seed for reproducibility
        if args.seed is not None:
            set_seed(args.seed)
            logger.info(f"Set random seed to {args.seed}")
        
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset_name}")
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        flush_logs()
        
        # Split datasets if train and test splits don't already exist
        train_test_split = raw_datasets["train"].train_test_split(
            test_size=args.validation_split_percentage / 100
        )
        raw_datasets["train"] = train_test_split["train"]
        raw_datasets["validation"] = train_test_split["test"]
        logger.info(f"Dataset loaded: {len(raw_datasets['train'])} training examples, {len(raw_datasets['validation'])} validation examples")
        flush_logs()

        logger.info(f"Loading tokenizer from: {args.tokenizer_name if args.tokenizer_name else args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Vocabulary size: {len(tokenizer)}")
        flush_logs()

        # Loading model with appropriate error handling
        try:
            if args.use_checkpoint:
                # Initialize model from base model
                logger.info(f"Loading base model from: {args.model_name_or_path}")
                model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
                # Load checkpoint from safetensors
                logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
                state_dict = load_file(args.checkpoint_path)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
            else:
                # Load model directly from Hugging Face
                logger.info(f"Loading model from: {args.model_name_or_path}")
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
                logger.info("Model loaded successfully")
            flush_logs()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Resize token embeddings if needed
        model.resize_token_embeddings(len(tokenizer))
        
        # Preprocessing function for dataset
        def tokenize_function(examples):
            prompts = examples['prompt']
            # Tokenize
            result = tokenizer(prompts, padding="max_length", truncation=True, max_length=1024)
            result["labels"] = result["input_ids"].copy()
            return result
        
        # Process dataset
        logger.info("Processing datasets...")
        flush_logs()
        
        try:
            processed_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                desc="Tokenizing and formatting dataset",
            )
            train_dataset = processed_datasets["train"]
            eval_dataset = processed_datasets["validation"]
            logger.info("Dataset processing complete")
            flush_logs()
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
        
        def pad_data(examples):
            try:
                input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
                labels = [torch.tensor(ex["labels"]) for ex in examples]
                attention_mask = [torch.tensor(ex["attention_mask"]) for ex in examples]
                padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                padded_labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
                padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0) # Assuming 0 for attention mask padding
                return {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "attention_mask": padded_attention_mask,
                }
            except Exception as e:
                logger.error(f"Error in padding data: {str(e)}")
                raise
        
        # Use custom collator
        data_collator = pad_data
        
        # Create data loaders
        try:
            logger.info("Creating data loaders...")
            train_dataloader = DataLoader(
                train_dataset.with_format("torch"), # Set format to PyTorch tensors
                shuffle=True,
                collate_fn=data_collator,
                batch_size=args.per_device_train_batch_size,
            )
            
            eval_dataloader = DataLoader(
                eval_dataset.with_format("torch"), # Set format to PyTorch tensors
                collate_fn=data_collator,
                batch_size=args.per_device_eval_batch_size,
            )
            logger.info("Data loaders created successfully")
            flush_logs()
        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            raise
        
        # Optimizer
        logger.info(f"Setting up optimizer with learning rate: {args.learning_rate}")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # Calculate training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.max_train_steps if args.max_train_steps else args.num_train_epochs * num_update_steps_per_epoch
        
        # Learning rate scheduler
        logger.info("Setting up learning rate scheduler")
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        # Prepare model, optimizer, dataloaders with accelerator
        logger.info("Preparing model, optimizer, and dataloaders with accelerator")
        try:
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
            )
            logger.info("Accelerator preparation complete")
            flush_logs()
        except Exception as e:
            logger.error(f"Error in accelerator preparation: {str(e)}")
            raise
        
        # Total batch size for logging
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        
        # Create checkpoint save directory if it doesn't exist
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        # Log info
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        flush_logs()
        
        # Initialize tracking variables
        completed_steps = 0
        starting_epoch = 0
        best_eval_loss = float('inf')
        last_flush_time = time.time()
        flush_interval = 60  # Flush logs every 60 seconds
        
        # State file to track progress
        state_file = os.path.join(args.output_dir, "training_state.txt")
        
        # Function to save state
        def save_state():
            if accelerator.is_main_process:
                with open(state_file, 'w') as f:
                    f.write(f"epoch={epoch}\n")
                    f.write(f"step={completed_steps}\n")
                    f.write(f"best_eval_loss={best_eval_loss}\n")
                    f.write(f"timestamp={time.time()}\n")
                logger.info(f"Saved training state to {state_file}")
                flush_logs()
        
        # Try to load previous state if exists
        if os.path.exists(state_file) and args.resume_from_checkpoint:
            logger.info(f"Found previous training state at {state_file}")
            try:
                with open(state_file, 'r') as f:
                    state_data = {}
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            state_data[key] = value
                
                starting_epoch = int(state_data.get('epoch', 0))
                completed_steps = int(state_data.get('step', 0))
                best_eval_loss = float(state_data.get('best_eval_loss', float('inf')))
                logger.info(f"Resuming from epoch {starting_epoch}, step {completed_steps}, best_eval_loss {best_eval_loss}")
                flush_logs()
            except Exception as e:
                logger.warning(f"Error loading state file: {str(e)}. Starting from beginning.")
                starting_epoch = 0
                completed_steps = 0
                best_eval_loss = float('inf')
        
        # Training loop
        logger.info("Starting training loop")
        flush_logs()
        
        try:
            for epoch in range(starting_epoch, args.num_train_epochs):
                logger.info(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
                model.train()
                total_loss = 0
                epoch_start_time = time.time()
                
                # Create progress bar for main process only
                if accelerator.is_local_main_process:
                    progress_bar = tqdm(range(num_update_steps_per_epoch), desc=f"Epoch {epoch+1}")
                
                for step, batch in enumerate(train_dataloader):
                    # Check if we need to flush logs
                    current_time = time.time()
                    if current_time - last_flush_time > flush_interval:
                        periodic_flush()
                        last_flush_time = current_time
                        
                        # Also save state periodically
                        save_state()
                    
                    # Forward pass
                    try:
                        outputs = model(**batch)
                        loss = outputs.loss
                        
                        # Accumulate loss for logging
                        total_loss += loss.detach().float()
                        
                        # Backward pass with gradient accumulation
                        loss = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss)
                    except Exception as e:
                        logger.error(f"Error in forward/backward pass at step {step}, epoch {epoch}: {str(e)}")
                        flush_logs()
                        raise
                    
                    # Update model parameters every args.gradient_accumulation_steps
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        try:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            
                            if accelerator.is_local_main_process:
                                progress_bar.update(1)
                            
                            completed_steps += 1
                        except Exception as e:
                            logger.error(f"Error in optimizer step at step {step}, epoch {epoch}: {str(e)}")
                            flush_logs()
                            raise
                        
                        # Log metrics
                        if completed_steps % args.logging_steps == 0:
                            try:
                                avg_loss = accelerator.gather(total_loss).mean().item() / args.logging_steps / args.gradient_accumulation_steps
                                perplexity = math.exp(avg_loss)
                                
                                logger.info(f"Step {completed_steps}/{max_train_steps}: train_loss: {avg_loss:.4f}, "
                                          f"train_perplexity: {perplexity:.4f}, "
                                          f"lr: {optimizer.param_groups[0]['lr']:.8f}, "
                                          f"epoch: {epoch+1}/{args.num_train_epochs}")
                                
                                accelerator.log(
                                    {
                                        "train_loss": avg_loss,
                                        "train_perplexity": perplexity,
                                        "learning_rate": optimizer.param_groups[0]["lr"],
                                        "epoch": epoch,
                                    },
                                    step=completed_steps,
                                )
                                total_loss = 0
                                flush_logs()
                            except Exception as e:
                                logger.error(f"Error logging metrics at step {step}, epoch {epoch}: {str(e)}")
                                flush_logs()
                        
                        # Break if we've reached max steps
                        if completed_steps >= max_train_steps:
                            logger.info(f"Reached max steps ({max_train_steps}). Stopping training.")
                            break
                        
                        # Evaluate at specified interval
                        if (args.evaluation_strategy == "steps" and completed_steps % args.eval_steps == 0) or \
                           (step == len(train_dataloader) - 1 and args.evaluation_strategy == "epoch"):
                            logger.info(f"Running evaluation at step {completed_steps}")
                            flush_logs()
                            
                            try:
                                model.eval()
                                eval_losses = []
                                
                                for eval_batch in eval_dataloader:
                                    with torch.no_grad():
                                        outputs = model(**eval_batch)
                                    eval_losses.append(accelerator.gather(outputs.loss))
                                
                                eval_losses = torch.cat(eval_losses)
                                eval_loss = torch.mean(eval_losses).item()
                                eval_perplexity = math.exp(eval_loss)
                                
                                logger.info(f"Step {completed_steps}: eval_loss: {eval_loss:.4f}, eval_perplexity: {eval_perplexity:.4f}")
                                
                                accelerator.log(
                                    {
                                        "eval_loss": eval_loss,
                                        "eval_perplexity": eval_perplexity,
                                        "epoch": epoch,
                                    },
                                    step=completed_steps,
                                )
                                flush_logs()
                                
                                # Save best model
                                if eval_loss < best_eval_loss:
                                    best_eval_loss = eval_loss
                                    logger.info(f"New best evaluation loss: {best_eval_loss:.4f}")
                                    
                                    if accelerator.is_main_process:
                                        best_model_dir = os.path.join(args.output_dir, "best_model")
                                        os.makedirs(best_model_dir, exist_ok=True)
                                        unwrapped_model = accelerator.unwrap_model(model)
                                        unwrapped_model.save_pretrained(
                                            best_model_dir,
                                            save_function=accelerator.save,
                                            safe_serialization=True
                                        )
                                        tokenizer.save_pretrained(best_model_dir)
                                        logger.info(f"Saved best model to {best_model_dir}")
                                        flush_logs()
                                
                                model.train()
                            except Exception as e:
                                logger.error(f"Error during evaluation at step {step}, epoch {epoch}: {str(e)}")
                                flush_logs()
                                # Continue training even if evaluation fails
                                model.train()

                        # Save model checkpoint at specified interval
                        if completed_steps % args.save_steps == 0:
                            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                            accelerator.wait_for_everyone()
                            
                            # Save checkpoint
                            if accelerator.is_main_process:
                                try:
                                    logger.info(f"Saving checkpoint to {checkpoint_dir}")
                                    os.makedirs(checkpoint_dir, exist_ok=True)
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(
                                        checkpoint_dir,
                                        save_function=accelerator.save,
                                        safe_serialization=True  # Use safetensors for storage
                                    )
                                    tokenizer.save_pretrained(checkpoint_dir)
                                    logger.info(f"Checkpoint saved to {checkpoint_dir}")
                                    
                                    # Save state as well
                                    save_state()
                                    flush_logs()
                                except Exception as e:
                                    logger.error(f"Error saving checkpoint at step {step}, epoch {epoch}: {str(e)}")
                                    flush_logs()
                
                # End of epoch
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Completed epoch {epoch+1}/{args.num_train_epochs} in {epoch_time:.2f} seconds")
                flush_logs()
                
                # Save final model at the end of each epoch (if evaluation strategy is epoch-based)
                if args.evaluation_strategy == "epoch":
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        try:
                            output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
                            logger.info(f"Saving end-of-epoch model to {output_dir}")
                            os.makedirs(output_dir, exist_ok=True)
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir,
                                save_function=accelerator.save,
                                safe_serialization=True
                            )
                            tokenizer.save_pretrained(output_dir)
                            logger.info(f"End-of-epoch model saved to {output_dir}")
                            
                            # Update state file
                            save_state()
                            flush_logs()
                        except Exception as e:
                            logger.error(f"Error saving epoch model: {str(e)}")
                            flush_logs()
            
            # Save final model
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                try:
                    logger.info(f"Training complete. Saving final model to {args.output_dir}")
                    # Unwrap model
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Save using safetensors
                    unwrapped_model.save_pretrained(
                        args.output_dir,
                        save_function=accelerator.save,
                        safe_serialization=True
                    )
                    # Save tokenizer
                    tokenizer.save_pretrained(args.output_dir)
                    logger.info(f"Final model saved to {args.output_dir}")
                    flush_logs()
                    
                    # Push to hub if specified
                    if args.push_to_hub:
                        try:
                            repo_name = args.hub_model_id if args.hub_model_id else os.path.basename(args.output_dir)
                            repo_name = repo_name.replace('/', '--')
                            logger.info(f"Pushing model to Hugging Face Hub: {repo_name}")
                            from huggingface_hub import HfApi
                            api = HfApi()
                            api.upload_folder(
                                folder_path=args.output_dir,
                                repo_id=repo_name,
                                token=args.hub_token,
                            )
                            logger.info("Model successfully pushed to Hugging Face Hub")
                            flush_logs()
                        except Exception as e:
                            logger.error(f"Error pushing to hub: {str(e)}")
                            flush_logs()
                except Exception as e:
                    logger.error(f"Error saving final model: {str(e)}")
                    flush_logs()
                
                # Training summary
                total_time = time.time() - start_time
                logger.info("=" * 50)
                logger.info(f"Training completed in {total_time:.2f} seconds")
                logger.info(f"Best validation loss: {best_eval_loss:.4f}")
                logger.info(f"Final training step: {completed_steps}")
                logger.info("=" * 50)
                flush_logs()
                
                # Remove state file as training completed successfully
                if os.path.exists(state_file):
                    os.remove(state_file)
                    logger.info(f"Removed training state file as training completed successfully")
                    flush_logs()
                
        except Exception as e:
            root_logger.error(f"Error during training: {str(e)}")
            root_logger.error("Training interrupted unexpectedly!")
            import traceback
            root_logger.error(traceback.format_exc())
            flush_logs()
            # Save state for potential resume
            save_state()
            # Re-raise the exception
            raise
            
    except Exception as e:
        root_logger.error(f"Fatal error: {str(e)}")
        import traceback
        root_logger.error(traceback.format_exc())
        flush_logs()
        raise
    
    finally:
        # Ensure logs are flushed before exiting
        logging.getLogger().info("Training script finishing - ensuring logs are flushed")
        flush_logs()

if __name__ == "__main__":
    main()