# evaluate_code.py - Script to evaluate generated code with CodeBLEU
import os
import torch
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

torch.manual_seed(42)
torch.mps.manual_seed(42)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def unrepeated_output(x):
    """
    Function to prevent repetitive outputs in generated text.
    Replace this with your actual implementation.
    """
    # Check if x is a torch.Tensor or a list
    assert isinstance(x, (torch.Tensor, list)), "Input must be a torch.Tensor or a list"
    
    # If the last token is EOS token, return as is
    # Assuming tokenizer is already imported or passed in
    if x[-1] == tokenizer.eos_token_id:
        return x
    
    # Initialize variables
    idx_bot = {}
    lengths = []
    num = 0
    len_between = 0
    l = x[-1]
    
    # Find occurrences of the last token
    for i in range(2, len(x)):
        len_between += 1
        if x[-i] == l:
            num += 1
            idx_bot[num] = i
            lengths.append(len_between)
            len_between = 0
    
    # Check for pattern breaks
    if len(lengths) >= 2:
        for i in range(2, min(len(lengths) + 1, num + 1)):
            if lengths[0] != lengths[i-1]:
                return x[:-idx_bot[i]]
    
    # If no pattern break is found, return original sequence
    return x

def load_data_from_files(data_dir, part=128, seq=0):
    """
    Load prompts and expected outputs from files.
    
    Args:
        data_dir: Directory containing the data files
    
    Returns:
        prompts: List of prompt strings
        expected_outputs: List of expected output strings
    """
    data_dir = Path(data_dir)
    prompt_dir = data_dir / "prompts"
    expected_dir = data_dir / "expected"
    
    prompts = []
    expected_outputs = []
    
    # Get list of prompt files
    prompt_files = sorted(list(prompt_dir.glob("*.txt")), 
                         key=lambda x: int(x.stem.split('_')[1]))
    
    # Get list of expected output files
    expected_files = sorted(list(expected_dir.glob("*.txt")), 
                           key=lambda x: int(x.stem.split('_')[1]))
    
    # Load prompts
    for file_path in prompt_files[part*(seq):part*(seq+1)]:
        with open(file_path, "r") as f:
            prompts.append(f.read())
    
    # Load expected outputs
    for file_path in expected_files[part*(seq):part*(seq+1)]:
        with open(file_path, "r") as f:
            expected_outputs.append(f.read())
    
    logger.info(f"Loaded {len(prompts)} prompts and {len(expected_outputs)} expected outputs from {data_dir}")
    return prompts, expected_outputs

def generate_outputs(model, tokenizer, prompts, max_length=256):
    """
    Generate outputs for the given prompts using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        max_length: Maximum length for generation
        
    Returns:
        List of generated code strings
    """
    logger.info(f"Generating outputs for {len(prompts)} prompts...")
    generated_outputs = []
    
    for i, prompt in enumerate(tqdm(prompts)):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate 
        ## min_length = 20, num_beams = 12,top_k = 15,show_output = False
        with torch.no_grad():
            # output_ids = model.generate(
            #     inputs.input_ids,
            #     max_length=max_length,
            #     min_length = 20,
            #     num_beams = 12,
            #     num_return_sequences=1,
            #     top_k = 15,
            #     temperature=0.7,
            #     top_p=0.95,
            #     do_sample=True,
            #     pad_token_id=tokenizer.eos_token_id
            # )
            output_ids = model.generate(
                                        inputs.input_ids,
                                        max_new_tokens=256, ## had to reduce due to out of memory errors
                                        min_length=32,
                                        num_beams=4,
                                        early_stopping=True,
                                        pad_token_id=tokenizer.eos_token_id
                                    )

        
        # Apply your unrepeated_output function to remove repetitions
        output_ids = unrepeated_output(output_ids[0])
        
        # Decode to text
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Remove the prompt from the generated text if needed
        prompt_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt_decoded):
            generated_text = generated_text[len(prompt_decoded):].strip()
        
        generated_outputs.append(generated_text)
    
    logger.info("Output generation complete")
    return generated_outputs

def save_outputs(generated_outputs, output_dir="generated_outputs"):
    """
    Save generated outputs to files.
    
    Args:
        generated_outputs: List of generated output strings
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each generated output to a separate file
    for i, output in enumerate(generated_outputs):
        with open(output_dir / f"generated_{i}.txt", "w") as f:
            f.write(output)
    
    # Save all outputs to a single file
    with open(output_dir / "all_generated.txt", "w") as f:
        for i, output in enumerate(generated_outputs):
            f.write(f"==== GENERATED {i} ====\n")
            f.write(output)
            f.write("\n\n")
    
    logger.info(f"Saved {len(generated_outputs)} generated outputs to {output_dir}")
    return output_dir

def calculate_perplexity(model, tokenizer, text):
    """
    Calculate perplexity of the given text using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to evaluate
        
    Returns:
        Perplexity score (float)
    """
    if not text.strip():
        raise ValueError("Empty string provided for perplexity calculation.")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return torch.exp(loss).item()

def calculate_metrics(expected_outputs, generated_outputs, model, tokenizer):
    """
    Calculate metrics for the generated outputs.
    
    Args:
        expected_outputs: List of expected output strings
        generated_outputs: List of generated output strings
        model: The language model for perplexity calculation
        tokenizer: The tokenizer
        
    Returns:
        Dictionary of metrics
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Calculate perplexity
    perplexities = []
    for output in generated_outputs:
        try:
            ppl = calculate_perplexity(model, tokenizer, output)
            perplexities.append(ppl)
        except Exception as e:
            logger.warning(f"Error calculating perplexity: {e}")
            perplexities.append(float('nan'))
    
    # Calculate BLEU
    tokenized_refs = [[r.split()] for r in expected_outputs]
    tokenized_cands = [c.split() for c in generated_outputs]
    
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(tokenized_refs, tokenized_cands, smoothing_function=smooth)
    
    # Calculate CodeBLEU
    try:
        # Import the CodeBLEU module we set up
        from codebleu_module.calc_codebleu import calculate_codebleu
        
        codebleu_result = calculate_codebleu(
            references=expected_outputs,
            candidates=generated_outputs,
            lang="python",
            weights={
                'ngram_match_weight': 0.25,
                'weighted_ngram_match_weight': 0.25,
                'syntax_match_weight': 0.25,
                'dataflow_match_weight': 0.25
            }
        )
    except Exception as e:
        logger.warning(f"Error calculating CodeBLEU: {e}")
        codebleu_result = {'codebleu': float('nan')}
    
    # Collect metrics
    metrics = {
        'bleu': bleu_score,
        'codebleu': codebleu_result.get('codebleu', float('nan')),
        'average_perplexity': np.nanmean(perplexities),
        'perplexities': perplexities
    }
    
    # Add CodeBLEU components if available
    for key, value in codebleu_result.items():
        if key != 'codebleu':
            metrics[key] = value
    
    return metrics

def save_metrics(metrics, output_dir="evaluation_results"):
    """
    Save metrics to a file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save the metrics file
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("=== Overall Metrics ===\n")
        f.write(f"BLEU Score: {metrics['bleu']:.4f}\n")
        f.write(f"CodeBLEU Score: {metrics['codebleu']:.4f}\n")
        f.write(f"Average Perplexity: {metrics['average_perplexity']:.4f}\n\n")
        
        f.write("=== CodeBLEU Component Scores ===\n")
        for key, value in metrics.items():
            if key not in ['bleu', 'codebleu', 'average_perplexity', 'perplexities'] and not isinstance(value, list):
                f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("=== Per-Sample Metrics ===\n")
        for i, ppl in enumerate(metrics['perplexities']):
            f.write(f"Sample {i}:\n")
            f.write(f"  Perplexity: {ppl:.4f}\n")
    
    logger.info(f"Saved metrics to {output_dir / 'metrics.txt'}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated code with CodeBLEU")
    parser.add_argument("--data_dir", type=str, default="data_files", help="Directory containing input data files")
    parser.add_argument("--model_path", type=str,\
                        default="/Users/jagathkumarreddyk/Documents/GitHub/CodeParrot2.0/finetuning/ft3/fine_tuned_gpt2/best_model",help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for generation")
    
    args = parser.parse_args()
    
    # Select device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer (GPT2)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info(f"Loading GPT-2 model from {args.model_path}")
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    # Set pad token if missing (common for GPT2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attach tokenizer globally for use in unrepeated_output
    globals()['tokenizer'] = tokenizer

    # Load data
    prompts, expected_outputs = load_data_from_files(args.data_dir)

    # Generate outputs
    generated_outputs = generate_outputs(model, tokenizer, prompts, args.max_length)

    # Save generated outputs
    save_outputs(generated_outputs, args.output_dir)

    # Calculate metrics
    metrics = calculate_metrics(expected_outputs, generated_outputs, model, tokenizer)

    # Save metrics
    save_metrics(metrics, args.output_dir)
 
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"CodeBLEU Score: {metrics['codebleu']:.4f}")
    print(f"Average Perplexity: {metrics['average_perplexity']:.4f}")
    print(f"Full results saved to: {args.output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")