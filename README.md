# PythonCodeGen-Repo
------------
This Repository contains the experimental setup and code
------------



-------------
# Python Code Generation using Lightweight LLMs

## Project Overview
Developed Python code generation models using lightweight Large Language Models (LLMs) capable of running locally on consumer hardware. Implemented advanced techniques including continual pretraining, 4-bit quantization, and Parameter-Efficient Fine-Tuning (PEFT) to optimize various model architectures (GPT-2, LLaMA 3.2, Gemma 3, Qwen 2.5) ranging from 124M to 3B parameters. Successfully engineered models that generate high-quality Python code without cloud dependencies, demonstrating expertise in NLP, transformer architectures, and ML optimization techniques.

## Key Contributions

### Model Development & Optimization
- Implemented continual pretraining of GPT-2 124M on 200,000 Python code samples from the CodeParrot dataset, achieving optimal loss minima through statistical optimization techniques
- Applied advanced quantization techniques (4-bit precision) and Parameter-Efficient Fine-Tuning (PEFT) with LoRA to optimize billion-parameter models (1B-3B) for efficient single-GPU training
- Engineered custom prompt templates with Hugging Face Transformers, tailored to each model architecture for optimal code generation performance
- Processed and transformed large-scale code datasets using pandas, numpy, and custom data pipelines
- Fine-tuned models on an 18K Python instruction dataset leveraging specialized loss functions that prioritized syntactic correctness and code compilability

### Experimental Design & Statistical Evaluation
- Designed rigorous experiments to evaluate code generation capabilities across model architectures, using statistical methods to ensure reliable comparisons
- Developed custom evaluation pipelines using pandas, numpy, and scikit-learn to analyze model performance across multiple metrics
- Conducted advanced statistical analysis of cross-entropy loss patterns to identify optimal model configurations and training regimes
- Implemented memory optimization techniques including gradient checkpointing and gradient accumulation to enable training with 1024-token sequences on limited hardware
- Evaluated model performance on diverse code generation tasks ranging from basic functions to complex library implementations, with statistical analysis of generation quality

### Research Findings & Data Analysis
- Conducted comprehensive statistical analysis proving that LLaMA 3.2 3B consistently outperformed other models across multiple code generation scenarios
- Generated visualizations and statistical models demonstrating that architecture design and pretraining quality significantly impact performance, sometimes outweighing raw parameter count
- Developed alternative evaluation methodologies after identifying statistical limitations in traditional metrics (perplexity, CodeBLEU) for code generation tasks
- Created statistical validation showing that reasoning-focused models (Qwen) didn't transfer their reasoning capabilities effectively to programming tasks despite theoretical advantages
- Implemented PyTorch-based monitoring and visualization tools to track model performance across training phases

### Technical Implementation & Infrastructure
- Leveraged PyTorch, TensorFlow, and Hugging Face Transformers for advanced model development and training
- Implemented the Unsloth framework with custom modifications for highly efficient fine-tuning of billion-parameter models
- Engineered scalable data processing pipelines using pandas, numpy, and custom Python tools to handle large-scale code datasets
- Optimized training processes through gradient accumulation and advanced scheduling to maximize performance on limited computational resources (single GPU with 10GB VRAM for Finetuning and Utilized Multi-GPU setup for pretraining)
- Implemented custom logging and monitoring tools for tracking training progress and model performance

## Results & Impact
- Successfully engineered python code generation models.
- Identified key limitations and developed concrete roadmap for future research directions in local code generation models
- Documented Report paper documenting methodology and findings, contributing to the field's understanding of practical tradeoffs between model size, performance in AI code assistants


## Technologies & Skills Demonstrated
**ML/DL**: PyTorch, TensorFlow, Hugging Face Transformers, Parameter-Efficient Fine-Tuning (PEFT), LoRA, Quantization, Accelerate

**Models**: GPT-2, LLaMA 3.2, Gemma 3, Qwen 2.5

**Data Science**: pandas, numpy, scikit-learn, statistical analysis, data visualization, experimental design

**Optimization**: gradient checkpointing, 4-bit quantization, efficient training strategies

**Infrastructure**: Single-GPU optimization, memory-efficient processing, Multi-GPU optimization
