# Python Code Generation using LLMs

## Dataset
First using the file `data/get_data.ipynb` download the pretraining and finetuning datasets.

## Pretraining
To run the pretraining code for this project without encountering multi-GPU errors, it's essential to use the `accelerator` module provided by Hugging Face. This module ensures proper hardware utilization and avoids common configuration issues on multi-GPU setups.

### Accelerator Setup
1. Install the Hugging Face `accelerate` module:
    ```bash
    pip install accelerate
    ```
2. Configure `accelerator`:
    ```bash
    accelerate config
    ```
3. Follow the interactive prompts and select:

    * Compute environment: (choose according to your setup, e.g., multi-node or single-node)

    * Distributed training: multi-GPU

    * Number of processes: At least 8 (we recommend 12)

    * Mixed precision training: (choose based on your hardware, typically fp16 for modern GPUs)

    * For any of the settings we have not mentioned choose `NO` if you're using the ML server. Or else choose ones appropriate the your hardware.

### Pretrain Run
```bash
accelerate launch pretraining/sft-trainer.py
```

## Finetuning
The process of tinetuning is straight forward. It requires no additional setup, all the model files will be automatically downloaded by `unsloth`.

### Finetune Run
```bash
python3 finetuning/<model-name>-<size>.py
```

Example -
```bash
python3 finetuning/llama3.2-3B.py
```