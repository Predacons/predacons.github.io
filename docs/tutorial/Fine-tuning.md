---
sidebar_position: 1
---
# Fine-tuning with Predacons

This tutorial demonstrates how to fine-tune a pre-trained language model using the Predacons library. We will use the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model as an example.

## Prerequisites

*   A machine with a GPU and CUDA installed (recommended).
*   Python 3.8 or higher.
*   The Predacons library installed (`pip install predacons`).
*   Hugging Face Transformers library installed (`pip install transformers datasets trl peft`).

## Steps

1.  **Import Libraries:**

    First, import the necessary libraries:

```python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU device

    import torch
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("No GPU available")

    import predacons
    from datasets import load_dataset
    

```

2.  **Load Dataset:**

    Load a dataset for fine-tuning. Here, we use `SkunkworksAI/reasoning-0.01` dataset

```python

    ds = load_dataset("SkunkworksAI/reasoning-0.01")
    

```

3.  **Define Model Path:**

    Specify the path to the pre-trained model you want to fine-tune.

```python

    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    

```

4.  **Configure Training Parameters:**

    Set up the training parameters such as output directory, batch size, number of epochs, and save steps.

```python

    output_dir = "pico_r1"
    overwrite_output_dir = False
    per_device_train_batch_size = 1
    num_train_epochs = 10
    save_steps = 50
    

```

5.  **Initialize and Run Trainer:**

    Use the `predacons.trainer` function to initialize the trainer and start the fine-tuning process. This example uses 4-bit quantization and LoRA.

```python

    trainer = predacons.trainer(
        use_legacy_trainer = False,
        train_dataset=ds,
        model_name = model_path,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        trust_remote_code = False,
        resume_from_checkpoint = False,
        auto_quantize = "4bit",
        auto_lora_config = True
    )

    trainer.train()
    

```

    **Parameters Explanation:**

    *   `use_legacy_trainer`:  Flag to use the legacy trainer implementation. Set to `False` for the newer implementation.
    *   `train_dataset`: The dataset to use for training.
    *   `model_name`: The name or path of the pre-trained model.
    *   `output_dir`: The directory where the fine-tuned model will be saved.
    *   `overwrite_output_dir`: Whether to overwrite the output directory if it exists.
    *   `per_device_train_batch_size`: The batch size per GPU.
    *   `num_train_epochs`: The number of training epochs.
    *   `save_steps`:  The number of steps between saving checkpoints.
    *   `trust_remote_code`:  Whether to trust remote code when loading the model.
    *   `resume_from_checkpoint`: Whether to resume training from a checkpoint.
    *   `auto_quantize`:  Enables automatic quantization ("4bit" or "8bit").
    *   `auto_lora_config`:  Enables automatic LoRA configuration.

6.  **Save the Fine-Tuned Model:**

    The `trainer.train()` method saves the fine-tuned model to the specified `output_dir`.

