# Train Predacons

This module provides functionalities for training Predacons models. It includes methods for both legacy and standard training approaches, allowing flexibility in how you train your models.

## train
```

def train(*args, **kwargs):


```
Trains the Predacons model. This function uses a more modern training setup than `train_legacy` and offers more flexibility in configuration.  It leverages the Hugging Face `Trainer` or `SFTTrainer` depending on the provided arguments.

**Args:**

-   `*args`:  Arbitrary positional arguments (not currently used).
-   `**kwargs`: Keyword arguments for configuring the training process.  See below for a detailed description of available keyword arguments.

**Keyword Args:**

-   `use_legacy_trainer` (bool): If `True`, uses a legacy training approach. Default is `False`.
-   `model_name` (str): The name or path of the pre-trained model to be used.
-   `train_file_path` (str): Path to the training dataset file.
-   `tokenizer` (Tokenizer): An instance of a tokenizer.
-   `output_dir` (str): Directory where the model and tokenizer will be saved after training.
-   `overwrite_output_dir` (bool): If `True`, overwrite the output directory.
-   `per_device_train_batch_size` (int): Batch size per device during training.
-   `num_train_epochs` (int): Total number of training epochs.
-   `quantization_config` (dict): Configuration for model quantization.
-   `auto_quantize` (str): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression.
-   `trust_remote_code` (bool): If `True`, allows the execution of remote code during model loading.
-   `peft_config` (dict): Configuration for Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA.
-   `auto_lora_config` (bool): If `True`, automatically configures LoRA for the model.
-   `training_args` (TrainingArguments): Configuration arguments for the Hugging Face `Trainer`.
-   `train_file_type` (str): Type of the training file. Supported types are "text", "csv", "json".
-   `train_dataset` (Dataset): A pre-loaded dataset. If provided, `train_file_path` is ignored.
-   `preprcess_function` (callable): A function to preprocess the dataset.
-   `resume_from_checkpoint` (str): Path to a directory containing a checkpoint from which training is to resume.
   `save_steps` (int): Number of steps after which the model is saved.

**Example:**

```

from predacons import predacons
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models/my_trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
)

predacons.train(
    model_name="bert-base-uncased",
    train_file_path="data/my_training_data.txt",
    training_args=training_args,
    trust_remote_code=False
)


```

## trainer
```

def trainer(*args, **kwargs):


```

Prepares a trainer instance for the Predacons model with the provided arguments and keyword arguments.

This function serves as a wrapper that calls the `trainer` method of the `TrainPredacons` class, forwarding all received arguments and keyword arguments. It is designed to configure and return a trainer instance without immediately starting the training process, allowing for further customization or inspection of the trainer configuration before training.

**Args:**

-   `*args`: Arbitrary positional arguments. Currently, this method does not utilize positional arguments but is designed to be flexible for future extensions.
-   `**kwargs`: Arbitrary keyword arguments used for configuring the training process. The supported keywords include:

    -   `use_legacy_trainer` (bool): If `True`, uses a legacy training approach. Default is `False`.
    -   `model_name` (str): The name or path of the pre-trained model to be used.
    -   `train_file_path` (str): Path to the training dataset file.
    -   `tokenizer` (Tokenizer): An instance of a tokenizer.
    -   `output_dir` (str): Directory where the model and tokenizer will be saved after training.
    -   `overwrite_output_dir` (bool): If `True`, overwrite the output directory.
    -   `per_device_train_batch_size` (int): Batch size per device during training.
    -   `num_train_epochs` (int): Total number of training epochs.
    -   `quantization_config` (dict): Configuration for model quantization.
    -   `auto_quantize` (str): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression.
    -   `trust_remote_code` (bool): If `True`, allows the execution of remote code during model loading.
    -   `peft_config` (dict): Configuration for Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA.
    -   `auto_lora_config` (bool): If `True`, automatically configures LoRA for the model.
    -   `training_args` (TrainingArguments): Configuration arguments for the Hugging Face `Trainer`.
    -   `train_file_type` (str): Type of the training file. Supported types are "text", "csv", "json".
    -   `train_dataset` (Dataset): A pre-loaded dataset. If provided, `train_file_path` is ignored.
    -   `preprcess_function` (callable): A function to preprocess the dataset.
    -   `resume_from_checkpoint` (str): Path to a directory containing a checkpoint from which training is to resume.
     `save_steps` (int): Number of steps after which the model is saved.

**Returns:**

-   Returns an instance of `Trainer` or `SFTTrainer`, configured according to the provided arguments. This object is ready to be used for training the model.

**Example Usage:**

```

trainer = predacons.trainer(
    model_name='bert-base-uncased',
    train_file_path='./data/train.txt',
    tokenizer=my_tokenizer,
    output_dir='./model_output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    auto_quantize='4bit'
)


```


## train_legacy
```

def train_legacy(train_file_path, model_name,
                 output_dir,
                 overwrite_output_dir,
                 per_device_train_batch_size,
                 num_train_epochs,
                 save_steps,
                 trust_remote_code=False,
                 resume_from_checkpoint=True):


```
Trains the Predacons model using the legacy training method.

**Args:**

-   `train_file_path` (str): The path to the training data file. This file contains the data used to train the model.
-   `model_name` (str): The name or path of the pre-trained model to be fine-tuned.  This could be a Hugging Face model name or a local directory containing the model.
-   `output_dir` (str): The directory where the trained model and related files will be saved.
-   `overwrite_output_dir` (bool): If `True`, the contents of the `output_dir` will be overwritten if it already exists.  If `False`, training will not proceed if the directory exists.
-   `per_device_train_batch_size` (int): The batch size used for training on each device (CPU or GPU).  This affects memory usage and training speed.
-   `num_train_epochs` (int): The number of times the training loop will iterate over the entire training dataset.
-   `save_steps` (int): The number of training steps between each checkpoint save.  Checkpoints allow you to resume training or evaluate the model at different stages.
-   `trust_remote_code` (bool, optional):  Whether to trust remote code when loading the model.  Set to `True` if using a model that requires executing custom code. Defaults to `False`.
-   `resume_from_checkpoint` (bool, optional): Whether to resume training from the latest checkpoint in the `output_dir`. Defaults to `True`.

**Example:**

```

from predacons import predacons

predacons.train_legacy(
    train_file_path="data/my_training_data.txt",
    model_name="bert-base-uncased",
    output_dir="models/my_trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    trust_remote_code=False
)


```

## trainer_legacy
```

def trainer_legacy(train_file_path, model_name,
                   output_dir,
                   overwrite_output_dir,
                   per_device_train_batch_size,
                   num_train_epochs,
                   save_steps,
                   trust_remote_code=False):


```

This function returns a trainer object using the legacy training setup. This allows for more customized training loops or integration with other training frameworks.

**Args:**

-   `train_file_path` (str): Path to the training data file.
-   `model_name` (str): Name or path of the pre-trained model.
-   `output_dir` (str): Directory to save the trained model.
-   `overwrite_output_dir` (bool): Whether to overwrite the output directory if it exists.
-   `per_device_train_batch_size` (int): Batch size per device during training.
-   `num_train_epochs` (int): Number of training epochs.
-   `save_steps` (int): Number of steps after which the model is saved.
-   `trust_remote_code` (bool, optional): Whether to trust remote code when loading the model. Defaults to `False`.

**Returns:**

-   `Trainer`: A Hugging Face `Trainer` object configured for training the specified model.

**Example:**

```

from predacons import predacons

trainer = predacons.trainer_legacy(
    train_file_path="data/my_training_data.txt",
    model_name="bert-base-uncased",
    output_dir="models/my_trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    trust_remote_code=False
)

trainer.train() # starts the training loop


```
