# Predacons Module Documentation

The `predacons` module provides a high-level interface for interacting with various functionalities of the Predacons library. It includes functions for data loading, model training, and text generation.

## Functions

### rollout()

```
---
python
predacons.rollout()

---
```

Prints the Predacons rollout message, version information, and a summary of available functions. This function provides a quick overview of the library's capabilities.

**Example:**

```
---
python
from predacons import predacons
predacons.rollout()

---
```

**Output:**

```
---

Predacons rollout !!!
Predacons Version: v0.0.129

read_documents_from_directory -- Load data from directory
    directory -- Directory path

read_multiple_files -- Load data from multiple files
    file_paths -- list of File paths

clean_text -- Clean text
    text -- Text

read_csv -- Read csv file
    file_path -- File path

train_legacy -- Train Predacons
    train_file_path -- Train file path
    model_name -- Model name
    output_dir -- Output directory
    overwrite_output_dir -- Overwrite output directory
    per_device_train_batch_size -- Per device train batch size
    num_train_epochs -- Number of train epochs
    save_steps -- Save steps
    trust_remote_code -- Trust remote code

trainer_legacy -- returns trainer
    train_file_path -- Train file path
    model_name -- Model name
    output_dir -- Output directory
    overwrite_output_dir -- Overwrite output directory
    per_device_train_batch_size -- Per device train batch size
    num_train_epochs -- Number of train epochs
    save_steps -- Save steps
    trust_remote_code -- Trust remote code

train -- Train Predacons
    train_file_path -- Train file path
    model_name -- Model name
    output_dir -- Output directory
    overwrite_output_dir -- Overwrite output directory
    per_device_train_batch_size -- Per device train batch size
    num_train_epochs -- Number of train epochs
    save_steps -- Save steps
    trust_remote_code -- Trust remote code

trainer -- returns trainer
    train_file_path -- Train file path
    model_name -- Model name
    output_dir -- Output directory
    overwrite_output_dir -- Overwrite output directory
    per_device_train_batch_size -- Per device train batch size
    num_train_epochs -- Number of train epochs
    save_steps -- Save steps
    trust_remote_code -- Trust remote code

generate_text -- Generate text (Deprecating soon, use text_generate instead)
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)

generate_output -- returns output and tokenizer (Deprecating soon, use generate instead)
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)
    apply_chat_template -- use chat template (defauly False)

generate -- Generate text or output
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)
    model -- give a preloaded Model (default None)
    tokenizer -- give a preloaded Tokenizer (default None)

text_generate -- Generate text and print
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)
    model -- give a preloaded Model (default None)
    tokenizer -- give a preloaded Tokenizer (default None)

text_stream -- stream text and print
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)

chat_generate -- Generate chat and print
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)
    model -- give a preloaded Model (default None)
    tokenizer -- give a preloaded Tokenizer (default None)
    apply_chat_template -- use chat template (defauly False)
    dont_print_output -- Dont print output (default False)
    gguf_file -- GGUF file path (default None)
    auto_quantize -- Automatically apply quantization (default None)

chat_stream -- Stream chat and print
    model_path -- Model path
    sequence -- Sequence
    max_length -- Max length
    trust_remote_code -- Trust remote code (default False)

load_model -- Load model
    model_path -- Model path
    trust_remote_code -- Trust remote code (default False)
    use_fast_generation -- Use fast generation using speculative decoding (default False)
    draft_model_name -- Draft model name / path (default None)

load_tokenizer -- Load tokenizer
    tokenizer_path -- Tokenizer path

generate_text_data_source_openai -- Generate text data source using openai
    client -- openai client
    gpt_model -- GPT model used for generation
    prompt -- Prompt to generate data source
    number_of_examples -- Number of examples
    temperature -- Temperature (default 0.5)

generate_text_data_source_ll -- Generate text data source using local or hugging face llm
    model_path -- Model path or hugging face model name
    sequence -- Sequence (prompt)
    max_length -- Max length of the generated text
    number_of_examples -- Number of examples
    trust_remote_code -- Trust remote code

Predacons rollout !!!

---
```

### read_documents_from_directory(directory, encoding="utf-8")

```
---
python
predacons.read_documents_from_directory(directory, encoding="utf-8")

---
```

Reads all text files from a specified directory.

**Parameters:**

- `directory` (str): The path to the directory containing the documents.
- `encoding` (str, optional): The encoding of the text files. Defaults to `"utf-8"`.

**Returns:**

- `list`: A list of strings, where each string is the content of a file in the directory.

**Example:**

```
---
python
from predacons import predacons
documents = predacons.read_documents_from_directory("data/")
print(documents)

---
```

### read_multiple_files(file_paths)

```
---
python
predacons.read_multiple_files(file_paths)

---
```

Reads data from a list of files.

**Parameters:**

- `file_paths` (list): A list of file paths to read.

**Returns:**

- `object`: The loaded data. The type of object returned depends on the contents of the file being read.

**Example:**

```
---
python
from predacons import predacons
file_paths = ["data/file1.txt", "data/file2.txt"]
data = predacons.read_multiple_files(file_paths)
print(data)

---
```

### clean_text(text)

```
---
python
predacons.clean_text(text)

---
```

Cleans the input text by removing unwanted characters or formatting.

**Parameters:**

- `text` (str): The text to be cleaned.

**Returns:**

- `str`: The cleaned text.

**Example:**

```
---
python
from predacons import predacons
dirty_text = "This is a dirty text with \n some unwanted characters! "
cleaned_text = predacons.clean_text(dirty_text)
print(cleaned_text)

---
```

### read_csv(file_path, encoding="utf-8")

```
---
python
predacons.read_csv(file_path, encoding="utf-8")

---
```

Reads a CSV file into a pandas DataFrame.

**Parameters:**

- `file_path` (str): The path to the CSV file.
- `encoding` (str, optional): The encoding of the CSV file. Defaults to `"utf-8"`.

**Returns:**

- `pandas.DataFrame`: A pandas DataFrame containing the data from the CSV file.

**Example:**

```
---
python
from predacons import predacons
import pandas as pd
csv_data = predacons.read_csv("data.csv")
print(type(csv_data))

---
```

### train_legacy(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, trust_remote_code=False, resume_from_checkpoint=True)

```
---
python
predacons.train_legacy(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, trust_remote_code=False, resume_from_checkpoint=True)

---
```

Trains a Predacons model using the legacy training method.

**Parameters:**

- `train_file_path` (str): The path to the training file.
- `model_name` (str): The name of the model to be trained.
- `output_dir` (str): The directory to save the trained model.
- `overwrite_output_dir` (bool): Whether to overwrite the output directory if it exists.
- `per_device_train_batch_size` (int): The batch size per device during training.
- `num_train_epochs` (int): The number of training epochs.
- `save_steps` (int): The number of steps after which to save the model.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
- `resume_from_checkpoint` (bool, optional): Whether to resume training from the latest checkpoint in `output_dir`. Defaults to `True`.

**Example:**

```
---
python
from predacons import predacons
predacons.train_legacy(
    train_file_path="train.txt",
    model_name="bert-base-uncased",
    output_dir="output",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    trust_remote_code=True
)

---
```

### trainer_legacy(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, trust_remote_code=False)

```
---
python
predacons.trainer_legacy(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, trust_remote_code=False)

---
```

Returns a trainer object for the legacy training method. Does not start training.

**Parameters:**

- `train_file_path` (str): The path to the training file.
- `model_name` (str): The name of the model to be trained.
- `output_dir` (str): The directory to save the trained model.
- `overwrite_output_dir` (bool): Whether to overwrite the output directory if it exists.
- `per_device_train_batch_size` (int): The batch size per device during training.
- `num_train_epochs` (int): The number of training epochs.
- `save_steps` (int): The number of steps after which to save the model.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.

**Returns:**

- `Trainer`: A Hugging Face Trainer object.

**Example:**

```
---
python
from predacons import predacons
trainer = predacons.trainer_legacy(
    train_file_path="train.txt",
    model_name="bert-base-uncased",
    output_dir="output",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    trust_remote_code=True
)

---
```

### train(\*args, \*\*kwargs)

```
---
python
predacons.train(*args, **kwargs)

---
```

Trains a Predacons model using the specified arguments and keyword arguments.

This is a wrapper around the `TrainPredacons.train` method. See the documentation for `TrainPredacons.train` for more details.

### trainer(\*args, \*\*kwargs)

```
---
python
predacons.trainer(*args, **kwargs)

---
```

Returns a trainer object for the training method. Does not start training.

This is a wrapper around the `TrainPredacons.trainer` method. See the documentation for `TrainPredacons.trainer` for more details.

### generate_text(model_path, sequence, max_length, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, gguf_file=None)

```
---
python
predacons.generate_text(model_path, sequence, max_length, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, gguf_file=None)

---
```

Generates text using a specified model. Deprecated, use `text_generate` instead.

**Parameters:**

- `model_path` (str): The path to the model.
- `sequence` (str): The input sequence to generate text from.
- `max_length` (int): The maximum length of the generated text.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
- `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
- `draft_model_name` (str, optional): The name of the draft model for fast generation. Defaults to `None`.
- `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.

**Returns:**

- `str`: The generated text.

**Example:**

```
---
python
from predacons import predacons
generated_text = predacons.generate_text(
    model_path="path/to/model",
    sequence="The quick brown fox",
    max_length=100,
    trust_remote_code=True
)
print(generated_text)

---
```

### generate_output(model_path, sequence, max_length, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, temperature=0.1, apply_chat_template=False, gguf_file=None, auto_quantize=None)

```
---
python
predacons.generate_output(model_path, sequence, max_length, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, temperature=0.1, apply_chat_template=False, gguf_file=None, auto_quantize=None)

---
```

Generates output using the specified model. Deprecated, use `generate` instead.

**Parameters:**

- `model_path` (str): The path to the model.
- `sequence` (str): The input sequence.
- `max_length` (int): The maximum length of the output.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
- `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
- `draft_model_name` (str, optional): The name of the draft model for fast generation. Defaults to `None`.
- `temperature` (float, optional): The temperature for generating text. Defaults to 0.1.
- `apply_chat_template` (bool, optional): Whether to apply a chat template. Defaults to `False`.
- `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
- `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

- `str`: The generated output.

**Example:**

```
---
python
from predacons import predacons
output = predacons.generate_output(
    model_path="path/to/model",
    sequence="Translate to French: Hello",
    max_length=100,
    trust_remote_code=True,
    temperature=0.5
)
print(output)

---
```

### generate(\*args, \*\*kwargs)

```
---
python
predacons.generate(*args, **kwargs)

---
```

Generates output based on the provided arguments.

**Parameters:**

- `*args`: Variable length arguments.
- `**kwargs`: Keyword arguments.

**Keyword Args:**
    - `model_path` (str): The path to the model file.
    - `sequence` (str): The input sequence to generate output from.
    - `max_length` (int, optional): The maximum length of the generated output. Defaults to 50.
    - `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to False.
    - `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to False.
    - `draft_model_name` (str, optional): The name of the draft model. Defaults to None.
    - `model` (object): The model object.
    - `tokenizer` (object): The tokenizer object.
    - `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to False.
    - `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to 0.1.
    - `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
    - `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.
    - `stream` (bool, optional): Whether to stream the output. Defaults to False. if True, thread and streamer will be returned.

**Returns:**

- `str`: The generated output.
    or
- `thread, streamer`: The thread and streamer object.

**Example:**

```
---
python
from predacons import predacons
output = predacons.generate(
    model_path="path/to/model",
    sequence="The quick brown fox",
    max_length=100,
    trust_remote_code=True
)
print(output)

---
```

### text_generate(\*args, \*\*kwargs)

```
---
python
predacons.text_generate(*args, **kwargs)

---
```

Generates text and prints it to the console.

**Parameters:**

- `*args`: Variable length arguments.
- `**kwargs`: Keyword arguments.  See `generate` function for more details.

**Returns:**

- `str`: The generated text.

**Example:**

```
---
python
from predacons import predacons
text = predacons.text_generate(
    model_path="path/to/model",
    sequence="The quick brown fox",
    max_length=100,
    trust_remote_code=True
)
print(text) # can be ommited, text also prints

---
```

### text_stream(\*args, \*\*kwargs)

```
---
python
predacons.text_stream(*args, **kwargs)

---
```

Streams text to the console.

**Parameters:**

- `*args`: Variable length arguments.
- `**kwargs`: Keyword arguments.   See `generate` function for more details.

**Returns:**

- `str`: The generated text.

**Example:**

```
---
python
from predacons import predacons
text = predacons.text_stream(
    model_path="path/to/model",
    sequence="The quick brown fox",
    max_length=100,
    trust_remote_code=True
)
print(text) # prints to console as well

---
```

### chat_generate(\*args, \*\*kwargs)

```
---
python
predacons.chat_generate(*args, **kwargs)

---
```

Generates a chat response and prints it to the console.

**Parameters:**

- `*args`: Variable length arguments.
- `**kwargs`: Keyword arguments.

**Keyword Args:**
    - `model_path` (str): The path to the model file.
    - `sequence` (str): The input sequence to generate output from.
    - `dont_print_output` (bool, optional): Whether to print the output. Defaults to False.
    - `max_length` (int, optional): The maximum length of the generated output. Defaults to 50.
    - `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to False.
    - `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to False.
    - `draft_model_name` (str, optional): The name of the draft model. Defaults to None.
    - `model` (object): The model object.
    - `tokenizer` (object): The tokenizer object.
    - `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to False.
    - `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to 0.1.
    - `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
    - `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

- `str`: The generated chat response.

**Example:**

```
---
python
from predacons import predacons
chat = [
    {"role": "user", "content": "Hey, what is a car?"}
]
chat_output = predacons.chat_generate(
    model_path="path/to/model",
    sequence=chat,
    max_length=50,
    trust_remote_code=True
)
print(chat_output) # Also prints by default.

---
```

### chat_stream(\*args, \*\*kwargs)

```
---
python
predacons.chat_stream(*args, **kwargs)

---
```

Streams a chat to the console.

**Parameters:**

- `*args`: Variable length arguments.
- `**kwargs`: Keyword arguments. See `chat_generate` function for more details.

**Returns:**

- `str`: The generated chat response.

**Example:**

```
---
python
from predacons import predacons
chat = [
    {"role": "user", "content": "Hey, what is a car?"}
]
chat_output = predacons.chat_stream(
    model_path="path/to/model",
    sequence=chat,
    max_length=50,
    trust_remote_code=True
)
print(chat_output) #prints to console

---
```

### load_model(model_path, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, gguf_file=None, auto_quantize=None)

```
---
python
predacons.load_model(model_path, trust_remote_code=False, use_fast_generation=False, draft_model_name=None, gguf_file=None, auto_quantize=None)

---
```

Loads a model from the specified path.

**Parameters:**

- `model_path` (str): The path to the model.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
- `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
- `draft_model_name` (str, optional): The name of the draft model for fast generation. Defaults to `None`.
- `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
- `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

- `Model`: The loaded model.

**Example:**

```
---
python
from predacons import predacons
model = predacons.load_model(
    model_path="path/to/model",
    trust_remote_code=True
)

---
```

### load_tokenizer(tokenizer_path, gguf_file=None)

```
---
python
predacons.load_tokenizer(tokenizer_path, gguf_file=None)

---
```

Loads a tokenizer from the specified path.

**Parameters:**

- `tokenizer_path` (str): The path to the tokenizer.
- `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.

**Returns:**

- `Tokenizer`: The loaded tokenizer.

**Example:**

```
---
python
from predacons import predacons
tokenizer = predacons.load_tokenizer(tokenizer_path="path/to/tokenizer")

---
```

### generate_text_data_source_openai(client, gpt_model, prompt, number_of_examples, temperature=0.5)

```
---
python
predacons.generate_text_data_source_openai(client, gpt_model, prompt, number_of_examples, temperature=0.5)

---
```

Generates text data using OpenAI's GPT models.

**Parameters:**

- `client`: The OpenAI client object.
- `gpt_model` (str): The name of the GPT model to use.
- `prompt` (str): The prompt to start the text generation.
- `number_of_examples` (int): The number of text examples to generate.
- `temperature` (float, optional): The temperature for generating text. Defaults to `0.5`.

**Returns:**

- The generated text data source.

**Example:**

```
---
python
from predacons import predacons
import openai
openai.api_key = "YOUR_API_KEY"
client = openai.OpenAI()
data = predacons.generate_text_data_source_openai(
    client=client,
    gpt_model="gpt-3.5-turbo",
    prompt="Write a short story about a dog",
    number_of_examples=5
)
print(data)

---
```

### generate_text_data_source_llm(model_path, sequence, max_length, number_of_examples, trust_remote_code=False)

```
---
python
predacons.generate_text_data_source_llm(model_path, sequence, max_length, number_of_examples, trust_remote_code=False)

---
```

Generates text data using a local or Hugging Face language model.

**Parameters:**

- `model_path` (str): The path to the language model.
- `sequence` (str): The input sequence to generate data from.
- `max_length` (int): The maximum length of the generated data.
- `number_of_examples` (int): The number of examples to generate.
- `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.

**Returns:**

- `str`: The generated text data source.

**Example:**

```
---
python
from predacons import predacons
data = predacons.generate_text_data_source_llm(
    model_path="gpt2",
    sequence="The quick brown fox",
    max_length=100,
    number_of_examples=5,
    trust_remote_code=True
)
print(data)

---
```