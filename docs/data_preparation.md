---
sidebar_position: 3
---
# Data Preparation

This module provides functionalities for preparing data for training language models. It includes methods for generating text data sources using both OpenAI's GPT models and local/Hugging Face language models.

## generate_text_data_source_openai
```python

def generate_text_data_source_openai(client, gpt_model, prompt, number_of_examples, temperature=0.5):

```
Generates a text data source using OpenAI's GPT model.

**Parameters:**

-   `client`: The OpenAI client object.
-   `gpt_model` (str): The GPT model to use for generating text (e.g., "gpt-3.5-turbo").
-   `prompt` (str): The prompt to start the text generation. This serves as the seed for the model to generate further text.
-   `number_of_examples` (int): The number of text examples to generate. Each example will be a continuation of the provided prompt.
-   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated text.  A higher temperature (e.g., 1.0) will result in more random and creative text, while a lower temperature (e.g., 0.2) will make the output more focused and deterministic.  Defaults to 0.5.

**Returns:**

-   `str`: The generated text data source. The output is typically a string containing multiple examples, potentially separated by newlines or other delimiters. The exact format depends on how the `LoadData` class formats the output.

**Example:**

```python

import openai
from predacons import predacons

# Initialize OpenAI client (replace with your API key)
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Define the prompt and other parameters
prompt = "Write a short story about a robot who learns to love."
number_of_examples = 3

# Generate the text data source
text_data = predacons.generate_text_data_source_openai(client, "gpt-3.5-turbo", prompt, number_of_examples)

# Print the generated data
print(text_data)

```

## generate_text_data_source_llm
```python

def generate_text_data_source_llm(model_path, sequence, max_length, number_of_examples, trust_remote_code=False):

```
Generate a text data source for language model training using a local or Hugging Face language model.

**Parameters:**

-   `model_path` (str): The path to the language model or the name of a Hugging Face model (e.g., "bert-base-uncased", or "/path/to/my/model").
-   `sequence` (str): The input sequence (prompt) to generate data from. This is the starting point for the text generation.
-   `max_length` (int): The maximum length of the generated text, including the initial sequence. This prevents the model from generating excessively long outputs.
-   `number_of_examples` (int): The number of examples to generate.
-   `trust_remote_code` (bool, optional): Whether to trust remote code when loading the model from Hugging Face. Set to `True` if the model requires executing custom code. Defaults to `False`.

**Returns:**

-   `str`: The generated text data source. The output is a string containing multiple generated text examples. The format of the string (e.g., separation between examples) is determined by the `DataPreparation` class.

**Example:**

```python

from predacons import predacons

# Define the model path, sequence, and other parameters
model_path = "gpt2"  # Or a path to a local model
sequence = "The quick brown fox"
max_length = 100
number_of_examples = 2

# Generate the text data source
text_data = predacons.generate_text_data_source_llm(model_path, sequence, max_length, number_of_examples)

# Print the generated data
print(text_data)

```