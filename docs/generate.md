---
sidebar_position: 5
---
# Generate

This module provides functionalities for generating text, chat outputs, and handling streaming using trained models. It leverages the `Generate` class for default generation methods and `GPTFast` for fast generation using speculative decoding.

## generate_text

```python

def generate_text(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None,gguf_file=None):


```

Generates text using the specified model. This method is being deprecated, and `text_generate` should be used instead.

**Args:**

*   `model_path` (str): The path to the model.
*   `sequence` (str): The input sequence to generate text from.
*   `max_length` (int): The maximum length of the generated text.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.

**Returns:**

*   `str`: The generated text.

**Example:**

```python

generated_text = predacons.generate_text(model_path="path/to/your/model",
                                         sequence="Seed text for generation",
                                         max_length=50)


```

**Note:** This method suggests loading the model first for repetitive generation using `text_generate` for better performance. If `use_fast_generation` is set to `True` and `draft_model_name` is not provided, the base model is used as the draft model, which may increase memory utilization.

## generate_output

```python

def generate_output(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None,temperature=0.1,apply_chat_template = False,gguf_file=None,auto_quantize=None):


```

Generates output using the specified model. This method is being deprecated, and `generate` should be used instead.

**Args:**

*   `model_path` (str): The path to the model.
*   `sequence` (str): The input sequence for generating the output.
*   `max_length` (int): The maximum length of the generated output.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `str`: The generated output.

**Example:**

```python

generated_output = predacons.generate_output(model_path="path/to/your/model",
                                         sequence="Seed text for generation",
                                         max_length=50)


```

**Note:** Similar to `generate_text`, this method suggests loading the model first for repetitive generation using `generate` for better performance. `apply_chat_template` is not supported with fast generation yet. If `use_fast_generation` is set to `True` and `draft_model_name` is not provided, the base model is used as the draft model, which may increase memory utilization.

## generate

```python

def generate(*args, **kwargs):


```

Generates output based on the provided arguments, offering flexibility for different generation scenarios, including chat templates and fast generation.

**Args:**

*   `*args`: Variable length arguments.
*   `**kwargs`: Keyword arguments.

**Keyword Args:**

*   `model_path` (str): The path to the model file.
*   `sequence` (str): The input sequence to generate output from.
*   `max_length` (int, optional): The maximum length of the generated output. Defaults to `50`.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `model` (object): The model object.
*   `tokenizer` (object): The tokenizer object.
 *   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.
    *   `stream` (bool, optional): Whether to stream the output. Defaults to `False`. if True, thread and streamer will be returned.

**Returns:**

*   `str`: The generated output.
    or
*   `tuple`: thread, streamer.
**Raises:**

*   `ValueError`: If the arguments are invalid.

**Example (using model path):**

```python

generated_text = predacons.generate(model_path="path/to/your/model",
                                         sequence="Seed text for generation",
                                         max_length=50)


```

**Example (using preloaded model and tokenizer):**

```python

model = AutoModelForCausalLM.from_pretrained("path/to/your/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
generated_text = predacons.generate(model=model,
                                         tokenizer=tokenizer,
                                         sequence="Seed text for generation",
                                         max_length=50)


```

**Note:** This function supports both passing the model path or pre-loading the model and tokenizer. `apply_chat_template` is not supported with fast generation yet. If `use_fast_generation` is set to `True` and `draft_model_name` is not provided, the base model is used as the draft model, which may increase memory utilization.

## text_generate

```python

def text_generate(*args, **kwargs):


```

Generates text and prints the output to the console.

**Args:**

*   `*args`: Variable length argument list.
*   `**kwargs`: Arbitrary keyword arguments. (Same as `generate`)

**Keyword Args:**

*   `model_path` (str): The path to the model file.
*   `sequence` (str): The input sequence to generate output from.
*   `max_length` (int, optional): The maximum length of the generated output. Defaults to `50`.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `model` (object): The model object.
*   `tokenizer` (object): The tokenizer object.
*   `stream` (bool, optional): Whether to stream the output. Defaults to `False`. if True, thread and streamer will be returned.
    *   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `str`: The generated text.
    or
*   `tuple`: thread, streamer.

**Example:**

```python

generated_text = predacons.text_generate(model_path="path/to/your/model",
                                         sequence="Seed text for generation",
                                         max_length=50)
print(generated_text)


```

**Note:** It calls the `generate` function internally and prints the decoded text to the console.

## text_stream

```python

def text_stream(*args, **kwargs):


```

Streams text and prints the output to the console.

**Args:**

*   `*args`: Variable length argument list.
*   `**kwargs`: Arbitrary keyword arguments. (Same as `generate`)

**Keyword Args:**

*   `model_path` (str): The path to the model file.
*   `sequence` (str): The input sequence to generate output from.
*   `max_length` (int, optional): The maximum length of the generated output. Defaults to `50`.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `model` (object): The model object.
*   `tokenizer` (object): The tokenizer object.
*   `stream` (bool, optional): Whether to stream the output. Defaults to `False`. if True, thread and streamer will be returned.
    *   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `str`: The generated text.

**Example:**

```python

for text in predacons.text_stream(model_path="path/to/your/model",
                                  sequence="Seed text for generation",
                                  max_length=50):
    print(text)


```

**Note:** It calls the `generate` function internally with streaming enabled and prints the streamed text to the console.

## chat_generate

```python

def chat_generate(*args, **kwargs):


```

Generates chat output using the specified model and prints the output to the console.

**Args:**

*   `*args`: Variable length argument list.
*   `**kwargs`: Arbitrary keyword arguments. (Same as `generate`)

**Keyword Args:**

*   `model_path` (str): The path to the model file.
*   `sequence` (str): The input sequence to generate output from.
*   `max_length` (int, optional): The maximum length of the generated output. Defaults to `50`.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `model` (object): The model object.
*   `tokenizer` (object): The tokenizer object.
*    `dont_print_output` (bool, optional): Whether to print the output. Defaults to `False`.
*   `stream` (bool, optional): Whether to stream the output. Defaults to `False`. if True, thread and streamer will be returned.
    *   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `str`: The generated chat output.

**Example:**

```python

chat = [
    {"role": "user", "content": "Hey, what is a car?"}
]
chat_output = predacons.chat_generate(model = model,
        sequence = chat,
        max_length = 50,
        tokenizer = tokenizers,
        trust_remote_code = True)


```

**Note:** It calls the `generate` function internally with `apply_chat_template` enabled and prints the decoded text to the console.

## chat_stream

```python

def chat_stream(*args, **kwargs):


```

Streams chat output using the specified model and prints the output to the console.

**Args:**

*   `*args`: Variable length argument list.
*   `**kwargs`: Arbitrary keyword arguments. (Same as `generate`)

**Keyword Args:**

*   `model_path` (str): The path to the model file.
*   `sequence` (str): The input sequence to generate output from.
*   `max_length` (int, optional): The maximum length of the generated output. Defaults to `50`.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `model` (object): The model object.
*   `tokenizer` (object): The tokenizer object.
*   `stream` (bool, optional): Whether to stream the output. Defaults to `False`. if True, thread and streamer will be returned.
    *   `apply_chat_template` (bool, optional): Whether to apply the chat template. Defaults to `False`.
*   `temperature` (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to `0.1`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `str`: The generated chat output.

**Example:**

```python

chat = [
    {"role": "user", "content": "Hey, what is a car?"}
]
for chat in predacons.chat_stream(model = model,
                                  sequence = chat,
                                  max_length = 50,
                                  tokenizer = tokenizers,
                                  trust_remote_code = True):
    print(chat)


```

**Note:** It calls the `generate` function internally with `apply_chat_template` and streaming enabled, and prints the streamed chat output to the console.

## load_model

```python

def load_model(model_path,trust_remote_code=False,use_fast_generation=False, draft_model_name=None,gguf_file=None,auto_quantize=None):


```

Loads a model from the specified `model_path`.

**Args:**

*   `model_path` (str): The path to the model.
*   `trust_remote_code` (bool, optional): Whether to trust remote code. Defaults to `False`.
*   `use_fast_generation` (bool, optional): Whether to use fast generation. Defaults to `False`.
*   `draft_model_name` (str, optional): The name of the draft model. Defaults to `None`.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.
*   `auto_quantize` (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

**Returns:**

*   `Model`: The loaded model.

**Raises:**

*   `FileNotFoundError`: If the `model_path` does not exist.

## load_tokenizer

```python

def load_tokenizer(tokenizer_path,gguf_file=None):


```

Loads a tokenizer from the specified path.

**Args:**

*   `tokenizer_path` (str): The path to the tokenizer file.
*   `gguf_file` (str, optional): The path to the GGUF file. Defaults to None.

**Returns:**

*   `Tokenizer`: The loaded tokenizer object.