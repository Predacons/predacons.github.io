---
sidebar_position: 6
---
# Speculative Fast Generation

This module provides functionalities for accelerating text generation using speculative decoding. It leverages a draft model to predict potential tokens, which are then verified by the main model, significantly reducing the computational cost of generation.

## GPTFast Class

The `GPTFast` class encapsulates methods for fast text generation.

### generate_text_fast
```python

def generate_text_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = False,gguf_file=None):


```
Generates text quickly using speculative decoding. This method loads both a main model and a draft model. The draft model proposes tokens, and the main model verifies them.

**Parameters:**

-   `model_path` (str): The path to the main model.
-   `draft_model_name` (str): The path to the draft model.  Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.
-   `sequence` (str): The input sequence to start generation.
-   `max_length` (int): The maximum length of the generated text.
-   `trust_remote_code` (bool, optional): Whether to trust remote code when loading the models. Defaults to `False`.
-	`gguf_file` (str, optional): The path to the GGUF file. Defaults to `None`.

**Returns:**

-   `str`: The generated text.

**Example:**

```python

from predacons import predacons

generated_text = predacons.generate_text(
    model_path="path/to/main/model",
    draft_model_name="path/to/draft/model",
    sequence="The quick brown fox",
    max_length=100,
    use_fast_generation=True
)
print(generated_text)


```

### generate_output_fast
```python

def generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = False,gguf_file=None):


```
Generates output sequences quickly using speculative decoding.  This method loads both a main model and a draft model.  The draft model proposes tokens, and the main model verifies them.

**Parameters:**

-   `model_path` (str): The path to the main model.
-   `draft_model_name` (str): The path to the draft model.  Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.
-   `sequence` (str): The input sequence to start generation.
-   `max_length` (int): The maximum length of the generated output.
-   `trust_remote_code` (bool, optional): Whether to trust remote code when loading the models. Defaults to `False`.
-	`gguf_file` (str, optional): The path to the GGUF file. Defaults to `None`.

**Returns:**

-   `torch.Tensor`: The generated token ids.
-   `transformers.PreTrainedTokenizer`: The tokenizer used.

**Example:**

```python

from predacons import predacons

output, tokenizer = predacons.generate(
    model_path="path/to/main/model",
    draft_model_name="path/to/draft/model",
    sequence="The quick brown fox",
    max_length=100,
    use_fast_generation=True
)
print(tokenizer.decode(output[0], skip_special_tokens=True))


```

### generate_output_from_model
```python

def generate_output_from_model(model, tokenizer, sequence, max_length):


```

Generates output from a pre-loaded `torch._dynamo.eval_frame.OptimizedModule` model and tokenizer using fast generation techniques.

**Parameters:**

-   `model` (torch._dynamo.eval_frame.OptimizedModule): The pre-loaded optimized main model.
-   `tokenizer` (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
-   `sequence` (str): The input sequence to start generation.
-   `max_length` (int): The maximum length of the generated output.

**Returns:**

-   `torch.Tensor`: The generated token ids.

**Example:**

```python

from predacons import predacons
import torch

# Assuming model and tokenizer are already loaded
model = torch.load("path/to/optimized/model.pt")
tokenizer = predacons.load_tokenizer("path/to/tokenizer")

output = predacons.generate(
    model=model,
    tokenizer=tokenizer,
    sequence="The quick brown fox",
    max_length=100
)
print(tokenizer.decode(output[0], skip_special_tokens=True))


```

### load_model
```python

def load_model(model_path, draft_model_name,trust_remote_code=False,gguf_file=None):


```
Loads both the main model and the draft model for fast generation.

**Parameters:**

-   `model_path` (str): The path to the main model.
-   `draft_model_name` (str): The path to the draft model. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.
-   `trust_remote_code` (bool, optional): Whether to trust remote code when loading the models. Defaults to `False`.
-	`gguf_file` (str, optional): The path to the GGUF file. Defaults to `None`.

**Returns:**

-   `transformers.PreTrainedModel`: The loaded main model.

**Example:**

```python

from predacons import predacons

model = predacons.load_model(
    model_path="path/to/main/model",
    draft_model_name="path/to/draft/model",
    use_fast_generation = True
)


```

**Note:**

-   Speculative decoding requires both a main model and a smaller, faster "draft" model.  The draft model should ideally be significantly smaller to provide a speed advantage.
-   `auto_quantize` is currently not supported with fast generation.
-   `apply_chat_template` is not supported with fast generation yet.
