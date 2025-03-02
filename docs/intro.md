---
sidebar_position: 1
---
# Predacons

Predacons is a Python library built on top of the Hugging Face Transformers library, designed to simplify and enhance various Natural Language Processing (NLP) tasks. It provides a user-friendly interface for data loading, model training, and text generation, enabling you to leverage the power of transfer learning with ease.

![PyPI](https://img.shields.io/pypi/v/predacons)   ![Downloads](https://img.shields.io/pypi/dm/predacons)   ![License](https://img.shields.io/pypi/l/predacons)   ![Python Version](https://img.shields.io/pypi/pyversions/predacons)

## Purpose

The primary goal of Predacons is to abstract away the complexities involved in working with transformer models, making them more accessible to a wider audience. Whether you're a seasoned machine learning practitioner or just getting started with NLP, Predacons offers a set of tools to streamline your workflow and accelerate your projects.

## Key Features

Predacons offers a comprehensive suite of functionalities, including:

*   **Easy Data Loading:** Load text data from various sources, including directories, individual files, and CSV files, with automatic handling of encoding.
*   **Text Cleaning Utilities:** Preprocess your text data with built-in cleaning functions to remove noise and improve model performance.
*   **Simplified Model Training:** Train transformer models on your custom datasets with minimal code, leveraging the power of the Hugging Face Trainer. Supports both standard and legacy training methods.
*   **Versatile Text Generation:** Generate text using trained models with various options, including fast generation using speculative decoding, temperature control, and chat templates.
*   **Text Streaming:** Stream text generation output for real-time applications.
*   **Chat Generation and Streaming:** Generate and stream chat responses, suitable for building conversational AI applications.
*   **Data Preparation Tools:** Generate synthetic datasets using OpenAI or local language models to augment your training data.
*   **Model and Tokenizer Loading:** Easily load pre-trained models and tokenizers from local paths or the Hugging Face Model Hub.
*   **Embeddings Generation:** Generate embeddings for sentences using pre-trained transformer models, with full compatibility with Langchain methods.

## Installation

To get started with Predacons, simply install it using pip:

```

pip install predacons


```

## Quick Start

Here's a basic example to demonstrate how to use Predacons for text generation:

```

from predacons import predacons

# Initialize Predacons (prints available functions)
predacons.rollout()

# Load a pre-trained model (replace with your model path)
model_path = "path/to/your/model"  # e.g., "bert-base-uncased"
sequence = "The quick brown fox"

# Generate text
generated_text = predacons.generate_text(model_path=model_path, sequence=sequence, max_length=50)
print(f"Generated text: {generated_text}")

# Example usage for chat generation
chat_sequence = [{"role": "user", "content": "Tell me a joke."}]
generated_chat = predacons.chat_generate(model_path=model_path, sequence=chat_sequence, max_length=100)
print(f"Generated chat: {generated_chat}")


```

## Documentation

For detailed information on specific functions and modules, please refer to the following documentation pages:

*   [Data Loading (`load_data.py`)](./load_data.md): Functions for reading and cleaning text data from various sources.
*   [Model Training (`train_predacons.py`)](./train.md): Tools for training Predacons models, including trainer configuration.
*   [Text Generation (`generate.py`)](./generate.md): Functions for generating text using trained models.
*   [Data Preparation (`data_preparation.py`)](./data_preparation.md): Utilities for creating and augmenting datasets.
*   [Speculative Fast Generation (`speculative_fast_generation.py`)](./speculative_fast_generation.md): Implementation of fast text generation using speculative decoding.

## Contributing

We welcome contributions to Predacons! If you have any ideas, suggestions, or bug reports, please open an issue on the [GitHub repository](https://github.com/Predacons/predacons). If you'd like to contribute code, please submit a pull request.

## License

This project is licensed under multiple licenses:

- For **free users**, the project is licensed under the terms of the GNU Affero General Public License (AGPL). See  [`LICENSE-AGPL`](https://github.com/Predacons/predacons/blob/main/LICENSE-AGPL) for more details.

- For **paid users**, there are two options:
    - A perpetual commercial license. See [`LICENSE-COMMERCIAL-PERPETUAL`](https://github.com/Predacons/predacons/blob/main/LICENSE-COMMERCIAL-PERPETUAL) for more details.
    - A yearly commercial license. See [`LICENSE-COMMERCIAL-YEARLY`](https://github.com/Predacons/predacons/blob/main/LICENSE-COMMERCIAL-YEARLY) for more details.

Please ensure you understand and comply with the license that applies to you.