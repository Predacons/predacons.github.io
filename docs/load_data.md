# Load Data

This module provides functionalities for loading data from various sources, including directories, multiple files, and CSV files. It also includes a utility for cleaning text data. These functions are built upon the `LoadData` class within the `predacons` library.

## read_documents_from_directory
```

def read_documents_from_directory(directory, encoding="utf-8"):

```

Reads text documents from files within a specified directory.

**Parameters:**

-   `directory` (str): The path to the directory containing the text files.
-   `encoding` (str, optional): The character encoding of the text files. Defaults to "utf-8".

**Returns:**

-   `list`: A list of strings, where each string represents the content of a text file in the directory.

**Example:**

```

from predacons import predacons

documents = predacons.read_documents_from_directory("path/to/your/directory")
for doc in documents:
    print(doc)

```

**Usage Notes:**

-   The function reads all files directly under the specified directory.  It does not recursively traverse subdirectories.
-   Ensure that the files in the directory are text-based and use the specified encoding.  Incorrect encoding may lead to errors or garbled text.
- The method internally utilizes the `LoadData.read_documents_from_directory` method

## read_multiple_files
```

def read_multiple_files(file_paths):

```

Reads data from multiple files specified in a list of file paths.

**Parameters:**

-   `file_paths` (list): A list of strings, where each string is the path to a file to be read.

**Returns:**

-   `object`:  The specific return type depends on the content of the files and how `LoadData.read_multiple_files` processes them. It might return a list of file contents, a concatenated string, or a more structured data object. Refer to the `LoadData.read_multiple_files` implementation for detailed return type information.

**Example:**

```

from predacons import predacons

file_paths = ["path/to/file1.txt", "path/to/file2.txt", "path/to/file3.txt"]
data = predacons.read_multiple_files(file_paths)
print(data)

```

**Usage Notes:**

- The method internally utilizes the `LoadData.read_multiple_files` method.  The exact behavior of this function depends on the implementation of `LoadData.read_multiple_files`.
-   Ensure the file paths are correct and the files exist.
-   The function's ability to handle different file types (e.g., text, CSV, JSON) depends on the implementation within `LoadData.read_multiple_files`.

## clean_text
```

def clean_text(text):

```

Cleans the input text by removing unwanted characters or formatting.

**Parameters:**

-   `text` (str): The text string to be cleaned.

**Returns:**

-   `str`: The cleaned text string.

**Example:**

```

from predacons import predacons

dirty_text = "This is some dirty text with\n\n  extra spaces and  \t\ttabs!"
cleaned_text = predacons.clean_text(dirty_text)
print(cleaned_text)

```

**Usage Notes:**

- The method internally utilizes the `LoadData.clean_text` method. The exact cleaning steps performed (e.g., removing extra spaces, tabs, newlines, special characters) depend on the implementation within `LoadData.clean_text`.
-   This function is useful for pre-processing text data before training a model or performing text generation.

## read_csv
```

def read_csv(file_path, encoding="utf-8"):

```

Reads data from a CSV file and returns it as a pandas DataFrame.

**Parameters:**

-   `file_path` (str): The path to the CSV file.
-   `encoding` (str, optional): The character encoding of the CSV file. Defaults to "utf-8".

**Returns:**

-   `pandas.DataFrame`: A pandas DataFrame containing the data from the CSV file.

**Example:**

```

from predacons import predacons
import pandas as pd  # Import pandas explicitly

csv_file = "path/to/your/data.csv"
data = predacons.read_csv(csv_file)

if isinstance(data, pd.DataFrame):
    print(data.head())  # Print the first few rows of the DataFrame
else:
    print("Error: Could not read CSV file into a DataFrame.")

```

**Usage Notes:**

- The method internally utilizes the `LoadData.read_csv` method.
-   Requires the `pandas` library to be installed.
-   The function assumes the CSV file is properly formatted.  Errors may occur if the file is malformed.
-   The `encoding` parameter is important for handling CSV files with non-ASCII characters.  Specify the correct encoding for your file.
-   The returned object is a standard pandas DataFrame, allowing you to use pandas' powerful data manipulation and analysis capabilities.