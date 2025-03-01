
# Load Data Module

The `LoadData` module provides functionalities for loading data from various file formats and cleaning text. It supports reading from PDF, DOCX, TXT, and CSV files, and offers a text cleaning method to remove unwanted characters and formatting.

## Class: `LoadData`

The `LoadData` class contains static methods for reading data from different file types and cleaning text. You don't need to instantiate this class to use its methods.

### Static Methods

#### `read_documents_from_directory(directory, encoding="utf-8")`

Reads documents from a specified directory, combining the text content from all supported files (`.pdf`, `.docx`, `.txt`).

**Parameters:**

-   `directory` (str): The path to the directory containing the documents.
-   `encoding` (str, optional): The encoding of the text files. Defaults to `"utf-8"`.

**Returns:**

-   `str`: A string containing the combined text from all documents in the directory.

**Example:**

```
---
python
from predacons import predacons

# Read documents from a directory
text_data = predacons.read_documents_from_directory('path/to/your/directory')
print(text_data)

---
```

#### `read_multiple_files(file_paths)`

Reads data from a list of specified file paths, combining the text content from all supported files (`.pdf`, `.docx`, `.txt`).

**Parameters:**

-   `file_paths` (list): A list of file paths to read data from.

**Returns:**

-   `str`: A string containing the combined text from all documents in the list.

**Example:**

```
---
python
from predacons import predacons

# Read data from multiple files
file_list = ['path/to/your/file1.txt', 'path/to/your/file2.pdf']
text_data = predacons.read_multiple_files(file_list)
print(text_data)

---
```

#### `clean_text(text)`

Cleans the given text by removing redundant newline characters and trimming whitespace.

**Parameters:**

-   `text` (str): The text to be cleaned.

**Returns:**

-   `str`: The cleaned text.

**Example:**

```
---
python
from predacons import predacons

# Clean a text string
dirty_text = "This is a dirty text.\n\n\nWith many newlines."
cleaned_text = predacons.clean_text(dirty_text)
print(cleaned_text)

---
```

#### `read_csv(file_path, encoding="utf-8")`

Reads a CSV file and returns the data as a pandas DataFrame.

**Parameters:**

-   `file_path` (str): The path to the CSV file.
-   `encoding` (str, optional): The encoding of the CSV file. Defaults to `"utf-8"`.

**Returns:**

-   `pandas.DataFrame`: The data from the CSV file.

**Example:**

```
---
python
from predacons import predacons

# Read a CSV file
csv_data = predacons.read_csv('path/to/your/file.csv')
print(csv_data.head())  # Print the first few rows of the DataFrame

---
```