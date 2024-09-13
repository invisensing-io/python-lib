# Invisensing python library

This library is a SDK to develop software for the invisensing infrastructure.

## Installation

Go to the release page to get the latest release's URL

```Shell
pip install <release url>
```
Basic usage

```Python
import invisensing.File as f

file = f.File('/path/to/file')      # Open a file
while (file.get_lines_left() > 0):  # Loop to read the file
    data = file.get_lines(5)        # Read 5 lines
    # Process data...
```
