import invisensing.File as f

file = f.File('/path/to/file')      # Open a file
while (file.get_lines_left() > 0):  # Loop to read the file
    data = file.get_lines(5)        # Read 5 lines
    # Process data...