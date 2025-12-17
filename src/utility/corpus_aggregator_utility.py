#
# Code to aggregate all sanskrit corpus data into a single file
#

import glob
import os

def aggregate_txt_files(directory_path, output_filename="/Users/prathara/Code/SSLM/SSLM/src/corpus.txt"):
    """
    Reads all .txt files from a directory and aggregates them into one large file.

    Args:
        directory_path (str): The path to the directory containing the .txt files.
        output_filename (str): The name of the single output file (will be created in the current working directory).
    """

    print(directory_path)

    # Use glob to find all files matching the pattern, and sort them alphabetically
    # The pattern matches all .txt files within the specified directory
    pattern = os.path.join(directory_path, '*.txt')

    print(pattern)

    file_paths = sorted(glob.glob(pattern))
    
    if not file_paths:
        print(f"No .txt files found in: {directory_path}")
        return

    print(f"Found {len(file_paths)} files to combine. Combining now...")

    # Open the output file in write mode
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            try:
                # Open and read the content of each input file
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    # Write the content to the output file
                    outfile.write(content)
                    # Add a newline for separation between files to prevent merging the last line of one file with the first line of the next
                    outfile.write("\n")
                print(f"  - Appended: {os.path.basename(file_path)}")
            except UnicodeDecodeError:
                print(f"  - Skipped (encoding error): {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  - Error reading {os.path.basename(file_path)}: {e}")

    print(f"\nMerging completed. All text files have been merged into: {output_filename}")

if __name__ == "__main__":
    # Example for the current directory
    aggregate_txt_files("/Users/prathara/Code/SSLM/SSLM/sankrit_corpus")
