from pypdf import PdfReader
import io

def read_pdf_line_by_line(pdf_file_path):
    """
    Reads a PDF file and yields each line of text.

    Args:
        pdf_file_path (str): The path to the PDF file.
    """
    try:
        with open(pdf_file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                # Extract text and split into lines, handling potential None return
                text = page.extract_text()
                if text:
                    for line in text.strip().splitlines():
                        yield line
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")


if __name__ == "__main__":
    # Example usage:
    file_path = 'sanskritwordlist.pdf' # Replace with your actual PDF file name
    i=0
    for line in read_pdf_line_by_line(file_path):
        i+=1
        print(line.strip()) # Use .strip() to remove leading/trailing whitespace
        if i>100: # Limit the output to the first 100 lines
            break

