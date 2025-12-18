#
# Code to play with data and find right code to move forward
#

import re
import json
from . import sanskrit_tokenizer as ST

# with open("/Users/prathara/Code/SSLM/SSLM/src/corpus.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

#     preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
#     preprocessed = [item.strip() for item in preprocessed if item.strip()]
#     all_words = sorted(set(preprocessed))
#     vocab = {token:integer for integer,token in enumerate(all_words)}

#     # serialize and write the vocab to a file
#     vocab_filename = '/Users/prathara/Code/SSLM/SSLM/src/vocab.json'
#     # open the file in write mode ('w') and use json.dump()
#     try:
#         with open(vocab_filename, 'w') as json_file:
#             json.dump(vocab, json_file, indent=4)
#     except IOError as e:
#         print(f"Error writing to file: {e}")

vocab_filename = '/Users/prathara/Code/SSLM/SSLM/src/vocab.json'
vocab = ''

try:
    with open(vocab_filename, "r") as file_handle:
        vocab = json.load(file_handle)

    # # Now you can use the data as a regular Python dictionary
    # for i, item in enumerate(data_dict.items()):
    #     print(item)
    #     if i >= 50:
    #         break
except FileNotFoundError:
    print("Error: The file 'data.json' was not found.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from the file.")

sanskrit_tokenizer = ST.SanskritTokenizer(vocab)
print(sanskrit_tokenizer.encode("सम्बन्धादेवमन्यत्रापि"))
