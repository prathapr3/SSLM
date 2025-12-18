#
# Code to play with data and find right code to move forward
#

import re
import json
import sanskrit_tokenizer as st
import tiktoken as tk
import sanskrit_data_loader as sdl
import general_utility as gu

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

# vocab_filename = 'vocab.json'

# try:
#     with open(vocab_filename, "r") as file_handle:
#         vocab = json.load(file_handle)

#     sanskrit_tokenizer = st.SanskritTokenizer(vocab)
#     print(sanskrit_tokenizer.decode(sanskrit_tokenizer.encode("श्वेतोधावति")))
#     print(sanskrit_tokenizer.decode(sanskrit_tokenizer.encode("अर्थः अस्मद्युपपदे समानाभिधेये सति प्रयुज्यमानेऽप्यप्रयुज्यमानेऽप्युत्तमपुरुषो भवति।")))
#     print(sanskrit_tokenizer.decode(sanskrit_tokenizer.encode("इयं हि कस्यापि करोति किंचित् तपस्विनी राजवरस्य पुत्री। या चीरमासाद्य वनस्य मध्ये जाता विसंज्ञा श्रमणीव काचित्॥")))

#     bpe_tokenizer = tk.get_encoding("gpt2")
#     print(bpe_tokenizer.decode(bpe_tokenizer.encode("श्वेतोधावति")))
#     print(bpe_tokenizer.decode(bpe_tokenizer.encode("अर्थः अस्मद्युपपदे समानाभिधेये सति प्रयुज्यमानेऽप्यप्रयुज्यमानेऽप्युत्तमपुरुषो भवति।")))
#     print(bpe_tokenizer.decode(bpe_tokenizer.encode("इयं हि कस्यापि करोति किंचित् तपस्विनी राजवरस्य पुत्री। या चीरमासाद्य वनस्य मध्ये जाता विसंज्ञा श्रमणीव काचित्॥")))

#     # # Now you can use the data as a regular Python dictionary
#     # for i, item in enumerate(data_dict.items()):
#     #     print(item)
#     #     if i >= 50:
#     #         break
# except FileNotFoundError:
#     print("Error: The file 'data.json' was not found.")
# except json.JSONDecodeError:
#     print("Error: Could not decode JSON from the file.")

with open("dev.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = gu.create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader = gu.create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

