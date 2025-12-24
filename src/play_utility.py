#
# Code to play with data and find right code to move forward
#
import general_utility as gu
import torch
import time

output_dim = 10
max_length = 4
context_length = max_length

with open("dev.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader, vocab = gu.create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
current_time_seconds = time.time()
torch.manual_seed(current_time_seconds)
embedding_layer = torch.nn.Embedding(vocab.n_vocab, output_dim)
token_embeddings = embedding_layer(inputs)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings = pos_embedding_layer(torch.arange(max_length))
combined_embeddings = token_embeddings + position_embeddings.unsqueeze(0)
#print(combined_embeddings.shape)    # Should be (batch_size, max_length, output_dim)
#print(combined_embeddings)

query = combined_embeddings[0][1]
print(query)
print(combined_embeddings[0])
attn_scores_2 = torch.empty(combined_embeddings[0].shape[0])
for i, x_i in enumerate(combined_embeddings[0]):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(combined_embeddings[0]):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)














# ----------------------------
# import json
# import re
# import sanskrit_tokenizer as st
# import tiktoken as tk
# import sanskrit_data_loader as sdl

# dataloader = gu.create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)

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

# vocab_filename = 'vocab.json'
# vocab_size = 10000
# output_dim = 768

# with open(vocab_filename, "r") as file_handle:
#     vocab = json.load(file_handle)
#     vocab_size = len(vocab)
#     print("Vocab Size:", vocab_size)
