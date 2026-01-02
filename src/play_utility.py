#
# Code to play with data and find right code to move forward
#

#import self_attention as sa

import torch
import time
import sanskrit_llm as sllm
import tiktoken as tk
import general_utility as gu

output_dim = 10
max_length = 4
context_length = max_length

model = sllm.load_sanskrit_llm_model()
tokenizer = tk.get_encoding("gpt2")
batch = []
txt1 = "स ते वीर्यं बलं दर्पमुत्सेकं च तथाविधम्। व्यपनेष्यति गात्रेभ्यः शरवर्षेण संयुगे॥"
txt2 = "स हि देवरसंयुक्तो मम भर्ता महाद्युतिः। निर्भयो वीर्यमाश्रित्य शून्ये वसति दण्डके॥"
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
torch.manual_seed(time.time())
model.eval()
output = gu.generate_text_simple(model=model, idx=batch, max_new_tokens=8, context_size=sllm.GPT_CONFIG_124M["context_length"])
print("Output shape:", output.shape)
print("Output:", output)
print("Decoded Output:", tokenizer.decode(output[0].tolist()))





# ----------------------------
# with open("dev.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# dataloader, vocab = gu.create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# print("Vocab size:", vocab.n_vocab)
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# current_time_seconds = time.time()
# torch.manual_seed(current_time_seconds)
# embedding_layer = torch.nn.Embedding(vocab.n_vocab, output_dim)
# token_embeddings = embedding_layer(inputs)
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# position_embeddings = pos_embedding_layer(torch.arange(max_length))
# combined_embeddings = token_embeddings + position_embeddings.unsqueeze(0)
# print(combined_embeddings.shape)    # Should be (batch_size, max_length, output_dim)
#print(combined_embeddings)

# mha = sa.MultiHeadAttention(output_dim, output_dim, context_length, dropout=0.1, num_heads=2, qkv_bias=False)
# context_vec_sa = mha(combined_embeddings)
#print("Context vector from SelfAttention:", context_vec_sa)

# for i in range(combined_embeddings.shape[0]):
#     query = combined_embeddings[i][1]
#     attn_scores = torch.empty(combined_embeddings[i].shape[0])
#     for j, x_j in enumerate(combined_embeddings[i]):
#         attn_scores[j] = torch.dot(x_j, query)
    
#     attn_weights = torch.softmax(attn_scores, dim=0)
#     # print("Attention weights:", attn_weights)
#     # print("Sum:", attn_weights.sum())

#     context_vec = torch.zeros(query.shape)
#     for k,x_k in enumerate(combined_embeddings[i]):
#         context_vec += attn_weights[k]*x_k
#     print(context_vec)

# first_element = combined_embeddings[0]
# print("First element shape:", first_element.shape)  # Should be (max_length, output_dim)
# print("First element:", first_element)

# W_query = torch.nn.Parameter(torch.rand(output_dim, output_dim), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(output_dim, output_dim), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(output_dim, output_dim), requires_grad=False)

# x_2 = first_element[1]  # Shape: (output_dim,)
# query_2 = torch.matmul(x_2, W_query)  # Shape: (output_dim,)
# key_2 = torch.matmul(x_2, W_key)  # Shape: (output_dim,)
# value_2 = torch.matmul(x_2, W_value)  # Shape: (output_dim,)
# print(query_2)

# keys = torch.matmul(first_element, W_key)  # Shape: (output_dim,)
# values = torch.matmul(first_element, W_value)  # Shape: (output_dim,)
# print(keys.shape)  # Should be (max_length, output_dim)
# print(values.shape)  # Should be (max_length, output_dim)


# attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# d_k = keys.shape[-1]
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)


# sa_v1 = sa.CausalAttention(output_dim, output_dim, context_length, dropout=False, qkv_bias=False)
# for i in range(combined_embeddings.shape[0]):
#     context_vec_sa = sa_v1(combined_embeddings[i])
#     #print("Context vector from SelfAttention:", context_vec_sa)

# sa_v2 = sa.MultiHeadAttentionWrapper(output_dim, output_dim, context_length, dropout=0.5, num_heads=2, qkv_bias=False)
# for i in range(combined_embeddings.shape[0]):
#     context_vec_sa = sa_v2(combined_embeddings[i])
#     print("Context vector from SelfAttention:", context_vec_sa)


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
