from torch.utils.data import DataLoader
import tiktoken as tk
from sanskrit_data_loader import GPTDatasetV1
from transformers import AutoTokenizer

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tk.get_encoding("gpt2")
    # Load with trust_remote_code=True because of custom logic
    # tokenizer = AutoTokenizer.from_pretrained("ArthaLabs/panini-tokenizer", trust_remote_code=True)

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader, tokenizer
