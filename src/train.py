import torch
import time
import sanskrit_llm as sllm
import tiktoken as tk
import general_utility as gu

file_path = "dev.txt"
text_data = ""
model = sllm.load_sanskrit_llm_model()
tokenizer = tk.get_encoding("gpt2")

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# take only 20% of the original data for faster experimentation
text_data = text_data[:int(0.2 * len(text_data))]

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(time.time())

train_loader, _ = gu.create_dataloader(train_data, batch_size=2, max_length=sllm.GPT_CONFIG_124M["context_length"], stride=sllm.GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)
val_loader, _ = gu.create_dataloader(val_data, batch_size=2, max_length=sllm.GPT_CONFIG_124M["context_length"], stride=sllm.GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0)

# Sanity check
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

if total_tokens * (train_ratio) < sllm.GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `sllm.GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < sllm.GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `sllm.GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 3
train_losses, val_losses, tokens_seen = gu.train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="स ते वीर्यं बलं दर्पमुत्सेकं च तथाविधम्। व्यपनेष्यति गात्रेभ्यः", tokenizer=tokenizer)
print("Training complete.")

# Save the trained model
torch.save(model.state_dict(), "sanskrit_llm_model.pth")
print("Model saved to sanskrit_llm_model.pth")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
gu.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=100, eval_iter=10, start_context="", tokenizer=None):
#     train_losses = []
#     val_losses = []
#     track_tokens_seen = []
#     tokens_seen = 0
#     global_step = -1

#     for epoch in range(num_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = gu.calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = gu.evaluate_model(model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

#         gu.generate_and_print_sample(model, tokenizer, device, start_context)
#     return train_losses, val_losses, track_tokens_seen

