# import torch
# from transformers import GPT2Tokenizer, GPT2Model

# # Load pre-trained GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')

# # Define a function to profile the feed-forward layers
# def profile_feed_forward(inputs):
#     with torch.autograd.profiler.profile() as prof:
#         # Define hooks to profile specific layers
#         def hook(module, input, output):
#             return output
        
#         # Register hook to the feed-forward layers
#         hook_handle = model.register_forward_hook(hook)
#         # Perform forward pass
#         outputs = model(inputs)
#         # Unregister hook
#         hook_handle.remove()
#     print(prof)

# inputs = torch.tensor(tokenizer.encode("Hello, how are you?", return_tensors="pt"))
# profile_feed_forward(inputs)


import torch
from transformers import GPT2Tokenizer, GPT2Model

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Define a function to profile the feed-forward layers
def profile_feed_forward(inputs):
    def feed_forward_hook(module, input, output):
        return output
    
    handles = []
    # Register hooks on feed-forward layers (in GPT-2, these are typically part of the MLP)
    for layer in model.h:
        handle = layer.mlp.register_forward_hook(feed_forward_hook)
        handles.append(handle)
    
    # Perform forward pass with profiling
    with torch.autograd.profiler.profile() as prof:
        outputs = model(inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()

    # Print the profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Example of profiling the feed-forward layers
inputs = tokenizer.encode("Hello, how are you?", return_tensors="pt")
profile_feed_forward(inputs)
