# import torch
# from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

# # Load pre-trained GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# config = GPT2Config.from_pretrained('gpt2', output_attentions=True)
# model = GPT2Model.from_pretrained('gpt2', config=config)

# # Define a function to profile only the attention layers using emit_itt
# def profile_attention(inputs):
#     def attention_hook(module, input, output):
#         with torch.autograd.profiler.emit_itt():
#             return output
    
#     handles = []
#     # Register hooks on attention layers
#     for layer in model.h:
#         handle = layer.attn.register_forward_hook(attention_hook)
#         handles.append(handle)
    
#     # Perform forward pass
#     outputs = model(inputs)
    
#     # Remove hooks
#     for handle in handles:
#         handle.remove()

#     # Note: Profiling data will be collected by VTune, not printed here
#     print("Profiling complete. Check VTune Profiler for detailed results.")

# # Example of profiling the attention layers
# inputs = tokenizer.encode("write an 1000 word essay about LLM Pruning", return_tensors="pt")
# profile_attention(inputs)


import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2', output_attentions=True)
model = GPT2Model.from_pretrained('gpt2', config=config)

# Define a function to profile only the attention layers
def profile_attention(inputs):
    def attention_hook(module, input, output):
        return output
    
    handles = []
    # Register hooks on attention layers
    for layer in model.h:
        handle = layer.attn.register_forward_hook(attention_hook)
        handles.append(handle)
    
    # Perform forward pass with profiling
    with torch.autograd.profiler.profile() as prof:
        outputs = model(inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()

    # Print the profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Example of profiling the attention layers
inputs = tokenizer.encode("Hello, how are you?", return_tensors="pt")
profile_attention(inputs)
