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

# Example of profiling the attention layers with outliers
inputs = tokenizer.encode("write a 1000 word essay about LLM Pruning", return_tensors="pt")

# Add outliers to the input tensor
# For example, let's set a specific token ID (e.g., 1000) to represent an outlier
outlier_index = 5  # Index where the outlier will be inserted
outlier_token_id = 50000  # Token ID representing the outlier
inputs[0, outlier_index] = outlier_token_id

profile_attention(inputs)
