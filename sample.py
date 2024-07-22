import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model with output attentions
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# Example input text
text = "Hi sujith here"

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Get the hidden states and attention weights
hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
attentions = outputs.attentions  # List of attention tensors for each layer

# Accessing attention layer of the first transformer block
if attentions is not None:
    first_layer_attention = attentions[0]  # shape: (batch_size, num_heads, sequence_length, sequence_length)
    # Example of accessing attention weights for a specific attention head
    attention_head_0 = first_layer_attention[:, 0, :, :]  # attention weights for the first head of the first layer
    # You can perform further analysis or visualization with the attention weights as needed

    print("Attention Weights for the First Head of the First Layer:")
    print(attention_head_0)

else:
    print("No attention weights returned by the model.")
