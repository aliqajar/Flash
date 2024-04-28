import torch
import torch.nn as nn
from bert_inside import MyBertModel
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# Pretraining configuration
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
max_position_embeddings = 512
type_vocab_size = 2
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.1
layer_norm_eps = 1e-12
intermediate_size = 3072
hidden_act = "gelu"
num_epochs = 10
batch_size = 32
learning_rate = 1e-4




# Load the WikiText-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=max_position_embeddings)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])


# Create data collator
def data_collator(examples):
    input_ids = [torch.tensor(e['input_ids']) for e in examples]
    token_type_ids = [torch.tensor(e['token_type_ids']) for e in examples]
    attention_mask = [torch.tensor(e['attention_mask']) for e in examples]

    # Pad the sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}



# Create data loader
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

# Initialize the BERT model
model = MyBertModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, max_position_embeddings,
                    type_vocab_size, attention_probs_dropout_prob, layer_norm_eps, hidden_dropout_prob,
                    intermediate_size, hidden_act)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Pretraining loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = outputs
        loss = torch.mean(pooled_output.pow(2))  # Dummy loss function using pooled_output
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")