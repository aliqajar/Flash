import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import math
import torch.nn.init as init

checkpoint_name = "cp_bert"
load_from_checkpoint = False
checkpoint_frequency = 5

# Parameters
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
max_length = 128
batch_size = 32
learning_rate = 5e-5
weight_decay = 1e-2
num_epochs = 1000

# params
# vocab_size = 30522
# hidden_size = 100
# num_hidden_layers = 4
# num_attention_heads = 4
# intermediate_size = 200
# max_position_embeddings = 128
# max_length = 32
# batch_size = 256
# learning_rate = 1e-4
# weight_decay = 1e-2
# num_epochs = 100    


class BertModel(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            num_attention_heads, 
            intermediate_size,
            max_position_embeddings):

        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

    
    def forward(self, input_ids, attention_mask):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        for layer  in self.layers:
            embeddings = layer(embeddings, attention_mask)
        return embeddings
    

class BertLayer(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, intermediate_size):

        super().__init__()

        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        residual = hidden_states + attention_output
        layernorm_output = self.layernorm1(residual)
        feed_forward_output = self.feed_forward(layernorm_output)
        residual = layernorm_output + feed_forward_output
        layernorm_output = self.layernorm2(residual)
        return layernorm_output
    

class BertAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    def forward(self, hidden_states, attention_mask=None):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :] * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    

class BertForNextTokenPrediction(nn.Module):

    def __init__(self, base_bert_model, vocab_size):
        super().__init__()

        self.bert = base_bert_model
        self.predictor = nn.Linear(base_bert_model.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        prediction_scores = self.predictor(outputs)
        return prediction_scores
    

# methods
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0, std=0.1)
    elif isinstance(module, nn.LayerNorm):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

def save_checkpoint(state, filename=checkpoint_name, directory='.'):
    checkpoint_dir = os.path.join(directory, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    epoch = state['epoch']
    formatted_filename = f"{filename}_{epoch}.pth.tar"
    torch.save(state, os.path.join(checkpoint_dir, formatted_filename))

def load_latest_checkpoint(checkpoint_name, directory='.'):
    checkpoint_dir = os.path.join(directory, checkpoint_name)
    pattern = f'{checkpoint_name}_*.pth.tar'
    list_of_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint = torch.load(latest_file)
    return checkpoint


def preprocess_function(examples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    texts = [text[:max_length-1] + "[SEP]" for text in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

def load_pretrained_weights(self, pretrained_model_path):
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
    self.load_state_dict(pretrained_state_dict, strict=False)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset and tokenizer
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'token_type_ids'])
    train_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    val_encoded_dataset = val_dataset.map(preprocess_function, batched=True)
    val_encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataloader = DataLoader(val_encoded_dataset, batch_size=32, shuffle=False)        

    base_bert_model = BertModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings).to(device)

    model = BertForNextTokenPrediction(base_bert_model, vocab_size).to(device)
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    start_epoch = 0
    best_val_loss = float('inf')

    # Optional checkpoint loading
    if load_from_checkpoint:
        checkpoint = load_latest_checkpoint(checkpoint_name)
        if checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print("Training checkpoint loaded successfully.")
        else:
            print("No training checkpoint found, starting from scratch.")


    # Training loop
    for epoch in range(start_epoch, num_epochs):

        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids, labels = input_ids[:, :-1], input_ids[:, 1:]
            attention_mask = attention_mask[:, :-1]

            outputs = model(input_ids, attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.reshape(-1, vocab_size), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        model.eval()
        val_total_loss = 0
        val_total_items = 0

        with torch.no_grad():
            for batch in val_dataloader:
                val_input_ids = batch['input_ids'].to(device)
                val_attention_mask = batch['attention_mask'].to(device)
                val_input_ids, val_labels = val_input_ids[:, :-1], val_input_ids[:, 1:]
                val_attention_mask = val_attention_mask[:, :-1]

                val_outputs = model(val_input_ids, val_attention_mask)
                val_loss_fn = nn.CrossEntropyLoss()
                val_loss = val_loss_fn(val_outputs.reshape(-1, val_outputs.size(-1)), val_labels.reshape(-1))

                val_total_loss += val_loss.item() * val_input_ids.size(0)
                val_total_items += val_input_ids.size(0)            

        train_avg_loss = total_loss / len(train_dataloader)
        val_avg_loss = val_total_loss / val_total_items
        print(f"Epoch {epoch+1}, Training Avg Loss: {train_avg_loss:.2f}, Evaluation Ave Loss: {val_avg_loss:.2f}")

        if epoch % checkpoint_frequency == 0:
            best_val_loss = min(train_avg_loss, best_val_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': loss,
                'val_loss': val_loss,
            })


if __name__ == "__main__":
    main()