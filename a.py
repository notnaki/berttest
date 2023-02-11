import torch
import transformers
from transformers import BertForQuestionAnswering, AdamW
from b import preprocess_squad
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
num_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
count = 0
print(count)
# Load the preprocessed SQuAD data
input_ids, attention_mask, token_type_ids, start_positions, end_positions = preprocess_squad('train-v2.0.json')
print(count)
count += 1
# Load the fine-tuned BERT model
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
model = model.to(device)
print(count)
count += 1
# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
print(count)
count += 1
dataset = TensorDataset(input_ids, attention_mask, token_type_ids, start_positions, end_positions)
print(count)
count += 1
# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
print(count)
count += 1
# Create the DataLoader for the training set
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(count)
count += 1
# Define the loss function
criterion = torch.nn.CrossEntropyLoss()
print(count)
count += 1
# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (input_ids_batch, attention_mask_batch, token_type_ids_batch, start_positions_batch, end_positions_batch) in enumerate(train_dataloader):
        # Move the inputs to the GPU
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        token_type_ids_batch = token_type_ids_batch.to(device)
        start_positions_batch = start_positions_batch.to(device)
        end_positions_batch = end_positions_batch.to(device)

        # Zero the gradients
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids_batch, attention_mask_batch, token_type_ids_batch)
        start_loss = criterion(start_logits, start_positions_batch)
        end_loss = criterion(end_logits, end_positions_batch)
        loss = start_loss + end_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print the loss for every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished training!')
