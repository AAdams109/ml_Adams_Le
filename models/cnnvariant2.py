import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import time
import psutil
import os
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

#Device Configuration 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#Shared Parameters Across Models 
vocab_size = 10000
max_length = 100
embed_dim = 128
batch_size = 32
num_epochs = 10
learning_rate = 0.001

#Load and Train-Test Split Data: when testing out the 3 datasets, merely edit which CVS is being read 
df = pd.read_csv('datasets/WELFake_subdatasets/WELFake_sub_dataset_1.csv')
full_article = df["title"] + " " + df["text"]
labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(full_article, labels, test_size=0.2, random_state=42, stratify=labels)

#Tokenize
def tokenize(text):
    tokens = text.lower().split()
    return tokens[:max_length]

tokenized_train = [tokenize(text) for text in X_train]

#Build Vocab
vocab = build_vocab_from_iterator(tokenized_train, specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

#Encode and Pad 
def encode(text):
    return torch.tensor([vocab[token] for token in tokenize(text)])

X_train_encoded = [encode(text) for text in X_train]
X_test_encoded = [encode(text) for text in X_test]

X_train_padded = pad_sequence(X_train_encoded, batch_first=True, padding_value=vocab["<pad>"])
X_test_padded = pad_sequence(X_test_encoded, batch_first=True, padding_value=vocab["<pad>"])

#Tensors
X_train_tensor = X_train_padded
X_test_tensor = X_test_padded
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#Dataloaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

#CNN Model (3 conv layers, 2 pool layers)
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters=64, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.mean(dim=2)  
        return self.fc(x)

#Get Memory Usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 
    #Convert From Bytes to Megabytes 


#Initialize Model 
model = CNN(len(vocab), embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

#Evaluate Model 
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_pred = torch.sigmoid(model(x))
            pred = (y_pred >= 0.5).int().squeeze(-1)
            preds.extend(pred.tolist())
            labels.extend(y.tolist())
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return accuracy, precision, recall, f1

#Train Model 
results = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end_time = time.time()

    #After Each Epoch, Save Data For CSV
    memory = get_memory_usage()
    accuracy, precision, recall, f1 = evaluate(model, test_loader)

    results.append({
        "epoch": epoch + 1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": total_loss / len(train_loader),
        "train_time": end_time - start_time,
        "memory": memory
    })

#Save Results Into CSV for Model Comparison 
df = pd.DataFrame(results)
#Change the Name of CSV Depending on What Dataset is Being Read. 
df.to_csv("datasets/cnn2_dataset1.csv", index=False)
