import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from my_nltk import tokanize, stem, bag_of_words

from model import NeuralNet

# try:
with open('intents.json', 'r') as f:
    intent = json.load(f)
    # print(intent)
"""
except FileNotFoundError:
    print("File not found")

except json.decoder.JSONDecodeError:
    print("error Error decoding JSON data in 'intents.json'. Please check the file contents.")
"""
all_words = []
tags = []
xy = []

for intent in intent['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokanize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
# print(all_words)

tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples



# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNet(input_size, hidden_size, output_size)

# loss fn
criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.long().to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backpropagration
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

print(device)

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
#define file name
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

