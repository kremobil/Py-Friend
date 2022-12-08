import json

import numpy
from torch.testing._internal.common_nn import input_size

from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('responses.json', 'r') as file:
    responses = json.load(file)

all_words = []
tags = []
xy = []
for intent in responses:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ingnore_words = ['?', '!', '.', ',', ':']
all_words = [stem(x) for x in all_words if x not in ingnore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words, tags)

X_train = []
Y_train = []
for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
print(input_size, len(all_words))
print(output_size, tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, output_size)

#loss and optimzer
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (word, label) in train_loader:
        words = word.to(device)
        labels = label.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (epoch +1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')