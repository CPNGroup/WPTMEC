import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset from a local file
data = pd.read_csv('data/samples.csv')
X = data.iloc[:, :-1].values.astype(np.float32)
y = data.iloc[:, -1].values.astype(np.int64)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler object
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.fc1 = nn.Linear(60, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 11)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 5000
loss_values = []  # Store loss values for plotting
for epoch in range(num_epochs):
    inputs = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, Loss: %.4f" % (epoch, loss.item()))
    loss_values.append(loss.item())  # Append loss value to the list

plt.plot(range(num_epochs), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Change in Loss over Epochs')
plt.show()

torch.save(model.state_dict(), 'model/model.pth')

