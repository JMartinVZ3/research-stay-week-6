# Importing necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Basically what this code does is predict churn by implementing both ltsm and gru and then compare
# the results from both models.

# Assuming we have some historical customer data
# Each record has 2 features (like total amount spent in the last month, number of website visits in the last month) and the sequence length is 5 (5 months data)
data = np.random.rand(100, 5, 2)  # 100 records, each record is a sequence of length 5
target = np.random.randint(
    2, size=100
)  # Whether the customer churned in the 6th month (binary, 1 if churned, 0 otherwise)

# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)


# Implementing the class from the first example
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Implementing the class from the second example
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Forward propagate GRU
        out, _ = self.gru(x, h0.detach())

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Initialize an LSTM and GRU
input_size = 2
hidden_size = 16
num_layers = 1
output_size = 1

epochs = 1000

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizers
criterion = nn.BCEWithLogitsLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.01)

# Training loop for LSTM
for epoch in range(epochs):
    lstm_optimizer.zero_grad()
    outputs = lstm_model(data)
    loss = criterion(outputs.view(-1), target)
    loss.backward()
    lstm_optimizer.step()

    if epoch % 100 == 0:  # Print an update every 100 epochs
        print(f"Epoch {epoch} / {epochs}, LSTM Loss: {loss.item()}")

# Training loop for GRU
for epoch in range(epochs):
    gru_optimizer.zero_grad()
    outputs = gru_model(data)
    loss = criterion(outputs.view(-1), target)
    loss.backward()
    gru_optimizer.step()

    if epoch % 100 == 0:  # Print an update every 100 epochs
        print(f"Epoch {epoch} / {epochs}, GRU Loss: {loss.item()}")
