import torch
import torch.nn as nn

class XORMLP(nn.Module):
    def __init__(self):
        # This takes care of initialising and remembering learned weights and biases
        # See your handwritten notes for PyTorch handles these and their gradients
        super().__init__()

        self.hidden1 = nn.Linear(2, 4) # Input layer (2 nodes) to Hidden Layer 1 (4 nodes)
        self.hidden2 = nn.Linear(4, 4) # Hidden Layer 1 (4 nodes) to Hidden Layer 2 (4 nodes)
        self.output = nn.Linear(4, 1) # Hidden Layer 2 (4 nodes) to Output Layer (1 node)
        
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Instantiate the model
model = XORMLP()
print(model)