import torch
import torchvision 
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

dataset = MNIST(root = 'data/', download = True)


image, label = dataset[100]
plt.imshow(image, cmap = 'gray')
print(f'Label: {label}')

plt.show()