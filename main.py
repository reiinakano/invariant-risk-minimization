import torch
from torch.autograd import grad
from torchvision import datasets


def main():
  train_mnist = datasets.mnist.MNIST('./data', train=True, download=True)
  test_mnist = datasets.mnist.MNIST('./data', train=False, download=True)
  print(len(train_mnist))
  print(len(test_mnist))

if __name__ == '__main__':
  main()
