import torch
from torch.autograd import grad
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr


def prepare_colored_mnist():
  train_mnist = datasets.mnist.MNIST('./data', train=True, download=True)
  print(len(train_mnist))

  train1_set = []
  train2_set = []
  test_set = []
  for idx, (im, label) in enumerate(train_mnist):
    if idx % 10000 == 0:
      print(f'Converting image {idx}')
    im_array = np.array(im)

    # Assign a binary label y to the image based on the digit
    binary_label = 0 if label < 5 else 1

    # Flip label with 25% probability
    if np.random.uniform() < 0.25:
      binary_label = binary_label ^ 1

    # Color the image either red or green according to its possibly flipped label
    color_red = binary_label == 0

    # Flip the color with a probability e that depends on the environment
    if idx < 20000:
      # 20% in the first training environment
      if np.random.uniform() < 0.2:
        color_red = not color_red
    elif idx < 40000:
      # 10% in the first training environment
      if np.random.uniform() < 0.1:
        color_red = not color_red
    else:
      # 90% in the test environment
      if np.random.uniform() < 0.9:
        color_red = not color_red

    colored_arr = color_grayscale_arr(im_array, red=color_red)

    if idx < 20000:
      train1_set.append((Image.fromarray(colored_arr), binary_label))
    elif idx < 40000:
      train2_set.append((Image.fromarray(colored_arr), binary_label))
    else:
      test_set.append((Image.fromarray(colored_arr), binary_label))

    print('original label', type(label), label)
    print('binary label', binary_label)
    print('assigned color', 'red' if color_red else 'green')
    plt.imshow(colored_arr)
    plt.show()

    break

  torch.save(train1_set, './data/ColoredMNIST/train1.pt')
  torch.save(train2_set, './data/ColoredMNIST/train2.pt')
  torch.save(test_set, './data/ColoredMNIST/test.pt')


def main():
  prepare_colored_mnist()


if __name__ == '__main__':
  main()
