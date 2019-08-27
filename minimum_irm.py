"""This is directly from https://arxiv.org/pdf/1907.02893.pdf"""
import torch
from torch.autograd import grad


def compute_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


def example_1(n=10000, d=2, env=1):
  x = torch.randn(n, d) * env
  y = x + torch.randn(n, d) * env
  z = y + torch.randn(n, d)
  return torch.cat((x, z), 1), y.sum(1, keepdim=True)


phi = torch.nn.Parameter(torch.ones(4, 1))
dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))
opt = torch.optim.SGD([phi], lr=1e-3)
mse = torch.nn.MSELoss(reduction="none")
environments = [example_1(env=0.1), example_1(env=1.0)]
for iteration in range(50000):
  error = 0
  penalty = 0
  for x_e , y_e in environments:
    p = torch.randperm(len(x_e))
    error_e = mse(x_e[p] @ phi * dummy_w, y_e [p])
    penalty += compute_penalty(error_e, dummy_w)
    error += error_e.mean()
  opt.zero_grad()
  (1e-5 * error + penalty).backward()
  opt.step()
  if iteration % 1000 == 0:
    print(phi)
    print('erm error', 1e-5 * error.data.cpu().numpy())
    print('grad penalty', penalty.data.cpu().numpy())
