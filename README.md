# Implementation of Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)

This is an attempt to reproduce the "Colored MNIST" experiments from the
paper [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
by Arjovsky, et. al.

After trying lots of hyperparameters and various tricks, I was able to "sort of"
achieve the results in the paper. I say "sort of" because the training process
is very unstable and dependent on the random seed. I would say about ~70% of 
runs will converge to a respectable value (train accuracy > 70%, test accuracy > 60%)
after a few tens of epochs.

The most common failure case is when the gradient norm penalty term is weighted
too highly relative to the ERM term. In this case, Φ converges to a function that 
returns the same value for all inputs. The classifier cannot recover from this point
and naturally, the accuracy is stuck at 50% for all environments.

Another failure case is when the gradient norm penalty is too low and the
optimization essentially acts as in ERM (train accuracy > 80%, test accuracy ~10%).

The most important trick I used to get this to work is through scheduled 
increase of the gradient norm penalty weight.
We start at 0 for the gradient norm penalty weight, essentially beginning as ERM,
then slowly increase it per epoch.

Feel free to leave an issue if you find a bug or a set of hyperparameters 
that makes this training stable. Otherwise, let's all just wait for the authors'
code, which they say will be available soon.

## How to run

Code depends on Pytorch.

Just run `main.py`. There is also an implementation of ERM in `main.py`
if you want to run a baseline.
