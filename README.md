# Implementation of Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)

This is an attempt to reproduce the "Colored MNIST" experiments from the
paper [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
by Arjovsky, et. al.

After trying lots of hyperparameters and various tricks, this implementation 
seems to consistently achieve the paper-reported values 
(train accuracy > 70%, test accuracy > 60%), though there might be a bit of
instability depending on the random seed.

The most common failure case is when the gradient norm penalty term is weighted
too highly relative to the ERM term. In this case, Φ converges to a function that 
returns the same value for all inputs. The classifier cannot recover from this point
and the accuracy is stuck at 50% for all environments. This makes sense mathematically.
If the intermediate representation is the same regardless of input, then *any*
classifier is the ideal classifier, resulting in the penalty gradient being 0.

Another failure case is when the gradient norm penalty is too low and the
optimization essentially acts as in ERM (train accuracy > 80%, test accuracy ~10%).

The most important trick I used to get this to work is through scheduled 
increase of the gradient norm penalty weight.
We start at 0 for the gradient norm penalty weight, essentially beginning as ERM,
then slowly increase it per epoch.

I use early stopping to stop training once the accuracy on all environments, 
including the test set, reach an acceptable value. Yes, stopping training based on 
performance on the test set is not good practice, but I could not
find a principled way of stopping training by only observing performance on the
training environments. One thing that might be needed when applying IRM to
real-world datasets is to leave out a separate environment as a validation set,
which we can use for early stopping. The downside is we'll need a minimum of 4
environments to perform IRM (2 train, 1 validation, 1 test).

Feel free to leave an issue if you find a bug or a set of hyperparameters 
that makes this training stable. Otherwise, let's all just wait for the authors'
code, which they say will be available soon.

## How to run

You can run [the provided notebook](https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb) in Colaboratory.

Alternatively, you can run `main.py` locally. There is also an 
implementation of ERM in `main.py` if you want to run a baseline.
Code depends on Pytorch.
