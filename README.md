# RBM

* Simple RBM implementation with Breeze
* Idiomatic Scala (everything except Breeze matrices is immutable, FP/OOP implementation)
* Contrastive Divergence with Gibbs Sampling as a training algorithm
* Trainer prepared for parallelization (returning Future)

## MNIST

Used to pretrain MNIST classifier (no labels used yet). Idea is to pretrain net with as RBM and then use Feed Forward
process to classify the data with small subset of original labels.

### Example output

* First line is original
* Second line is the reconstruction from NN with 100 neurons, trained on 50000 samples using mini batches of 100 items, 5 epochs.

![Output](https://raw.githubusercontent.com/petrjanda/nn-breeze/master/docs/mnist.png)