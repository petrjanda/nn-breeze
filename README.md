# RBM

* Simple RBM implementation with Breeze
* Idiomatic Scala (everything except Breeze matrices is immutable, FP/OOP implementation)
* Contrastive Divergence with Gibbs Sampling as a training algorithm
* Trainer prepared for parallelization (returning Future)

## MNIST

Used to pretrain MNIST classifier (no labels used yet). Idea is to pretrain net with as RBM and then use Feed Forward
process to classify the data with small subset of original labels.

### Example output

![Output](https://raw.githubusercontent.com/petrjanda/nn-breeze/master/docs/mnist.png)