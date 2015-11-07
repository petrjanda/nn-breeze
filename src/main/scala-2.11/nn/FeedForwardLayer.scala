package nn

import breeze.linalg.{*, DenseVector, DenseMatrix}
import nn.training.RBMGradient

// NEURAL NETWORKS
object FeedForwardLayer {
  def apply(numInput: Int, numOutput: Int, activation: ActivationFn) = {
    val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
    val b = DenseVector.rand(numOutput, NNRand.uniform)

    new FeedForwardLayer(W, b, activation)
  }
}

class FeedForwardLayer(W: Mat, b: Vec, activation: ActivationFn) {
  def propUp(x: Mat): Mat = {
    val o: Mat = x.t * W

    activation(o(*, ::) :+ b)
  }

  def prop(x: Mat): Mat = propUp(x)

  def update(g: RBMGradient) =
    new FeedForwardLayer(W :+ g.W, b :+ g.b, activation)
}