package nn

import breeze.linalg.{*, DenseMatrix, DenseVector}
import nn.training.RBMGradient

// NEURAL NETWORKS
object FeedForwardLayer {
  def apply(numInput: Int, numOutput: Int, activation: ActivationFn) = {
    val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
    val b = DenseVector.rand(numOutput, NNRand.uniform)

    new FeedForwardLayer(W, b, activation)
  }
}

case class FeedForwardLayer(W: Mat, b: Vec, activation: ActivationFn) extends Layer[RBMGradient] {
  lazy val numInputs = W.cols
  lazy val numOutputs = W.rows

  def propUp(x: Mat): Mat = {
    val o: Mat = x.t * W

    activation(o(*, ::) :+ b)
  }

  def prop(x: Mat): Mat = propUp(x)

  def propDown(x: Mat): Mat = ???

  def update(g: RBMGradient): Layer[RBMGradient] = ???
}