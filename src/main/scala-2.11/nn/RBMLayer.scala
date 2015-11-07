package nn

import breeze.linalg._
import nn.training.RBMGradient

object RBMLayer {
  def apply(numInput: Int, numOutput: Int, activation: ActivationFn, hiddenActivation: ActivationFn) = {
    val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
    val b = DenseVector.zeros[Double](numOutput)
    val hiddenB = DenseVector.zeros[Double](numInput)

    new RBMLayer(W, b, hiddenB, activation, hiddenActivation)
  }
}

class RBMLayer(val W: Mat, b: Vec, hiddenB: Vec, val activation: ActivationFn, val hiddenActivation: ActivationFn)
  extends FeedForwardLayer(W, b, activation) {
  lazy val numInputs = W.cols
  lazy val numOutputs = W.rows

  def propDown(x: Mat): Mat = {
    val o: Mat = x * W.t

    hiddenActivation(o(*, ::) :+ hiddenB).t
  }

  override def prop(x: Mat): Mat = propDown(propUp(x))

  override def update(g: RBMGradient) =
    new RBMLayer(W :+ g.W, b :+ g.b, hiddenB :+ g.hiddenB, activation, hiddenActivation)
}