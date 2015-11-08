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

case class RBMLayer(val W: Mat, b: Vec, hiddenB: Vec, val activation: ActivationFn, val hiddenActivation: ActivationFn)
  extends Layer[RBMGradient] {
  lazy val numInputs = W.cols
  lazy val numOutputs = W.rows

  def propUp(x: Mat): Mat = {
    val o: Mat = x.t * W

    activation(o(*, ::) :+ b)
  }

  def propDown(x: Mat): Mat = {
    val o: Mat = x * W.t

    hiddenActivation(o(*, ::) :+ hiddenB).t
  }

  def prop(x: Mat): Mat = propDown(propUp(x))

  override def update(g: RBMGradient): Layer[RBMGradient] =
    this.copy(W =  W :+ g.W, b = b :+ g.b, hiddenB = hiddenB :+ g.hiddenB)
}