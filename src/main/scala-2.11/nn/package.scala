import breeze.linalg._
import breeze.numerics.{sigmoid => sig, tanh => th, log}
import breeze.stats._
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis, Rand, Binomial}
import nn.Trainer

import org.apache.commons.math3.random.MersenneTwister


package object nn {
  type Mat = DenseMatrix[Double]

  type Vec = DenseVector[Double]

  type ActivationFn = Mat => Mat


  // FUNCTIONS

  object Activation {
    val linear: ActivationFn = (x: Mat) => x

    val sigmoid: ActivationFn = (x: Mat) => sig(x)

    val tanh: ActivationFn = (x: Mat) => th(x)
  }

  object LearningFunction {
    val constant = (iteration: Int) => 1

    def annealing(rate: Double): Int => Double = { _ => 1.0 - rate }
  }

  object LossFunction {
    val crossEntropy: (Mat, Mat) => Double = (y, a) => {
      val p = -y :* log(a) :+ (1.0 - y) :* log(1.0 - a)

      mean(sum(p(::, *)))
    }
  }


  // NEURAL NETWORKS

  trait Layer {
    def W: Mat
    def b: Vec
    def activation: Mat => Mat

    def propUp(x: Mat): Mat
  }

  trait BiDirectionalLayer extends Layer {
    def hiddenB: Vec // hidden bias
    def hiddenActivation: Mat => Mat

    def propDown(x: Mat): Mat
  }

  object FFLayer {
    def apply(numInput: Int, numOutput: Int, activation: ActivationFn) = {
      val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
      val b = DenseVector.rand(numInput, NNRand.uniform)

      new FFLayer(W, b, activation)
    }
  }

  class FFLayer(W: Mat, b: Vec, activation: ActivationFn) {
    def propUp(x: Mat): Mat = {
      val o = x.t * W

      activation(o(*, ::) :+ b)
    }
  }

  object NNRand extends RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(1234)))

  object RBMLayer {
    def apply(numInput: Int, numOutput: Int, activation: ActivationFn, hiddenActivation: ActivationFn) = {
      val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
      val b = DenseVector.zeros[Double](numOutput)
      val hiddenB = DenseVector.zeros[Double](numInput)

      new RBMLayer(W, b, hiddenB, activation, hiddenActivation)
    }
  }

  class RBMLayer(val W: Mat, b: Vec, hiddenB: Vec, val activation: ActivationFn, val hiddenActivation: ActivationFn) extends FFLayer(W, b, activation) {
    lazy val numInputs = W.cols
    lazy val numOutputs = W.rows

    def propDown(x: Mat): Mat = {
      val o: Mat = x * W.t

      hiddenActivation(o(*, ::) :+ hiddenB).t
    }

    def prop(x: Mat): Mat = propUp(propDown(x))

    def update(g: Trainer.RBMGradient) = {
      new RBMLayer(W :+ g.W, b :+ g.b, hiddenB :+ g.hiddenB, activation, hiddenActivation)
    }
  }

  def printMat(n: String, m: Mat) =
    println(s"$n: ${m.rows}, ${m.cols}")

  def printVec(n: String, m: Vec) =
    println(s"$n: ${m.size}")

  def trainRbm(rbm: RBMLayer, input: Iterator[Mat]): RBMLayer =
    input.foldLeft(rbm) {
      case(l, in) =>
        l.update(Trainer.ContrastiveDivergence.diff(l, in, 1))
    }
}

object Main extends App {
  val x = new DenseMatrix(2, 1, Array(
    0.0, 1.0,
    0.0, 1.0,
    1.0, 1.0
  ))

  val input = Range(0, 100000).toList.map { _ => x }
  val initial = nn.RBMLayer(2, 3, nn.Activation.sigmoid, nn.Activation.sigmoid)
  val rbm = nn.trainRbm(initial, input.toIterator)

  println(rbm.propDown(rbm.propUp(x)))
}
