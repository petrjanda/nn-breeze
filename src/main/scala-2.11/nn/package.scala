import breeze.linalg._
import breeze.numerics.{log, sigmoid => sig, tanh => th}
import breeze.stats._
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import nn.training.RBMGradient
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
    def constant(rate: Double): Int => Double = { _ => 1.0 - rate }
  }


  type LossFn = (Mat, Mat) => Double

  val crossEntropy: LossFn = (y, a) => {
    val p = -y :* log(a) :+ (1.0 - y) :* log(1.0 - a)

    mean(sum(p(::, *)))
  }


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

  object NNRand extends RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(12345)))

  object RBMLayer {
    def apply(numInput: Int, numOutput: Int, activation: ActivationFn, hiddenActivation: ActivationFn) = {
      val W = DenseMatrix.rand(numInput, numOutput, NNRand.uniform)
      val b = DenseVector.zeros[Double](numOutput)
      val hiddenB = DenseVector.zeros[Double](numInput)

      new RBMLayer(W, b, hiddenB, activation, hiddenActivation)
    }
  }

  class RBMLayer(val W: Mat, b: Vec, hiddenB: Vec, val activation: ActivationFn, val hiddenActivation: ActivationFn) extends FeedForwardLayer(W, b, activation) {
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

  def printMat(n: String, m: Mat) = {
    println(s"$n: ${m.rows}, ${m.cols}")
    println(m.data.toList)
  }

  def printVec(n: String, m: Vec) =
    println(s"$n: ${m.size}")
}

object Main extends App {
  import nn._
  import scala.concurrent.ExecutionContext.Implicits.global

  val x = new DenseMatrix(3, 5, Array(
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 1.0, 0.0
  ))

  val input = Range(0, 100000).toList.map { _ => x }
  val initial = nn.RBMLayer(3, 2, nn.Activation.sigmoid, nn.Activation.sigmoid)

  val trainer = training.train[RBMLayer](
    initial, training.contrastiveDivergence _, crossEntropy
  ) _

  trainer(input.toIterator)
}
