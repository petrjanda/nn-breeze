import breeze.linalg._
import breeze.numerics.{log, sigmoid => sig, tanh => th}
import breeze.stats._
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister

package object nn {
  type Mat = DenseMatrix[Double]
  type Vec = DenseVector[Double]


  // Activation functions

  type ActivationFn = Mat => Mat

  val linear: ActivationFn = (x: Mat) => x

  val sigmoid: ActivationFn = (x: Mat) => sig(x)

  val tanh: ActivationFn = (x: Mat) => th(x)


  // Learning function

  type LearningFn = Int => Double

  def constant(rate: Double): LearningFn = { _ => rate }


  // Loss functions

  type LossFn = (Mat, Mat) => Double

  val crossEntropy: LossFn = (y, a) => {
    val p = -y :* log(a) :+ (1.0 - y) :* log(1.0 - a)

    mean(sum(p(::, *)))
  }


  // Auxiliary

  object NNRand extends RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(12345)))

  def printMat(n: String, m: Mat) = {
    println(s"$n: ${m.rows}, ${m.cols}")
    println(m.data.toList)
  }

  def printVec(n: String, m: Vec) =
    println(s"$n: ${m.size}")
}