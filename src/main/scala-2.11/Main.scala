import breeze.linalg._
import nn._
import nn.training.RBMGradient

import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  val x = new DenseMatrix(3, 5, Array(
    0.0, 1.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 1.0, 0.0
  ))

  val input = Range(0, 100000).toList.map { _ => x }

  val trainer = training.train[RBMGradient](
    RBMLayer(3, 2, sigmoid, sigmoid), training.contrastiveDivergence _, crossEntropy
  ) _

  trainer(input.toIterator).map { rbm =>
    println(rbm.prop(x))
  }
}
