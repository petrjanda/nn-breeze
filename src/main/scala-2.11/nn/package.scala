import breeze.linalg._

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
  val initial = RBMLayer(3, 2, sigmoid, sigmoid)

  val trainer = training.train[RBMLayer](
    initial, training.contrastiveDivergence _, crossEntropy
  ) _

  trainer(input.toIterator)
}
