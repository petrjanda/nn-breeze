import breeze.linalg.DenseMatrix
import org.scalatest.{Matchers, FreeSpec}

class CrossEntropyTest extends FreeSpec with Matchers {
  "Should be -> 0 for similar matrices" - {
    val y = new DenseMatrix(3, 2, Array(
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0
    ))

    val a = new DenseMatrix(3, 2, Array(
      0.01, 0.01, 0.99,
      0.21, 0.69, 0.11
    ))

    nn.LossFunction.crossEntropy(y, a) should equal(-0.4266118986072549)
  }

  "Should have error for know values case" - {
    val y = new DenseMatrix(3, 3, Array(
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0
    ))

    val a = new DenseMatrix(3, 3, Array(
      0.1,  0.3, 0.6,
      0.2,  0.6, 0.2,
      0.3,  0.4, 0.3
    ))

    nn.LossFunction.crossEntropy(y, a) should equal(-1.0471265439125454)
  }

}
