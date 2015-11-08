import breeze.linalg.DenseMatrix
import nn._

package object utils {
  def drawSample(p: Plot, count: Int, mat: Mat, pos: (Int, Int) = (0, 0)) = {
    Range(0, count).foreach { i =>
      p.draw(DenseMatrix.create[Double](28, 28, mat(::, i).toDenseVector.data), (28 * (i + pos._1), 28 * pos._2))
    }
  }
}
