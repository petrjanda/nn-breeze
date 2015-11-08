package utils

import breeze.linalg.DenseMatrix
import nn._

class Plot(rows: Int, cols: Int) {
  val data = DenseMatrix.zeros[Double](rows, cols)

  def draw(m: Mat, coords: (Int, Int)) = {
    data(
      coords._1 to (coords._1 + m.rows - 1),
      coords._2 to (coords._2 + m.cols - 1)
    ) := m
  }

  def plot = {
    Range(0, data.cols).foreach { i =>
      println(data(::, i).toDenseVector.data.map {i => if(i > 0.5) "x" else "." }.mkString(""))
    }
  }
}