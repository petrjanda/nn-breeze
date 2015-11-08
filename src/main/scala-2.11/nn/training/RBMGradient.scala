package nn.training

import nn._

trait Gradient[T] {
  def scale(ratio: Double): T
}

case class RBMGradient(W: Mat, b: Vec, hiddenB: Vec) extends Gradient[RBMGradient] {
   def scale(ratio: Double): RBMGradient =
     RBMGradient(W * ratio, b * ratio, hiddenB * ratio)
 }
