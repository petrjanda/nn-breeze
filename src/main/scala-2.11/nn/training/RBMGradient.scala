package nn.training

import nn._

case class RBMGradient(W: Mat, b: Vec, hiddenB: Vec) {
   def scale(ratio: Double): RBMGradient =
     RBMGradient(W / ratio, b / ratio, hiddenB / ratio)
 }
