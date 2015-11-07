package nn.training

case class RBMGradient(W: Mat, b: Vec, hiddenB: Vec) {
   def scale(ratio: Double): RBMGradient =
     RBMGradient(W / ratio, b / ratio, hiddenB / ratio)
 }
