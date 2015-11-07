package nn

import scala.concurrent.{ExecutionContext, Future}
import scala.util.Success

package object training {

  type TrainingAlgorithm[G, T <: Layer[G]] = (T, Mat) => G

  def train[G](updater: TrainingAlgorithm[G, Layer[G]], loss: LossFn)
              (rbm: Layer[G], input: Mat)(implicit ec: ExecutionContext): Future[Layer[G]] = {
    Future.successful(rbm.update(updater(rbm, input))).andThen {
      case Success(rbm) => println(s"loss: ${loss(input, rbm.prop(input))}")
    }
  }

  // RBM specific

  def contrastiveDivergence(rbm: Layer[RBMGradient], batch: Mat)(implicit ec: ExecutionContext): RBMGradient =
    ContrastiveDivergence.diff(rbm, batch, 1)
}


