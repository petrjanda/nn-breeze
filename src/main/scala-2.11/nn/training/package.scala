package nn

import scala.concurrent.{ExecutionContext, Future}
import scala.util.Success

package object training {

  type TrainingAlgorithm[G, T <: Layer[G]] = (T, Mat) => G

  def train[G](rbm: Layer[G], updater: TrainingAlgorithm[G, Layer[G]], loss: LossFn)
              (input: Iterator[Mat])(implicit ec: ExecutionContext): Future[Layer[G]] = {
    input.foldLeft(Future.successful(rbm)) {
      case (rbm, batch) =>
        rbm.map(r => r.update(updater(r, batch))).andThen {
          case Success(rbm) => println(s"loss: ${loss(batch, rbm.prop(batch))}")
        }
    }
  }

  // RBM specific

  def contrastiveDivergence(rbm: Layer[RBMGradient], batch: Mat)(implicit ec: ExecutionContext): RBMGradient =
    ContrastiveDivergence.diff(rbm, batch, 1)
}


