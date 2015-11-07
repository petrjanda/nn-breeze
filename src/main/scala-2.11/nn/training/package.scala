package nn

import scala.concurrent.{ExecutionContext, Future}
import scala.util.Success

package object training {
  type TrainingAlgorithm[T] = (T, Mat) => RBMGradient

  def contrastiveDivergence(rbm: RBMLayer, batch: Mat)(implicit ec: ExecutionContext): RBMGradient =
    ContrastiveDivergence.diff(rbm, batch, 1)

  def train[T <: FeedForwardLayer](rbm: T, updater: TrainingAlgorithm[T], loss: LossFn)
                                  (input: Iterator[Mat])
                                  (implicit ec: ExecutionContext): Future[T] =
    input.foldLeft(Future.successful(rbm)) {
      case (rbm, batch) =>
        rbm.map(r => r.update(updater(r, batch)).asInstanceOf[T]).andThen {
          case Success(rbm) => println(s"loss: ${loss(batch, rbm.prop(batch))}")
        }
    }
}


