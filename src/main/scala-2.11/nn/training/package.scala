package nn

import nn.Trainer.RBMGradient

import scala.concurrent.{ExecutionContext, Future}

package object training {
  type TrainingAlgorithm[T] = (T, Mat) => RBMGradient

  def contrastiveDivergence(rbm: RBMLayer, batch: Mat)(implicit ec: ExecutionContext) =
    Trainer.ContrastiveDivergence.diff(rbm, batch, 1)

  def train[T <: FeedForwardLayer](rbm: T, input: Iterator[Mat], updater: TrainingAlgorithm[T], loss: LossFunction.LossFn)
                                  (implicit ec: ExecutionContext): Future[T] =
    input.foldLeft(Future.successful(rbm)) {
      case (rbm, batch) =>
        rbm.map(r => println(loss(batch, r.prop(batch))))
        rbm.map(r => r.update(updater(r, batch)).asInstanceOf[T])
    }
}


