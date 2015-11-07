package nn

import scala.concurrent.{ExecutionContext, Future}

package object training {
  type TrainingAlgorithm[T] = (Future[T], Mat) => Future[T]

  def contrastiveDivergence(rbm: Future[RBMLayer], batch: Mat)(implicit ec: ExecutionContext) =
    rbm.map(r => r.update(Trainer.ContrastiveDivergence.diff(r, batch, 1)))

  def train[T](rbm: T, input: Iterator[Mat], updater: TrainingAlgorithm[T]): Future[T] =
    input.foldLeft(Future.successful(rbm)) { updater }
}
