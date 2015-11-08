package nn

import scala.concurrent.{ExecutionContext, Future}
import scala.util.Success
import nn.training.Gradient

package object training {

  type TrainingAlgorithm[G, T <: Layer[G]] = (T, Mat) => G

  def setupTrainer[G <: Gradient[G]](algo: TrainingAlgorithm[G, Layer[G]], loss: LossFn)
              (rbm: Layer[G], input: Mat)(implicit ec: ExecutionContext): Future[Layer[G]] = {
    val gradient: G = algo(rbm, input).scale(.1)

    Future.successful(rbm.update(gradient)).andThen {
      case Success(rbm) => println(s"loss: ${loss(input, rbm.prop(input))}")
    }
  }

  def miniBatches[T](mat: Mat, size: Int = 50)
                    (count: Int = mat.cols / (size + 2), epochs: Int = 1): Stream[Mat] = {
    def next = (c: Int) => mat(::, (c * size) to (c * (size + 1)))

    Range(0, count * epochs).toStream.map { i => next(i % count) }
  }

  def train(
    rbm: Layer[RBMGradient],
    trainer: (Layer[RBMGradient], Mat) => Future[Layer[RBMGradient]],
    input: Stream[Mat]
  )(implicit ec: ExecutionContext) =
      input.foldLeft(Future.successful(rbm)) {
        case (rbm, batch) =>
          rbm.flatMap(trainer(_, batch))
      }

  // RBM specific

  def contrastiveDivergence(rbm: Layer[RBMGradient], batch: Mat)(implicit ec: ExecutionContext): RBMGradient =
    ContrastiveDivergence.diff(rbm, batch, 1)
}


