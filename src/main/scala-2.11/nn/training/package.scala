package nn

import scala.concurrent.{ExecutionContext, Future}
import scala.util.Success

package object training {

  type TrainingAlgorithm[G, T <: Layer[G]] = (T, Mat) => G

  def setupTrainer[G <: Gradient[G]](
    algorithm: TrainingAlgorithm[G, Layer[G]],
    loss: LossFn,
    learning: LearningFn
  )(
    rbm: Layer[G],
    input: Mat,
    iteration: Int
  )(implicit ec: ExecutionContext): Future[Layer[G]] = {
    val gradient: G = algorithm(rbm, input).scale(learning(iteration))

    Future.successful(rbm.update(gradient)).andThen {
      case Success(rbm) =>
        if(iteration % 100 == 0 || iteration == 1)
          println(s"iteration: $iteration, loss: ${loss(input, rbm.prop(input)) / input.cols / input.rows}, rate: ${learning(iteration)}")
    }
  }


  case class DataSet(stream: Stream[Mat], numIterations: Int, numFeatures: Int)

  def miniBatches[T](mat: Mat, size: Int = 50)
                    (count: Int = mat.cols / (size + 2), epochs: Int = 1): DataSet = {
    def next = (c: Int) => mat(::, (c * size) to (c * (size + 1)))

    DataSet(
      Range(0, count * epochs).toStream.map { i => next(i % count) }, 
      numIterations = count * epochs,
      numFeatures = mat.rows
    )
  }

  def train(
    rbm: Layer[RBMGradient],
    trainer: (Layer[RBMGradient], Mat, Int) => Future[Layer[RBMGradient]],
    input: Stream[Mat]
  )(implicit ec: ExecutionContext) = {
    var i = 0

    input.foldLeft(Future.successful(rbm)) {
      case (rbm, batch) =>
        rbm.flatMap { rbm =>
          i += 1
          trainer(rbm, batch, i)
        }
    }
  }

  // RBM specific

  def contrastiveDivergence(rbm: Layer[RBMGradient], batch: Mat)(implicit ec: ExecutionContext): RBMGradient =
    ContrastiveDivergence.diff(rbm, batch, 1)
}


