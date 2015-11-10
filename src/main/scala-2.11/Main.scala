import breeze.linalg.DenseMatrix
import nn._
import nn.training._
import utils.{FileRepo, Plot, _}

import scala.concurrent.{Future, Await}
import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  val repo = new FileRepo("nn/")

  def loadMnist(examples: Int) =
    MNIST.read(
      "data/train-labels-idx1-ubyte",
      "data/train-images-idx3-ubyte",
      Some(examples)
    )._1

  args.headOption match {
    case Some("train") =>
      val input = miniBatches(loadMnist(30000), size = 100)(epochs = 5)
      val init: Layer[RBMGradient] = RBMLayer(input.numFeatures, 10, sigmoid, sigmoid)

      val trainer: (Layer[RBMGradient], Mat, Int) => Future[Layer[RBMGradient]] =
        setupTrainer(
          algorithm = contrastiveDivergence _,
          loss = crossEntropy,
          learning = annealing(.1, input.numIterations)
        ) _

      import scala.concurrent.duration._

      val rbm = Await.result(train(
        rbm = init,
        trainer = trainer,
        input = input.stream
      ), 120 seconds)

      repo.save(rbm, "mnist.o")
      println("Done.")

    case Some("reconstruct") =>
      val rbm = repo.load[Layer[RBMGradient]]("mnist.o").get
      val x = loadMnist(31)
      val reconstruction = rbm.prop(x)

      val p = new Plot(6 * 28, 56)

      MNIST.drawMnistSample(p, 10 to 15, x)
      MNIST.drawMnistSample(p, 10 to 15, reconstruction, (0, 1))

      p.plot

    case _ => throw new IllegalArgumentException("Missing command!")
  }
}


