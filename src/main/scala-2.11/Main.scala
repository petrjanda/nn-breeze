import nn._
import nn.training._
import utils.{FileRepo, Plot, _}

import scala.concurrent.Await
import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  val nn = new FileRepo("nn/")

  def loadMnist(examples: Int) =
    MNIST.read(
      "data/train-labels-idx1-ubyte",
      "data/train-images-idx3-ubyte",
      Some(examples)
    )._1

  args.headOption match {
    case Some("train") =>
      val input = miniBatches(loadMnist(60000), size = 100)(epochs = 10)
      val init: Layer[RBMGradient] = RBMLayer(input.numFeatures, 100, sigmoid, sigmoid)

      val trainer = setupTrainer(
        algo = training.contrastiveDivergence _,
        loss = crossEntropy,
        learning = annealing(.1, input.numIterations)
      ) _

      import scala.concurrent.duration._

      val rbm = Await.result(train(
        rbm = init,
        trainer = trainer,
        input = input.stream
      ), 120 seconds)

      nn.save(rbm, "mnist.o")
      println("Done.")

    case Some("reconstruct") =>
      val rbm = nn.load[Layer[RBMGradient]]("mnist.o").get
      val x = loadMnist(7)
      val reconstruction = rbm.prop(x)

      val p = new Plot(196,84)

      MNIST.drawMnistSample(p, 7, x)
      MNIST.drawMnistSample(p, 7, rbm.W, (0, 1))
      MNIST.drawMnistSample(p, 7, reconstruction, (0, 2))

      p.plot

    case _ => throw new IllegalArgumentException("Missing command!")
  }
}


