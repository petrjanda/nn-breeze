import nn._
import nn.training._
import utils.{FileRepo, Plot, _}

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
      val x = loadMnist(1500)
      val init: Layer[RBMGradient] = RBMLayer(784, 100, sigmoid, sigmoid)
      val trainer = setupTrainer(training.contrastiveDivergence _, crossEntropy) _
      val input = miniBatches(x, size = 10)(epochs = 20)

      train(init, trainer, input).map {
        rbm =>
          nn.save(rbm, "mnist.o")
          println("Done.")
      }

    case Some("reconstruct") =>
      val rbm = nn.load[Layer[RBMGradient]]("mnist.o").get
      val x = loadMnist(7)
      val reconstruction = rbm.prop(x)

      val p = new Plot(196,84)

      drawSample(p, 7, x)
      drawSample(p, 7, rbm.W, (0, 1))
      drawSample(p, 7, reconstruction, (0, 2))

      p.plot

    case _ => throw new IllegalArgumentException("Missing command!")
  }
}


