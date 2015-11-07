import breeze.linalg._
import nn._
import nn.training.RBMGradient
import utils.FileRepo

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

object Main extends App {

  val x = new DenseMatrix(3, 5, Array(
    0.0, 1.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 1.0, 0.0
  ))

  val nn = new FileRepo("nn/")

  args.headOption match {
    case Some("train") =>
      val input = Range(0, 100000).toList.map { _ => x }

      val trainer = training.train(
        training.contrastiveDivergence _, crossEntropy
      ) _

//      RBMLayer(3, 2, sigmoid, sigmoid)
//      trainer(input.toIterator).map { rbm =>
//        nn.save(rbm, "rbm.o")
//        println("Done.")
//      }

    case Some("predict") =>
      val rbm = nn.load[RBMLayer]("rbm.o")

      println(rbm.map(_.prop(x)))

    case Some("mnist") =>
      val x = MNIST.read("data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte", Some(10000))

      import breeze.plot._

      import breeze.stats._

      println(mean(x._1(::, 1)))

//      Range(0, 5).toList.foreach { i =>
//        val f = Figure()
//        f.subplot(0) += image(DenseMatrix.create(28, 28, x._1(::, i).data))
//        f.saveas(s"images/test-$i.png")
//      }

      val init: Layer[RBMGradient] = RBMLayer(784, 100, sigmoid, sigmoid)
      val trainer = training.train(
        training.contrastiveDivergence _, crossEntropy
      ) _

      val rbm = Range(0, 1000).toList.foldLeft(Future.successful(init)) { case(rbm, i) => rbm.flatMap { r => trainer(r, x._1(::, 10*i to 10*(i+1))) } }

      rbm.map { rbm =>
        nn.save(rbm, "mnist.o")
        println("Done.")
      }

    case _ => throw new IllegalArgumentException("Missing command!")
  }
}


