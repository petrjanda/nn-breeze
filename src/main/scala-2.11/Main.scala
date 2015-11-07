import breeze.linalg._
import nn._
import utils.FileRepo

import scala.concurrent.ExecutionContext.Implicits.global

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
        RBMLayer(3, 2, sigmoid, sigmoid), training.contrastiveDivergence _, crossEntropy
      ) _

      trainer(input.toIterator).map { rbm =>
        nn.save(rbm, "rbm.o")
        println("Done.")
      }

    case Some("predict") =>
      val rbm = nn.load[RBMLayer]("rbm.o")

      println(rbm.map(_.prop(x)))

    case _ => throw new IllegalArgumentException("Missing command!")
  }
}


