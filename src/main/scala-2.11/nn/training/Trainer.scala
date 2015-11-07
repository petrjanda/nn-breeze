package nn.training

import breeze.linalg.{*, DenseMatrix}
import breeze.stats._
import breeze.stats.distributions.Binomial
import nn._

case class RBMGradient(W: Mat, b: Vec, hiddenB: Vec) {
  def scale(ratio: Double): RBMGradient =
    RBMGradient(W / ratio, b / ratio, hiddenB / ratio)
}

object ContrastiveDivergence {
  def diff(nn: RBMLayer, input: Mat, k: Int) = {
    val gibbs = new GibbsSampler(nn)
    val probHidden = gibbs.sampleHiddenGivenVisible(input)

    val hvh = Range(0, k).foldLeft(
      GibbsSampler.HVHSample(
        GibbsSampler.Sample(
          DenseMatrix.zeros(input.rows, nn.numInputs),
          DenseMatrix.zeros(input.rows, nn.numOutputs)
        ), probHidden
      )
    )((old, _) => {
      gibbs.sampleHiddenVisibleHidden(old.nhSample)
    })

    val wGradient = input * probHidden.sample - hvh.nvSamples * hvh.nhMeans

    val h: Mat = probHidden.sample - hvh.nhMeans
    val hBiasGradient: Mat = mean(h(::, *))

    val v: Mat = input - hvh.nvSamples
    val vBiasGradient: Mat = mean(v.t(::, *))

    // Swap biases!
    RBMGradient(wGradient, hBiasGradient.t(::, 0), vBiasGradient.t(::, 0))
  }
}


object GibbsSampler {
  case class Sample(mean: Mat, sample: Mat)

  case class HVHSample(nvMean: Mat, nvSamples: Mat, nhMeans: Mat, nhSample: Mat)

  object HVHSample {
    def apply(vh: Sample, hv: Sample): HVHSample = HVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}

class GibbsSampler(rbm: RBMLayer) {
  import GibbsSampler._

  val isBinomialVisible = rbm.activation == nn.Activation.sigmoid
  val isBinomialHidden = rbm.hiddenActivation == nn.Activation.sigmoid

  def sampleHiddenGivenVisible(v: Mat) = {
    val mean = rbm.propUp(v)

    Sample(mean, sample(mean, isBinomialHidden))
  }

  def sampleVisibleGivenHidden(h: Mat) = {
    val mean = rbm.propDown(h)

    Sample(mean, sample(mean, isBinomialVisible))
  }

  def sampleHiddenVisibleHidden(h: Mat) = {
    val vh = sampleVisibleGivenHidden(h)
    val hv = sampleHiddenGivenVisible(vh.sample)

    HVHSample(vh, hv)
  }

  private def sample(mean: Mat, isBinomial: Boolean) =
    mean.map { v => Binomial(1, v).sample.toDouble }
}