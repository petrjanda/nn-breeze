package nn

trait Layer[G] {
  def W: Mat

  def numInputs: Int

  def numOutputs: Int

  def propUp(x: Mat): Mat

  def propDown(x: Mat): Mat

  def prop(x: Mat): Mat

  def update(g: G): Layer[G]
}
