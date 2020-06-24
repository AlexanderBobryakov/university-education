package org.mai.dep110.scala.implicits

object ComplexNumberExample extends App {
  case class Complex[T: Numeric](re: T, im: T) {
    override def toString: String = s"$re+${im}i"
    def +(that: Complex[T]) = Complex(implicitly[Numeric[T]].plus(this.re, that.re), implicitly[Numeric[T]].plus(this.im, that.im))
    def -(that: Complex[T]) = Complex(implicitly[Numeric[T]].minus(this.re, that.re), implicitly[Numeric[T]].minus(this.im, that.im))
  }

  val c = Complex(1,2.0) + Complex(3.5,5)

  println(c)
}
