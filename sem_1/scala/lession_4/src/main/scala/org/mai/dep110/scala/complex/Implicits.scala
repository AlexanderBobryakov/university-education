package org.mai.dep110.scala.complex

case class Complex[A: Arithmetic](re: A, im: A) {
  override def toString: String = s"$re+${im}i"

  def +(that: Complex[A]) = Complex(implicitly[Arithmetic[A]].plus(this.re, that.re), implicitly[Arithmetic[A]].plus(this.im, that.im))
  def -(that: Complex[A]) = Complex(implicitly[Arithmetic[A]].subtract(this.re, that.re), implicitly[Arithmetic[A]].subtract(this.im, that.im))

  def TupleToComplex(tuple2: (A, A)): Complex[A] = {
    Complex(tuple2._1, tuple2._2)
  }
}

trait Arithmetic[A] {
  def plus(a: A, b: A): A
  def subtract(a: A, b: A): A
  def zero: A
}

object Implicits {

  implicit object IntegerComplex extends Arithmetic[Int] {
    override def plus(a: Int, b: Int): Int = a + b
    override def subtract(a: Int, b: Int): Int = a + b
    override def zero: Int = 0
  }

  implicit object DoubleComplex extends Arithmetic[Double] {
    override def plus(a: Double, b: Double): Double = a + b
    override def subtract(a: Double, b: Double): Double = a + b
    override def zero: Double = 0.0
  }

  implicit def TupleToComplex[A : Arithmetic](tuple2: (A, A)): Complex[A] = {
    Complex(tuple2._1, tuple2._2)
  }

  implicit def ToComplex[A](n: A)(implicit a: Arithmetic[A]): Complex[A] = {
    Complex(n, a.zero)
  }

  implicit class ToImaginary[A](im: A)(implicit  a: Arithmetic[A]) {
    def imaginary: Complex[A] = Complex(a.zero, im)
  }
  implicit class ToReal[A](re: A)(implicit  a: Arithmetic[A]) {
    def real: Complex[A] = Complex(re, a.zero)
  }

  implicit class Plus[A: Arithmetic](re: A) {
    def +(n: A) = Complex(re, n)
  }
}

