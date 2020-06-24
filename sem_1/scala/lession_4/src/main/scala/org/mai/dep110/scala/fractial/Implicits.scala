package org.mai.dep110.scala.fractial

case class Fractional[A: Arithmetic](numerator: A, denominator: A) {
  override def toString: String = s"${numerator}/${denominator}"

  val value: A = implicitly[Arithmetic[A]].divide(numerator, denominator)

  def +(that: Fractional[A]) = {
    val num1 = implicitly[Arithmetic[A]].multiply(this.numerator, that.denominator)
    val num2 = implicitly[Arithmetic[A]].multiply(this.denominator, that.numerator)
    val num = implicitly[Arithmetic[A]].plus(num1, num2)
    val denum = implicitly[Arithmetic[A]].multiply(this.denominator, that.denominator)

    Fractional(num, denum)
  }

  def -(that: Fractional[A]) = {
    val num1 = implicitly[Arithmetic[A]].multiply(this.numerator, that.denominator)
    val num2 = implicitly[Arithmetic[A]].multiply(this.denominator, that.numerator)
    val num = implicitly[Arithmetic[A]].subtract(num1, num2)
    val denum = implicitly[Arithmetic[A]].multiply(this.denominator, that.denominator)

    Fractional(num, denum)
  }

}

object Implicits {
  implicit object IntArithmetic extends Arithmetic[Int] {
    override def one: Int = 1

    override def plus(a: Int, b: Int): Int = a + b

    override def subtract(a: Int, b: Int): Int = a - b

    override def multiply(a: Int, b: Int): Int = a * b

    override def divide(n: Int, d: Int): Int = n/d
  }

  implicit object LongArithmentic extends Arithmetic[Long] {
    override def one: Long = 1l

    override def plus(a: Long, b: Long): Long = a + b

    override def subtract(a: Long, b: Long): Long = a - b

    override def multiply(a: Long, b: Long): Long = a * b

    override def divide(a: Long, b: Long): Long = a / b
  }

  implicit def intToFractional(n: Int): Fractional[Int] = {
    Fractional(n, 1)
  }
  implicit def longToFractional(n: Long): Fractional[Long] = {
    Fractional(n, 1)
  }

  implicit class FractionalConvertion[A: Arithmetic](a: A) {
    def toNumerator = Fractional(a, implicitly[Arithmetic[A]].one)
    def toDenumerator = Fractional(implicitly[Arithmetic[A]].one, a)

    def :/ (denum: A) = {
      Fractional(a, denum)
    }
  }
}

trait Arithmetic[A] {
  def one: A
  def plus(a:A, b:A): A
  def subtract(a:A, b:A): A
  def multiply(a:A, b:A): A
  def divide(a:A, b:A): A
}