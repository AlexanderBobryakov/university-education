package org.mai.dep110.scala.implicits

object ContextBounds extends App {

  trait Summarator[A] {
    def zero: A
    def plus(a: A, b: A): A
  }

  // Implementation for integers
  implicit object IntegerMonoid extends Summarator[Int] {
    override def zero: Int = 0
    override def plus(a: Int, b: Int): Int = a + b
  }

  // Implementation for strings
  implicit object StringMonoid extends Summarator[String] {
    override def zero: String = ""
    override def plus(a: String, b: String): String = a.concat(b)
  }

  // Our generic function that knows which implementation to use based on type parameter 'A'
  def sumWithImplicitParameter[A](values: Seq[A])(implicit ev: Summarator[A]): A = values.foldLeft(ev.zero)(ev.plus)

  //syntax sugar
  def sumWithContextBound[A: Summarator](values: Seq[A]): A = {
    val ev = implicitly[Summarator[A]]
    values.foldLeft(ev.zero)(ev.plus)
  }

  //should sum all elements in list
  val intList = List(1,2,3,4,5,6,7,8,9,10)
  println(sumWithContextBound(intList))
  println(sumWithImplicitParameter(intList))

  val stringSeq = Seq("Hello", " ", "World", "!")
  println(sumWithContextBound(stringSeq))
  println(sumWithImplicitParameter(stringSeq))

}
