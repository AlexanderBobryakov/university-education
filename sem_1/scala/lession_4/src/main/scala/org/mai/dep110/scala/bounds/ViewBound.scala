package org.mai.dep110.scala.bounds

object ViewBound extends App {

  implicit def rangeToInt(r: Range) = r.sum
  implicit def stringToInt(r: String) = r.length

  val ranges = new Maths[Range]
  println(ranges.max(1 to 10, 2 to 11))

  val strings = new Maths[String]
  println(strings.max("Hello World!", "Hello Scala!!"))
}

class Maths[T <% Int]() {
  def max(t1: T, t2: T) = if (t1 > t2) t1 else t2
}
