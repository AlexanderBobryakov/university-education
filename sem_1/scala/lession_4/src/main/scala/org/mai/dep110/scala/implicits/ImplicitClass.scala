package org.mai.dep110.scala.implicits

import scala.util.Try

object ImplicitClass extends App {

  implicit class Trigonometry(d: Double) {
    def cos: Double = Math.cos(d)
    def sin: Double = Math.sin(d)
    def tan: Double = Math.tan(d)
    def ctan: Double = 1/Math.tan(d)
  }

  println(Math.PI.cos)
  println(2.sin)
  println(16.0.tan)

  val irisLine = "5.7,4.4,1.5,0.4,Iris-setosa"

  val iris = irisLine.toIris.get

  println(iris)

  implicit class StringToIris(str: String) {
    def toIris: Option[Iris] = str.split(",") match {
      case Array(a,b,c,d,e) if isDouble(a) && isDouble(b) && isDouble(c) && isDouble(d) =>
        Some(
          Iris(
            a.toDouble,
            b.toDouble,
            c.toDouble,
            d.toDouble,
            e))
      case others => None
    }

    def isDouble(str: String): Boolean = Try(str.toDouble).isSuccess
  }
}

case class Iris(sepalLength: Double, sepalWidth: Double, petalLength: Double, petalWidth: Double, species: String)

