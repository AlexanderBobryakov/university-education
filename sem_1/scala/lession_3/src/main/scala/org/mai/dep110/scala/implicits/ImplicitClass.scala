package org.mai.dep110.scala.implicits

import scala.util.Try

object ImplicitClass extends App {

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

