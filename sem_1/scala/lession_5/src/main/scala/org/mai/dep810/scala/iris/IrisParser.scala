package org.mai.dep810.scala.iris

import scala.io.Source
import scala.util.Try

object IrisParser {
  def loadFromFile(path: String): List[Iris] = {
    Source
      .fromFile(path)
      .getLines
      .map(line => line.toIris)
      .filter{
        case Some(iris) => true
        case None => false
      }
      .map{
        case Some(iris) => iris
      }
      .toList
  }

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
