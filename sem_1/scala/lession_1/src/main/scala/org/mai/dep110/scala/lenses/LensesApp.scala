package org.mai.dep110.scala.lenses

import scala.io.Source

object LensesApp extends App {

  val lenses = Source
    .fromFile("lenses.data")
    .getLines()
    .map{ line => parse(line.split(","))}
    .filter{
      case Some(_) => true
      case None => false
    }
    .map{
      case Some(lense) => lense
    }
    .toList

  println(lenses)

  def parse(parts: Array[String]): Option[Lense] = {
    if(parts.length != 5) {
      None
    } else {
      Some(Lense(
        parts(0).toInt,
        parts(1).toInt,
        parts(2).toInt,
        if(parts(3).toInt == 1) true else false,
        parts(4).toInt))
    }
  }
}

case class Lense(lenseClass: Int, patientAge: Int, spectaclePrescription: Int, isAstigmatic: Boolean, tearProductionRate: Int) {
}
