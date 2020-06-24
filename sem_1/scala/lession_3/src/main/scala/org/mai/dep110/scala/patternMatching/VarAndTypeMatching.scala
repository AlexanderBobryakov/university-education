package org.mai.dep110.scala.patternMatching

object VarAndTypeMatching extends App{

  val testSeq = Seq(Nil, "test", false, List(1,2), 1, 2.0)

  testSeq
    .map(constMatching)
    .foreach(println)

  def constMatching(a: Any) = a match {
    case 1 => "zero"
    case false => "false"
    case "test" => s"String test"
    case List(1,2) => "List(1,2)"
    case Nil => "empty list"
  }

//  val testTypes = Seq(new Empty, 12, 3l, 2.0d, 231f, "hello", true)
//  testTypes
//    .map(typeMatching)
//    .foreach(println)
//
//  def typeMatching(a: Any) = a match {
//    case i: Int => s"Int: $i"
//    case l: Long => s"Long: $l"
//    case d: Double => s"Double: $d"
//    case s: String => s"String: $s"
//    case e: Empty => e.getClass.getCanonicalName
//  }

}

class Empty