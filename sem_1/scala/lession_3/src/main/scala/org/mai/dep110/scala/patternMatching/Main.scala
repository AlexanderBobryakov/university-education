package org.mai.dep110.scala.patternMatching

object Main extends App {

  println(printWhatYouGaveMe(42))
  println(printWhatYouGaveMe(Person("Pushkin", "Alexander")))
  println(printWhatYouGaveMe(List("one", 2, 3.0)))



  def printWhatYouGaveMe(x: Any): String = x match {

    // constant patterns
    case 0 => "zero"
    case true => "true"
    case "hello" => "you said 'hello'"
    case Nil => "an empty List"

    // sequence patterns
    case List(0, _, _) => "a three-element list with 0 as the first element"
    case List(1, _*) => "a list beginning with 1, having any number of elements"
    case Vector(1, _*) => "a vector starting with 1, having any number of elements"

    // tuples
    case (a, b) => s"got $a and $b"
    case (a, b, c) => s"got $a, $b, and $c"

    // constructor patterns
    case Person(first, "Alexander") => s"found an Alexander, first name = $first"
    case Dog("Lucky") => "found a dog named Lucky"

    // typed patterns
    case s: String => s"you gave me this string: $s"
    case i: Int => s"thanks for the int: $i"
    case f: Float => s"thanks for the float: $f"
    case a: Array[Int] => s"an array of int: ${a.mkString(",")}"
    case as: Array[String] => s"an array of strings: ${as.mkString(",")}"
    case d: Dog => s"dog: ${d.name}"
    case list: List[_] => s"thanks for the List: $list"
    case m: Map[_, _] => m.toString

      // the default wildcard pattern
    case _ => "Unknown"
  }


//  def sum(a: Seq[Int]): Int = a match {
//    case Nil => 0
//    case x::rest => x + sum(rest)
//  }
//
//  def sort(a: List[Int]): List[Int] = {
//    a match {
//      case Nil => List()
//      case xs::rest => sort(rest.filter(i => i < xs))++(xs::sort(rest.filter(i => i>=xs)))
//    }
//  }
}

case class Person(firstName: String, lastName: String)
case class Dog(name: String)