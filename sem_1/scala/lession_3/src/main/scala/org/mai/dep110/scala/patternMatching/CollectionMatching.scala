package org.mai.dep110.scala.patternMatching

import scala.collection.Traversable

object CollectionMatching extends App {

  Array(2, "string", 2.0, Pet("Barsik", 5)) match {
    case Array(i, s, d: Double, p @ Pet(n, a)) if n == "Barsik" => println(s"$i, $s,double is $d,Barsik is $p")

    case Array(i, s, d: Double, p) => println(s"$i, $s,double is $d, $p")
    case Array(i, s, d, p) => println(s"$i, $s, $d, $p")
    case Array(i, s: String, _*) if i == 2 => println(s"First is 2, second element is string: $s")
    case Array(_, s: String, _*) => println(s"second element is string: $s")
    case a: Array[Any] => println("it is array")
    case _ => print("will catch everything")
  }

  Seq(1,2,3,4) match {
    case a::b::c::Nil => println(s"Three items: $a, $b, $c")
    case a::b::c::d::Nil => println(s"Four items: $a, $b, $c, $d")
  }



  println(m((Seq(1,2,3,4,5,6))))

  def m(seq: Traversable[Int]): Int = seq match {
    case head::tail => Math.max(head, m(tail))
    case Nil => Int.MinValue
  }

  println( qs( Seq(2,5,3,8,6,5,9,8,3,7,9) ) )

  def qs(c: Traversable[Int]): Traversable[Int] = c match {
    case head::tail => qs(tail.filter(_ < head)) ++ Traversable(head) ++ qs(tail.filter(_ >= head))
    case Nil => Traversable[Int]()
  }

}

case class Pet(nickName: String, age: Int)
