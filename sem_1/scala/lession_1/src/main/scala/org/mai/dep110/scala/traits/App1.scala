package org.mai.dep110.scala.traits

//object class App1 inherits Trait1 and mixins Trait2
object App1 extends Trait1 with Trait2 {
  def main(s: Array[String]): Unit = {
    m1("Hello")
    m3("Hello")
  }

  override def m2(i: Int): Int = 42
}

//trait with one abstract method
trait Trait1 {
  def m1(s: String): Unit = {
    println("From Trait1")
    println(s)
  }

  def m2(i: Int): Int
}

trait Trait2 {
  def m3(s: String): Unit = {
    println("From Trait2")
    println(s)
  }
}


