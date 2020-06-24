package org.mai.dep110.scala

object Main extends Greeting with App {
  println(greeting)
  saing("Hello world!")

  private[this] def saing(message: String): Unit = println(message)
}

trait Greeting {
  lazy val greeting: String = "hello"
}
