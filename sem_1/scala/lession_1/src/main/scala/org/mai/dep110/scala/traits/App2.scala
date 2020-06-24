package org.mai.dep110.scala.traits

object App2 {

  def main(args: Array[String]): Unit = {
    val person = new Person("Ivan Ivanov", 42)
    person.sleep(600)
    person.eat("bread")
    person.work("writing code")
    person.play("chess")
    person.teach("teaching")

    val p = new Worker("Petr Petrov", 45) with LazyPerson {
      override def idling(): Unit = println("idling")
    }
    p.idling
  }
}

abstract class Human (name: String, age: Int) {
  def sleep(period: Int): Unit
  def eat(food: String)

}

class Worker(name: String, age: Int) extends Human(name, age) {
  override def sleep(period: Int): Unit = {println("sleeping")}

  override def eat(food: String): Unit = {
    println("eating")
    thinking("my thoughts")
  }

  def work(job: String): Unit = {println("working")}

  private[this] def thinking(thouths: String): Unit = println(thouths)
}

trait Player {
  def play(game: String): Unit = {println("playing")}
}

trait Teacher {
  def teach(game: String): Unit = {println("teach")}
}

trait LazyPerson {
  def idling(): Unit
}

class Person(name: String, age: Int) extends Worker(name, age) with Player with Teacher {

}




