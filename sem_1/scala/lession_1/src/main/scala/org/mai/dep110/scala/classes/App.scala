package org.mai.dep110.scala.classes

//object class is singleton
object App {

  //program entry point
  def main(args: Array[String]): Unit = {
    val person = new Person("Ivan Ivanov", true, 42)
    println(person)

    val person2 = new Person("Petrov")

    //new word is not required for case class
    val book1 = Book("Title 1", "isbn 1")
    val book2 = Book("Title 2", "isbn 2")
    val book3 = Book("Title 1", "isbn 1")

    //case classes are compared by theirs parameters
    println(book1 == book2)
    println(book1 == book3)

    // case class paramteres are accessable as public fields
    println(book2.isbn)

    //get book field values
    for(i <- book1.productIterator) {
      println(i)
    }

    //unnaply book
    val tuple = Book.unapply(book2).get
    println(tuple)

    val (title, isbn) = Book.unapply(book1).get
    println(s"$title, $isbn")

  }
}

//Class with constructor
class Person(name: String, sex: Boolean, age: Int) {

  //we may override methods
  override def toString: String = {
    s"name: $name, sex: $sex, age: $age"
  }

  def this(name: String) {
    this(name, true, 0)
  }

}

case class Book(title: String, isbn: String)
