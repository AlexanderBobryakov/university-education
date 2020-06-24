package org.mai.dep110.scala.collections

import Sex.Sex

import scala.collection.mutable.HashSet
import scala.collection.JavaConverters._

object Main extends App {

  val persons = Seq(
    Person("Ivan", "Ivanov", 40, Sex.Male),
    Person("Petr", "Petrov", 30, Sex.Male),
    Person("Svetlana", "Svetlova", 25, Sex.Female),
    Person("Tatiana", "Kuznetsova", 35, Sex.Female),
    Person("Olga", "Fedorova", 45, Sex.Female)
  )
  println(persons)

  val personsConcatenated = persons ++ Seq(Person("Elena", "Dmitrieva", 42, Sex.Female), Person("Sidor", "Sidorov", 27, Sex.Male))
  println(personsConcatenated)



//  val personsMutable = HashSet[Person](
//    Person("Ivan", "Ivanov", 40, Sex.Male),
//    Person("Petr", "Petrov", 30, Sex.Male),
//    Person("Svetlana", "Svetlova", 25, Sex.Female),
//    Person("Tatiana", "Kuznetsova", 35, Sex.Female),
//    Person("Olga", "Fedorova", 45, Sex.Female)
//  )
//  personsMutable += Person("Elena", "Dmitrieva", 42, Sex.Female)
//  println(personsMutable)

//  val personsJava = persons.asJava
//  println(personsJava)
//  //will throw java.lang.UnsupportedOperationException
//  personsJava.add(Person("Elena", "Dmitrieva", 42, Sex.Female))

  //map example
//  personsConcatenated
//    .map{ person =>
//      s"${person.firstName} ${person.lastName}"
//    }
//    .foreach(println(_))

  //flatMap example
//  val flatNames = personsConcatenated.flatMap(person => List(person.firstName, person.lastName))
//  val names = personsConcatenated.map(person => List(person.firstName, person.lastName))
//  println(flatNames)
//  println(names)
//
//  //toSet example
//  println(flatNames.toSet)

  //obtain collection elements
//  println(persons.head)
//  println(persons(3))

  //obtain subcollection
//  println(persons.drop(2))
//  println(persons.take(2))
//  println(persons.filter(p => p.sex == Sex.Female))

//  val partitioned = personsConcatenated.partition(p => p.sex == Sex.Male)
//  println(partitioned.getClass)
//  println(partitioned._1)
//  println(partitioned._2)
//
//  val grouped = personsConcatenated.groupBy(p => p.sex)
//  println(grouped)

//  val personsMap1 = personsConcatenated.map(p => p.lastName -> p).toMap
//  val personsMap2 = personsConcatenated.map(p => (p.lastName, p)).toMap
//  println(personsMap1)
//  println(personsMap2)

  println(
    personsConcatenated.map(_.lastName).reduceLeft(_ + ", " + _)
  )

  println(
    personsConcatenated.map(_.lastName).mkString(", ")
  )

  val tuple = personsConcatenated
    .foldLeft(Tuple2[Int, Int](0,0)) { (accamulator, person) =>
      (accamulator._1 + person.age, accamulator._2+1)
    }
  val avg = tuple._1.toDouble/tuple._2.toDouble
  println(avg)


  println(
    personsConcatenated.map(p => p.age).min
  )

  println(
    personsConcatenated.minBy(p => p.age)
  )


}

object Sex extends Enumeration {
  type Sex = Value
  val Male, Female = Value
}

case class Person(firstName: String, lastName: String, age: Int, sex: Sex)

