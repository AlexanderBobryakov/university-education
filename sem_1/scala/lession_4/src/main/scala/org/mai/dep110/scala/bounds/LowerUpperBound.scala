package org.mai.dep110.scala.bounds

object LowerUpperBound extends App{
  val animal = new Animal
  val dog = new Dog
  val shep = new Shepherd
  val cat = new Cat

  val animalCarer = new AnimalCarer

  //animalCarer.upperDisplay(animal)
  animalCarer.upperDisplay(dog)
  animalCarer.upperDisplay(shep)


  animalCarer.lowerDisplay(animal)
  animalCarer.lowerDisplay(shep)
  animalCarer.lowerDisplay(dog)
  animalCarer.lowerDisplay(cat)
}


class Animal
class Dog extends Animal
class Shepherd extends Dog
class Cat extends Animal

class AnimalCarer{
  def upperDisplay [T <: Dog](t: T){
    println(s"Upper: $t")
  }

  def lowerDisplay [T >: Shepherd](t: T){
    println(s"Lower: $t")
  }
}