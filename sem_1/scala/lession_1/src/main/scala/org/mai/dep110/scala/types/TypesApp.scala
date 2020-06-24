package org.mai.dep110.scala.types

object TypesApp extends App {

  //defining ints
  val int1 = 10
  val int2: Int = 11

  //+ is a method of Int
  val int4 = int2.+(1)

  //infix operators usage
  val int3 = int2 + 1

  //defining string
  val string1 = "String"
  val string2: String = "This is string object"

  string2.split(" ").foreach(println)

  //infix operators execution
  string2 split(" ") foreach println

  //defining list of ints
  val list1 = List(1,2,3,4)
  val list2: List[Integer] = List(1,2,3,4)

  //condition block
  if(list1.isEmpty) {
    println("list1 is empty")
  } else {
    println("list1 is not empty")
  }


  //loop example
  for(i <- 1 to 10) {
    println(i)
  }

  for(i <- 1 until 10) {
    println(s"i = $i")
  }

  //loop with guards
  for(i <- 1 to 10 if i % 2 == 0 if i % 3 != 0) {
    println(i)
  }

  //loop example
  1.to(10).foreach(println(_))
  //loop example infix version
  (1 to 10) foreach println

  //loop example
  val list = List("one", "two", "three", "four")
  for(s <- list) {
    println(s)
  }

  //loop example
  list foreach println

  //loop example
  var i = 1
  while(i < 3) {
    println(i)
    i = i+1
  }

  //reference to function
  val f1: String => Unit = println
  f1("Hello")


  val f2: (String, Int) => List[String] = (s: String, i: Int) => (1 to i).map(_ => s).toList
  println(f2("Test", 3))
  println(f2("Set", 5))



}
