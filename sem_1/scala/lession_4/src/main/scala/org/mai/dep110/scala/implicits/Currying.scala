package org.mai.dep110.scala.implicits

object Currying extends App {
  //обычная функция с типом (Int, Int) => Int
  def f(a: Int, b: Int) = a * b

  //каррированная функция с типом Int => (Int => Int)
  def g(a: Int)(b: Int) = a * b

  //каррированная функция,
  // полученная из обычной с типом Int => (Int => Int)
  def curryedF: Int => Int => Int = (f _).curried

  //распечатает true
  println( g(2)(3) == curryedF(2)(3) )


  //частично определенная функция из обычной
  def partialF = f(10, _: Int)

  //частично определенная функция из карриованной
  def partialG = g(10) _

  println( partialG(3) == partialF(3) )

  //декаррирование функции
  def decurriedG = Function.uncurried(g _)
  println(decurriedG(1,2))
}
