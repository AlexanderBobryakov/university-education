package org.mai.dep110.scala

import org.scalatest._

class MianSpec extends FlatSpec with Matchers {
  "The Main object" should "say hello" in {
    Main.greeting shouldEqual "hello"
  }
}
