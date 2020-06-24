package org.mai.dep110.scala.patternMatching

object Regexp extends App{

  //digits matcher
  val intMatcher = "(\\d+)".r
  val s = Seq("123", "2.0", "123d")
  s.map{
    case intMatcher(num) => num
    case other => s"$other is not number"
  }.foreach(println(_))


  //email matcher
  val strings = Seq(
  "I do CrossFit and run (a lot).  If needed, contact me via email at bluefeet@stackoverflow.com",
  "I work for our Stack Exchange overlords on the Community Growth team. If you'd like to tell me what you had for lunch, you can email me at abby@stackoverflow.com. ",
  "A for professional and enthusiast programmers Pranav Ram at Stack Overflow, A for professional and enthusiast programmers pranavcbalan@gmail.com. Loves to solve problems, learn new things and help others.",
  "Statistically insignificant, gets lucky all the time."
  )

  val emailChecker = ".*\\s(\\w+@\\w+\\.\\w+).*".r

  strings
    .map{ string =>
      string match {
        case emailChecker(email) => email
        case other => "No enail provided."
      }
    }
    .foreach(println(_))

}
