package org.mai.dep110.scala.patternMatching

object CaseClassMatching extends App {

  val books = Seq(
    Book(1, "Book 1", Seq(1,2)),
    Book(2, "Book 2", Seq(2,3)),
    Book(3, "Book 3", Seq(4)),
  )

  books
    .map {
      case Book(1, title, _) => s"Book with id 1 has title $title"
      case Book(_, "Book 2", _) => "Book 2 is in the list"
      case b@Book(_, _, authors) if authors.contains(4) => "Books of author 4 is in list"
      case _ => "no match"
    }
    .foreach(println)

//  val authors = Seq(
//    Author(1, "Petrov"),
//    Author(2, "Ivanov"),
//    Author(3, "Fedorov"),
//    Author(4, "Sidorov")
//  )
//
//  //val bookMap = books.map(book => book.id -> book).toMap
//  val authorMap = authors.map(author => author.id -> author).toMap
//  println(books.map { book =>
//
//    (
//      book,
//      authors
//        .foldLeft(List[Author]()) { (acc, author) =>
//          if (book.authors.contains(author.id)) {
//            author :: acc
//          } else {
//            acc
//          }
//        }
//    )
//
////    val authors = book.authors
////      .map(a => authorMap.get(a))
////      .filter{
////        case Some(a) => true
////        case None => false
////      }
////      .map{
////        case Some(author) => author
////      }
////
////    (book, authors)
//
////    val authors = book.authors
////      .map(a => authorMap.getOrElse(a, Author(-1, "")))
////      .filter{
////        case a @ Author(id, name) if id > 0 => true
////        case _ => false
////      }
////
////    (book, authors)
//
//  })

}

case class Book(id:Int, title: String, authors: Seq[Int])
case class Author(id: Int, name: String)
