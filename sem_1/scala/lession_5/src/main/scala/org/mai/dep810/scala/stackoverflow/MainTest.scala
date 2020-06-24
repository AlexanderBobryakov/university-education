package org.mai.dep810.scala.stackoverflow

import java.io.{BufferedWriter, File, FileWriter}
import java.time.LocalDateTime

import org.log4s._
import scalikejdbc.config._

object MainTest extends App{

  private[this] val logger = getLogger

  DBs.setup(stackOverflowDB)

  val helper = new DBHelper {
    override def db: Symbol = stackOverflowDB
  }

  helper.dropTables
  helper.createTables
  helper.clearData

  helper.saveData(Seq(
    User(1,"name 1", "location 1", 1, 1, 1, 1, 1, LocalDateTime.now, LocalDateTime.now),
    User(2,"name 2", "location 2", 2, 2, 2, 2, 2, LocalDateTime.now, LocalDateTime.now)
  ), Seq(), Seq())


  val bf = new BufferedWriter(new FileWriter(new File("test.csv")))
  helper
    .extract("select * from users")
    .foreach{ line =>
      bf.write(line)
      bf.newLine()
    }
  bf.flush()
  bf.close()
}
