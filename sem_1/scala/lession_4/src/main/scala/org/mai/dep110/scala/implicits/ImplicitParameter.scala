package org.mai.dep110.scala.implicits

import java.sql.{Connection, DriverManager}

import scala.collection.mutable

object ImplicitParameter extends App {

  implicit val connection = DriverManager.getConnection("jdbc:h2:mem:basket", "", "")

  createTable
  addAbonent(Abonent(1, "Petrov"))(connection)
  addAbonent(Abonent(2, "Sidorov"))
  addAbonent(Abonent(3, "Ivanov"))
  getAbonents.foreach(println(_))


  def createTable(implicit connection: Connection): Unit = {
    connection.createStatement().execute("""CREATE TABLE ABONENTS(
                                                    abonent_id int primary key,
                                                    abonent_name varchar(255))""")
  }

  def addAbonent(abonent: Abonent)(implicit connection: Connection): Unit = {

    val insertSql = "insert into abonents values(?, ?)"

    executeAndClose(connection.prepareStatement(insertSql)) { preparedStatement =>
      preparedStatement.setInt(1, abonent.id)
      preparedStatement.setString(2, abonent.name)
      preparedStatement.execute()
    }
  }

  def getAbonents(implicit connection: Connection): Seq[Abonent] = {
    val abonents = mutable.HashSet[Abonent]()
    executeAndClose(connection.createStatement()) { statement =>
      executeAndClose(statement.executeQuery("select * from abonents")) { resultSet =>
        while(resultSet.next()){
          val id = resultSet.getInt("abonent_id")
          val name = resultSet.getString("abonent_name")
          abonents += Abonent(id, name)
        }
      }
    }
    abonents.toSeq
  }


  def executeAndClose[T <: AutoCloseable, V](resource: T)(f: T => V): V ={
    try{
      f(resource)
    } catch {
      case e: Exception => println(e.getMessage); throw e;
    } finally {
      resource.close()
    }
  }

}

case class Abonent(id: Int, name: String)