package org.mai.dep810.scala.jdbc

import java.time.LocalDateTime

import org.log4s._
import scalikejdbc._
import scalikejdbc.config._
import scopt.OptionParser

object Main {

  private[this] val logger = getLogger

  val db = 'jdbcTest

  def main(args: Array[String]): Unit = {

//    logger.info("Manually setup jdnc session")
//    ConnectionPool.singleton("jdbc:h2:file:./jdbcTestDB", "user", "pass")
//    implicit val session = AutoSession
//
//    logger.info("Dropping table")
//    sql"drop table if exists users".execute.apply()(session)
//
//    logger.info("Creating table")
//    sql"""
//          create table users (
//            id          int,
//            name        varchar(100),
//            created_time timestamp
//         )""".execute.apply()(session)
//
//    logger.info("Inserting rows")
//    sql"insert into users values(1, 'Ivanov', ${LocalDateTime.now})".update.apply()
//
//    val user = User(2, "Petrov", LocalDateTime.now)
//    sql"insert into users values(${user.id}, ${user.name}, ${user.createdTime})".update.apply()
//
//    logger.info("Selecting rows")
//    val users = sql"""select * from users"""
//      .map(rs => User(rs.int("id"), rs.string("name"), rs.localDateTime("created_time")))
//      .list()
//      .apply()
//
//    users foreach println
//
//    session.close()
//
//    logger.info("Session closed")




    logger.info("Setup jdnc session")
    DBs.setup(db)
    //DBs.setupAll()

    logger.info("Drop tables")
    dropTables

    logger.info("Create tables")
    createTables

    val users = Seq(
      User(1, "Petrov", LocalDateTime.now()),
      User(2, "Sidorov", LocalDateTime.now()),
      User(3, "ivanov", LocalDateTime.now()),
      User(4, "Fedorov", LocalDateTime.now()),
      User(5, "Titov", LocalDateTime.now()),
      User(6, "Ptitsyn", LocalDateTime.now()),
      User(7, "Korolev", LocalDateTime.now()),
      User(8, "Tsvetkov", LocalDateTime.now()),
      User(9, "Suslov", LocalDateTime.now())
    )
    logger.info("Inserting users")
    //insertUsers(users)
    insertUsersSQLSyntax(users)

    val u = User.syntax("u")
    NamedDB(db).readOnly { implicit session =>
      withSQL(
        select.from(User as u).where.in(u.id, Seq(2,4,6,8)).orderBy(u.name).limit(3)
      )
        .map( User(u.resultName))
        .list()
        .apply()
    }.foreach(println)


    DBs.closeAll()
  }

  def createTables: Unit = NamedDB(db).autoCommit{ implicit session =>
    sql"""
         create table users (
            id          int,
            name        varchar(100),
            created_time timestamp
         )
       """.execute.apply()
  }

  def dropTables: Unit = NamedDB(db).autoCommit { implicit session =>
    sql"drop table if exists users".execute.apply()
  }

  def insertUsers(users: Seq[User]) = NamedDB(db).autoCommit{ implicit session =>
    users.foreach{ user =>
      sql"insert into users(id, name) values(${user.id}, ${user.name})".update.apply()
    }
  }

  def insertUsersSQLSyntax(users: Seq[User]) = NamedDB(db).autoCommit { implicit session =>

    val u = User.column
    users.foreach { user =>
      withSQL(
        insert.into(User).namedValues(
          u.id -> user.id,
          u.name -> user.name,
          u.createdTime -> user.createdTime)
      ).update().apply()
    }
  }


}

case class User(id: Int, name: String, createdTime: LocalDateTime)

object User extends SQLSyntaxSupport[User] {

  override val tableName: String = "users"
  override def connectionPoolName: Any = 'jdbcTest

  def apply(u: ResultName[User])(rs: WrappedResultSet): User = {
    new User(rs.int(u.id), rs.string(u.name), rs.localDateTime(u.createdTime))
  }
}


case class Config(
                    clear: Boolean = false,
                    dropTables: Boolean = false,
                    createTables: Boolean = false
                 )