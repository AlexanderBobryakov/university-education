package org.mai.dep810.scala.iris

import scalikejdbc._
import scalikejdbc.config._
import scopt.OptionParser

object Main extends App {

  private[this] val logger = getClass
  val irisDB = 'iris

  val parser = new OptionParser[Config]("IrisLoader") {
    head("IrisLoader", "1.0")
    cmd("load")
      .action((_, c) => c.copy(commandLoad = "load"))
      .text("Load - это команда загрузки данных из файла")
      .children(
        opt[String]("file")
          .required()
          .action((f, c) => c.copy(file = f))
          .text("Путь к файлу с данными"),
        opt[Unit]("append")
          .abbr("a")
          .action((f, c) => c.copy(append = true))
          .text("Не удалять данные при загрозке. По умолчанию данные будут перезатираться")
      )
    cmd("clean")
      .action((_, c) => c.copy(commandClear = "clean"))
      .text("Удалить данные из базы данных")
      .children(
        opt[Unit]("dropTables")
          .abbr("dt")
          .action((_, c) => c.copy(dropTables = true))
          .text("Удалить таблицы")
      )
    cmd("init")
      .action((_, c) => c.copy(commandInit = "init"))
      .text("Создать таблицы")
      .children(
        opt[Unit]("forse")
          .abbr("f")
          .action((_, c) => c.copy(forse = true))
          .text("Пересоздать таблицы, если существуют")
      )
    checkConfig{c=>
      if(c.commandInit.isEmpty && c.commandLoad.isEmpty && c.commandClear.isEmpty) failure("Нужно указать хотя бы одну комманду") else success
    }
  }

  parser.parse(args,Config()) match {
    case Some(config) =>

      DBs.setup(irisDB)

      if(!config.commandClear.isEmpty) {
        if(config.dropTables) {
          dropTables
        } else {
          clearData
        }
      }

      if(!config.commandInit.isEmpty) {
        if(config.forse) {
          dropTables
        }
        createTables
      }

      if(!config.commandLoad.isEmpty) {
        if(!config.append) {
          clearData
        }
        loadIrisToDB(IrisParser.loadFromFile(config.file))
      }

      DBs.closeAll()

    case None =>
  }




  def createTables = NamedDB(irisDB).autoCommit { implicit session =>
    sql"""
         create table if not exists iris (
            sepal_length Double,
            sepal_width Double,
            petal_length Double,
            petal_width Double,
            species Varchar(100)
         )
       """.execute.apply()
  }

  def dropTables = NamedDB(irisDB).autoCommit { implicit session =>
    val query = "drop table if exists iris"
    session.execute(query)
    //sql"drop table if exists iris".execute.apply()
  }

  def clearData = NamedDB(irisDB).autoCommit { implicit session =>
    sql"delete from iris".update.apply()
  }

  def checkIfTablesExist: Boolean = NamedDB(irisDB).autoCommit { implicit session =>



    sql"select count(*) as cnt from information_schema.tables where table_name = 'IRIS'".map(rs => rs.int("cnt"))
      .single
      .apply() match {
        case Some(c) => c == 1
        case None => false
      }
  }


  def loadIrisToDB(data: List[Iris]) = NamedDB(irisDB).autoCommit { implicit session =>
    val i = Iris.column
    data.foreach { iris =>
      withSQL(
        insert.into(Iris).namedValues(
          i.sepalLength -> iris.sepalLength,
          i.sepalWidth -> iris.sepalWidth,
          i.petalLength -> iris.petalLength,
          i.petalWidth -> iris.petalWidth,
          i.species -> iris.species)
      ).update().apply()
    }
  }

}

case class Iris(
                 sepalLength: Double,
                 sepalWidth: Double,
                 petalLength: Double,
                 petalWidth: Double,
                 species: String
               )

object Iris extends SQLSyntaxSupport[Iris] {
  override val tableName: String = "iris"
  override def connectionPoolName: Any = 'iris

  def apply(i: ResultName[Iris])(rs: WrappedResultSet): Unit = {
    new Iris(
      rs.double(i.sepalLength),
      rs.double(i.sepalWidth),
      rs.double(i.petalLength),
      rs.double(i.petalWidth),
      rs.string(i.species))
  }
}

case class Config (
                    commandLoad: String = "",
                    commandClear: String = "",
                    commandInit: String = "",
                    file: String = "",
                    append: Boolean = false,
                    dropTables: Boolean = false,
                    forse: Boolean = false
                  )