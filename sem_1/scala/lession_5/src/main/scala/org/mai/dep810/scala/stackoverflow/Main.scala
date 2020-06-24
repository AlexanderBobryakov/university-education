package org.mai.dep810.scala.stackoverflow

import java.io.{BufferedWriter, File, FileWriter}

import scalikejdbc.config.DBs
import org.log4s._

object Main extends App {

  private[this] val logger = getLogger
  val soDB = stackOverflowDB

  configParser.parse(args,Config()) match {
    case Some(config) =>

      DBs.setup(soDB)

      val dbHelper = new DBHelper {
        override def db: Symbol = soDB
      }
      val dataLoader = new DataLoader {
        override def basePath: String = stackOverflowBasePath
      }


      if(!config.commandClean.isEmpty) {
        if(config.dropTables) {
          //dropTables
          dbHelper.dropTables
        } else {
          //clearData
          dbHelper.clearData
        }
      }

      if(!config.commandInit.isEmpty) {
        if(config.forse) {
          //dropTables
          dbHelper.dropTables
        }
        //createTables
        dbHelper.createTables
      }

      if(!config.commandLoad.isEmpty) {
        if(!config.append) {
          //clearData
          dbHelper.clearData
        }

        //load data

        val (users, posts, comments) = dataLoader.loadData()
        dbHelper.saveData(users, posts, comments)

      }

      if(!config.commandExtract.isEmpty) {

        //extract data
        val lines = dbHelper.extract(config.query)

        val bf = new BufferedWriter(new FileWriter(new File(config.file)))
        lines.foreach{line =>
          bf.write(line)
          bf.newLine()
        }
        bf.flush()
        bf.close()
      }

      DBs.closeAll()

    case None =>
  }


}
