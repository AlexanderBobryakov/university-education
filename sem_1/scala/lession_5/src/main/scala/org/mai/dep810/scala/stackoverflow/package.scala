package org.mai.dep810.scala

import scopt.OptionParser

package object stackoverflow {

  val stackOverflowDB: Symbol = 'so

  val stackOverflowBasePath: String = "stackoverflow/"

  val configParser = new OptionParser[Config]("StackOverflowLoader") {
    head("StackOverflowLoader", "1.0")
    cmd("load")
      .action((_, c) => c.copy(commandLoad = "load"))
      .text("Load - это команда загрузки данных из файла")
      .children(
        opt[String]("path")
          .required()
          .action((f, c) => c.copy(path = f))
          .text("Путь к папке с файлами"),
        opt[Unit]("append")
          .abbr("a")
          .action((f, c) => c.copy(append = true))
          .text("Не удалять данные при загрозке. По умолчанию данные будут перезатираться")
      )
    cmd("clean")
      .action((_, c) => c.copy(commandClean = "clean"))
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
    cmd("extract")
      .action((_, c) => c.copy(commandExtract = "extract"))
      .text("Выгрузить данные")
      .children(
        opt[String]("file")
          .required
          .action((f, c) => c.copy(file = f))
          .text("Файл, куда выгрузятся данные"),
        opt[String]("query")
          .abbr("q")
          .required
          .action((q, c) => c.copy(query = q))
          .text("Запрос на выбор данных")
      )
    checkConfig{c=>
      if(c.commandInit.isEmpty &&
        c.commandLoad.isEmpty &&
        c.commandClean.isEmpty &&
        c.commandExtract.isEmpty) failure("Нужно указать хотя бы одну комманду") else success
    }
  }
}
