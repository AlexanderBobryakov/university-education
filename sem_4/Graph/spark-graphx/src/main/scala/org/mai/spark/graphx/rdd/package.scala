package org.mai.spark.graphx

import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.sql.SparkSession

package object rdd {
  def setLoggerLevel = {
    org.apache.log4j.LogManager.getLogger("com").setLevel(org.apache.log4j.Level.OFF)
    org.apache.log4j.LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF)
    org.apache.log4j.LogManager.getLogger("org.mai").setLevel(org.apache.log4j.Level.INFO)
  }

  def master: String = sys.env.getOrElse("SPARK_MASTER", "local[2]")

  def createSparkSession(appName: String) = {
    setLoggerLevel
    SparkSession.builder().master(master).appName(appName).getOrCreate()
  }

}
