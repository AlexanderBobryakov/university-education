package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType}
import org.log4s.getLogger

object Transport {
  val appName: String = "Transport"

  private[this] val logger = getLogger

  def main(args: Array[String]): Unit = {
    logger.info(s"Start spark job $appName")

    val spark = createSparkSession(appName)
//    spark.conf.set("spark.testing.memory", "2147480000")

//    testBFS(spark)
    testDijkstra(spark)

    Thread.sleep(100000)
    logger.info(s"Finish spark job $appName")
  }

  def testDijkstra(spark: SparkSession) = {
    val algo = new Dijkstra {}

    val transport = prepareGraph(spark).mapEdges(_.attr._2.toDouble)

    //val sssp = algo.allPaths(transport, 1L)
    //    val sssp = algo.pairMinDistance(transport, 1L, 8L)
    val sssp = algo.pairMinPath(transport, 1L, 4L)

    sssp.vertices.foreach(println)
  }

  def testBFS(spark: SparkSession) = {
    val algo = new BreadthFirstSearch {}

    val transport = prepareGraph(spark).mapEdges(_.attr._2.toDouble)
    transport.pageRank(0.15)

//    val sssp = algo.allPaths(transport, 1L)
//    val sssp = algo.allNodePath(transport, 1L)
    val sssp = algo.searchPath(transport, 1L, 5L)

    sssp.vertices.foreach(println)
  }

  def prepareGraph(spark: SparkSession) = {
    import spark.implicits._

    val vertices = spark
      .read
      .option("header", value = true)
      .csv("transport-nodes.csv")
      .withColumn("rowId", row_number.over(Window.orderBy("id")).cast(LongType))
      .as("v")

    val edgesRDD = spark
      .read
      .option("header", value = true)
      .csv("transport-relationships.csv")
      .as("e")
      .join(vertices, $"v.id" === $"e.src")
      .select('src, $"v.rowId".as("srcId"), 'dst, 'relationship, 'cost)
      .as("d")
      .join(vertices, $"v.id" === $"d.dst")
      .select('srcId, $"v.rowId".as("dstId"), 'relationship, 'cost.cast(LongType))
      .as[(Long, Long, String, Long)]
      .rdd
      .map { case (srcId, dstId, relationship, cost) => Edge(srcId, dstId, (relationship, cost))}

    val verticesRdd = vertices
      .as[(String, String, String, String, Long)]
      .rdd
      .map{ case (cityName, latitude, longitude, population, id) => (id, (cityName, latitude.toDouble, longitude.toDouble, population.toInt))}

    Graph(verticesRdd, edgesRDD)
  }

}
