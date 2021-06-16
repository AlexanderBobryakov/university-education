package org.mai.spark.graphx.rdd

import org.apache.spark.sql.SparkSession

object Main extends App{
  val spark = SparkSession
  .builder()
  .master("local[*]")
  .appName("graphTest")
  .getOrCreate()

  import org.apache.spark.graphx.{Edge, Graph, VertexId}
  //val vertices = List("A", "B", "C", "D")
  //  .zip(1L to 4L)
  //  .map(_.swap)
  case class Props(name: String, value: Int)
  val verticesRDD = spark.sparkContext.parallelize(List(
    (1L, Props("A", 10)),
    (2L, Props("B", 20)),
    (3L, Props("C", 30)),
    (4L, Props("D", 40))))
  val edgesRDD = spark.sparkContext.parallelize(List(
    Edge(1L, 2L, 1),
    Edge(2L, 3L, 2),
    Edge(3L, 4L, 3),
    Edge(4L, 1L, 4)))

  val graph: Graph[Props, Int] = Graph(verticesRDD, edgesRDD)
  graph.inDegrees.foreach(println)
}
