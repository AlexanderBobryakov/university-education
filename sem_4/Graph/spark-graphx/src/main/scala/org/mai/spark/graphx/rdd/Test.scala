package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Edge, EdgeDirection, Graph, VertexId, VertexRDD}
import org.apache.spark.graphx.Graph
import org.apache.spark.sql.SparkSession
import org.log4s.getLogger

object Test {

  val appName: String = "Test"

  private[this] val logger = getLogger

  def main(args: Array[String]): Unit = {
    logger.info(s"Start spark job $appName")

    val spark = createSparkSession(appName)

//    testBFS(spark)
//    testDijkstra(spark)
    testYensKShortestPaths(spark)
//    testConnectedComponents(spark)

    Thread.sleep(500000)
    logger.info(s"Finish spark job $appName")
  }

  def testBFS(spark: SparkSession) = {
    val graph = generateGraph2(spark)
      .mapEdges(_.attr.toDouble)

    val algo = new BreadthFirstSearch {}

//    val sssp = algo.allPaths(graph, 1L)
    val sssp = algo.allNodePath(graph, 1L)
//    val sssp = algo.searchPath(graph, 1L, 4L)

    sssp.vertices.foreach(println)
  }

  def testDijkstra(spark: SparkSession) = {
    val graph = generateGraph2(spark)
      .mapEdges(_.attr.toDouble)

    val algo = new Dijkstra {}

//    val sssp = algo.allDistances(graph, 1L)
//    val sssp = algo.allPaths(graph, 1L)
//    val sssp = algo.pairMinDistance(graph, 1L, 2L)
    val sssp = algo.pairMinPath(graph, 1L, 2L)

    sssp.vertices.foreach(println)
  }

  def testYensKShortestPaths(spark: SparkSession) = {
    val graph = generateGraph4(spark)
      .mapEdges(_.attr.toDouble)

    val algo = new YenSKShortestPaths {}

    val sssp = algo.getShortestPaths(graph, 1L, 8L, 6)

    sssp.foreach(println)
  }

  def testConnectedComponents(spark: SparkSession) = {
    val algo = new ConnectedCompoments {}

    val graph = getBiDirectionalMultyComponentGraph(spark)
    

    val res = algo.run[String](graph)

    res.vertices.foreach(println)
  }

  def generateGraph(spark: SparkSession): Graph[String, Double] = {
    val vertices = spark.sparkContext.parallelize(List((1l, "A"), (2L, "B"), (3L, "C"), (4L, "D")))
    val edges = spark.sparkContext.parallelize(List(
      Edge(1L, 2L, 5.0),
      Edge(1L, 3L, 1.0),
      Edge(1L, 4L, 6.0),
      Edge(2L, 4L, 2.0),
      Edge(3L, 2L, 2.0),
      Edge(3L, 4L, 3.0)
    ))

    val graph = Graph(vertices, edges, "Dummy")

    graph
  }

  def generateGraph2(spark: SparkSession) = {
    val vertices = List("A", "B", "C", "D", "F", "G", "H", "J", "I", "K", "L").zip(1L to 11L).map(_.swap)
    val verticesRDD = spark.sparkContext.parallelize(vertices)

    val edgesRDD = spark.sparkContext.parallelize(List(
      Edge(1L, 2L, 10),
      Edge(2L, 3L, 2),
      Edge(3L, 4L, 3),
      Edge(1L, 5L, 1),
      Edge(5L, 6L, 1),
      Edge(6L, 7L, 1),
      Edge(7L, 8L, 1),
      Edge(8L, 2L, 1),
      Edge(1L, 9L, 1),
      Edge(9L, 10L, 1),
      Edge(10L, 11L, 1),
      Edge(11L, 2L, 1)
    ))

    Graph(verticesRDD, edgesRDD)
  }

  def generateGraph3(spark: SparkSession) = {
    val vertices = List("C", "D", "F", "E", "G", "H").zip(1L to 6L).map(_.swap)
    val verticesRDD = spark.sparkContext.parallelize(vertices)

    val edgesRDD = spark.sparkContext.parallelize(List(
      Edge(1L, 4L, 2),
      Edge(1L, 2L, 3),
      Edge(2L, 3L, 4),
      Edge(4L, 2L, 1),
      Edge(4L, 3L, 2),
      Edge(4L, 5L, 3),
      Edge(5L, 3L, 2),
      Edge(5L, 6L, 2),
      Edge(3L, 6L, 1)
    ))

    Graph(verticesRDD, edgesRDD)
  }

  def generateGraph4(spark: SparkSession) = {
    val vertices = List("A", "B", "C", "D", "E", "F", "G", "H").zip(1L to 8L).map(_.swap)
    val verticesRDD = spark.sparkContext.parallelize(vertices)

    val edgesRDD = spark.sparkContext.parallelize(List(
      Edge(1L, 2L, 1),
      Edge(2L, 3L, 1),
      Edge(3L, 8L, 1),
      Edge(1L, 4L, 2),
      Edge(4L, 5L, 2),
      Edge(5L, 8L, 2),
      Edge(1L, 6L, 3),
      Edge(6L, 7L, 3),
      Edge(7L, 8L, 3),
      Edge(2L, 5L, 1),
      Edge(6L, 5L, 3)
    ))

    Graph(verticesRDD, edgesRDD)
  }

  def getBiDirectionalMultyComponentGraph(spark: SparkSession) = {
    val vertices = spark.sparkContext.parallelize((1l to 10l).map(it => (it, s"$it")))

    val edgesList = List(
      Edge(1L, 2L, 1),
      Edge(2L, 3L, 1),
      Edge(3L, 1L, 1),
      Edge(4L, 5L, 1),
      Edge(5L, 6L, 1),
      Edge(7L, 8L, 1),
      Edge(8L, 9L, 1),
      Edge(9L, 10L, 1),
      Edge(7L, 9L, 1),
      Edge(7L, 10L, 1),
      Edge(9L, 10L, 1),
      Edge(8L, 10L, 1)
    )

    val edges = spark.sparkContext.parallelize(
      edgesList ++ edgesList.map(e => Edge(e.dstId, e.srcId, e.attr))
    )

    Graph(vertices, edges)
  }

}
