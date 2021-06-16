package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Graph, VertexId}

trait BreadthFirstSearch {

  //поиск в ширину
  def allPaths[ED](graph: Graph[ED, Double], sourceId: VertexId): Graph[(Double, List[VertexId]), Double] = {
    //инициализация графа
    //в качестве свойств вершины выбраны расстояние от sourceId и список вершин от sourceId
    val initialGrapth = graph.mapVertices((id, _) =>
      if(id == sourceId) (0.0, List(id)) else (Double.PositiveInfinity, List.empty[VertexId])
    )
    val sssp = initialGrapth.pregel((Double.PositiveInfinity, List.empty[VertexId])) (
      (_, dist, newDist) => if(dist._1 < newDist._1) dist else newDist, //изменяем своства вершины
      triplet => {
        //проверка по состоянию вершин триплета
        // на начальную вершину триплета уже присылали сообщения
        // на конечную - нет
        if (triplet.srcAttr._1 != Double.PositiveInfinity && triplet.dstAttr._1 == Double.PositiveInfinity) {
          Iterator((triplet.dstId, (triplet.srcAttr._1 + triplet.attr, triplet.srcAttr._2 :+ triplet.dstId)))
        } else {
          Iterator.empty
        }
      },
      (a, b) => if(a._1 < b._1) a else b //сворачиваем множественные сообщения на вершину
    )

    sssp
  }

  def allNodePath[ED](graph: Graph[ED, Double], sourceId: VertexId) = {

    var g2 = graph.mapVertices( (vid, _) => if (vid == sourceId) 0.0 else Double.PositiveInfinity)

    var stop = false

    while( !stop ) {

      val newDistances = g2.aggregateMessages[Double](
        ctx => {
          if(ctx.srcAttr < Double.PositiveInfinity && ctx.dstAttr == Double.PositiveInfinity) {
            ctx.sendToDst(ctx.srcAttr + ctx.attr)
          }
        },
        (a,b) => math.min(a,b)
      )

      g2 = g2.outerJoinVertices(newDistances)((vid, vd, newSum) =>
        math.min(vd, newSum.getOrElse(Double.PositiveInfinity))
      )

//      stop = g2
//        .vertices
//        .filter(v => v._2 == Double.PositiveInfinity)
//        .isEmpty()

      stop = g2
        .triplets
        .filter(t => t.srcAttr != Double.PositiveInfinity && t.dstAttr == Double.PositiveInfinity)
        .isEmpty()
    }

    graph.outerJoinVertices(g2.vertices)((vid, vd, dist) =>
      (vd, dist.getOrElse(Double.PositiveInfinity)))
  }

  //поиск в ширину с остановкой
  def calculateDistance[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId) = {
    //ининциализируем вершины, получаем Graph[Double]
    var g2 = graph.mapVertices( (vid, _) => if (vid == sourceId) 0.0 else Double.PositiveInfinity)
    var stop = false
    while( !stop ) {
      //при итерации ищем триплет на начале которого уже выставили дистанцию, а на конеце - нет
      val newDistances = g2.aggregateMessages[Double](
        ctx => {
          if(ctx.srcAttr < Double.PositiveInfinity && ctx.dstAttr == Double.PositiveInfinity) {
            ctx.sendToDst(ctx.srcAttr + ctx.attr)
          }
        },
        (a,b) => math.min(a,b)
      )
      g2 = g2.outerJoinVertices(newDistances)((_, vd, newSum) =>
        math.min(vd, newSum.getOrElse(Double.PositiveInfinity))
      )
      //проверяем, достигнут ли targetId или есть ли еще необработанные триплеты
      stop = g2
        .triplets
        .filter(t => (t.dstId == targetId && t.dstAttr == Double.PositiveInfinity) ||
          (t.srcAttr != Double.PositiveInfinity && t.dstAttr == Double.PositiveInfinity))
        .isEmpty()
    }
    graph.outerJoinVertices(g2.vertices)((vid, vd, dist) =>
      (vd, dist.getOrElse(Double.PositiveInfinity)))
  }

  def searchPath[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId) = {
    var g2 = graph.mapVertices( (vid, _) => if (vid == sourceId) (0.0, List(vid)) else (Double.PositiveInfinity, List.empty[VertexId]))

    var stop = false

    while( !stop ) {

      val newDistances = g2.aggregateMessages[(Double, List[VertexId])](
        ctx => {
          if(ctx.srcAttr._1 < Double.PositiveInfinity && ctx.dstAttr._1 == Double.PositiveInfinity) {
            ctx.sendToDst((ctx.srcAttr._1 + ctx.attr, ctx.srcAttr._2 :+ ctx.dstId))
          }
        },
        (a,b) => if(a._1 < b._1) a else b
      )

      g2 = g2.outerJoinVertices(newDistances) { (vid, vd, newSum) =>
        val newDist = newSum.getOrElse((Double.PositiveInfinity, List.empty[VertexId]))
        if(vd._1 < newDist._1) vd else newDist
      }

      stop = g2
        .triplets
        .filter(t => (t.dstId == targetId && t.dstAttr._1 == Double.PositiveInfinity) ||
          (t.srcAttr._1 != Double.PositiveInfinity && t.dstAttr._1 == Double.PositiveInfinity))
        .isEmpty()
    }

    graph.outerJoinVertices(g2.vertices)((vid, vd, dist) =>
      (vd, dist.getOrElse(Double.PositiveInfinity)))
  }

}
