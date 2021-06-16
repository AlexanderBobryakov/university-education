package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Graph, VertexId}

trait YenSKShortestPaths {

  def getShortestPaths[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId, k: Int) = {
    val a1 = dijkstraShortestPath[ED](graph, sourceId, targetId)

    var stop = false
    var container = List(a1)

    while(!stop && container.size < k) {
      val kPath = getKPath[ED](graph, sourceId, targetId, container)

      container = container ++ kPath

      stop = kPath.isEmpty
    }

    container
  }

  def getKPath[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId, container: List[ShortPath[ED]]) = {
    val spurs = getSpurNodesAndEdges(container)

    val rk = spurs.filter(_._2.nonEmpty).flatMap {group =>
      val spurId = group._1._1
      val rkGraph = graph.mapEdges { e =>
        if(group._2.exists{ t => t._2.last._1 == e.dstId && t._2.head._1 == e.srcId}) {
          Double.PositiveInfinity
        } else {
          e.attr
        }
      }
      val ak = dijkstraShortestPath[ED](rkGraph, spurId, targetId)
      group._2.map {p =>
        ShortPath[ED](sourceId, ak.targetId, ak.distance + p._1.map(_._3).sum, p._1 ++ ak.path.tail)
      }
    }.filter(_.distance < Double.PositiveInfinity)

    if(rk.nonEmpty) {
      List(rk.minBy(_.distance))
    } else {
      List.empty[ShortPath[ED]]
    }

  }

  def getSpurNodesAndEdges[ED](container: List[ShortPath[ED]]) = {
    val rootSpur = container.flatMap { sp =>
      (1 until sp.path.size).map(it => (sp.path.take(it), sp.path.drop(it-1).take(2)))
    }.groupBy(_._1.last)

    rootSpur
  }


  def dijkstraShortestPath[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId) = {

    var g2 = graph.mapVertices( (vid,vd) =>
      if(vid == sourceId)
        Message[ED](false, 0.0, 0.0, vd, List((vid, vd, 0.0)))
      else
        Message[ED](false, Double.PositiveInfinity, Double.PositiveInfinity, vd, List.empty[(VertexId, ED, Double)])
    )

    var stop = false

    while( !stop ) {

      val newDistances = g2.aggregateMessages[Message[ED]](
        ctx => {
          if(!ctx.srcAttr.changed && ctx.srcAttr.distance < Double.PositiveInfinity) {
            val newMessage = Message[ED](ctx.srcAttr.changed, ctx.srcAttr.distance + ctx.attr, ctx.attr, ctx.srcAttr.vertexAttr, ctx.srcAttr.path) //ctx.srcAttr.copy(distance = ctx.srcAttr.distance + ctx.attr, lastShoulder = ctx.attr)
            ctx.sendToDst( newMessage )
          }
        },
        (a,b) => if(a.distance < b.distance) a else b
      )

      g2 = g2.outerJoinVertices(newDistances){(id, vd, newSum) =>

        val nd = newSum.getOrElse(Message[ED](false, Double.PositiveInfinity, Double.PositiveInfinity, vd.vertexAttr, List.empty[(VertexId, ED, Double)]))

        val changed = if(newSum.isEmpty) {
          if(vd.distance < Double.PositiveInfinity) {
            true
          } else {
            false
          }
        } else {
          if(vd.distance < Double.PositiveInfinity) {
            !(vd.distance < nd.distance)
          } else {
            false
          }
        }

        val newVD = if(vd.distance < nd.distance) {
          Message[ED](changed, vd.distance, vd.lastShoulder, vd.vertexAttr, vd.path)
        } else {
          Message[ED](changed, nd.distance, nd.lastShoulder, vd.vertexAttr, nd.path :+ (id, vd.vertexAttr, nd.lastShoulder))
        }

        newVD
      }

      val mins = g2
        .vertices
        .filter {e =>
          (e._1 != sourceId && e._2.distance < Double.PositiveInfinity && !e._2.changed ) ||
            (e._1 == targetId && e._2.distance < Double.PositiveInfinity)
        }
        .groupBy(e => e._2.distance)

      stop = if(mins.isEmpty()) {
        true
      } else {
        mins
          .min()(Ordering.by(_._1))
          ._2
          .exists(ed => ed._1 == targetId)
      }
    }

    graph.outerJoinVertices(g2.vertices) { (vid, vd, dist) =>
      val distMessage = dist.getOrElse(Message[ED](false, Double.PositiveInfinity, Double.PositiveInfinity, vd, List.empty[(VertexId, ED, Double)]))
      ShortPath(sourceId, vid, distMessage.distance, distMessage.path)
    }.vertices.filter(_._1 == targetId).first()._2
  }

}

case class Message[ED](
                  changed: Boolean,
                  distance: Double,
                  lastShoulder: Double,
                  vertexAttr: ED,
                  path: List[(VertexId, ED, Double)]
                  )

case class ShortPath[ED](
                    sourceId: VertexId,
                    targetId: VertexId,
                    distance: Double,
                    path: List[(VertexId, ED, Double)]
                    )
