package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Graph, VertexId}

trait Dijkstra {

  //рассчет кратчайших расстояний от sourceId по алгоритм Дейкстры
  def allDistances[ED](graph: Graph[ED, Double], sourceId: VertexId) = {
    val initialGrapth = graph.mapVertices((id, ed) =>
      if(id == sourceId) 0.0 else Double.PositiveInfinity
    )

    val sssp = initialGrapth.pregel(Double.PositiveInfinity) (
      (_, dist, newDist) => math.min(dist, newDist),
      triplet => {
        //Distance accumulator
        if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
          Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
        } else {
          Iterator.empty
        }
      },
      (a, b) => math.min(a, b)
    )
    graph.outerJoinVertices(sssp.vertices)((vid, vd, dist) => (vd, dist.getOrElse(Double.PositiveInfinity)))
  }

  //рассчет кратчайших путей от sourceId по алгоритму Дейкстры
  def allPaths[ED](graph: Graph[ED, Double], sourceId: VertexId) = {
    val nullED: Option[ED] = None

    val initialGrapth = graph.mapVertices((id,ed) =>
      if(id == sourceId) (Option(ed), List(Option(ed)), 0.0) else (Option(ed), List.empty[Option[ED]], Double.PositiveInfinity)
    )

    val sssp = initialGrapth.pregel((nullED, List.empty[Option[ED]], Double.PositiveInfinity)) (
      (_, dist, newDist) => if(dist._3 > newDist._3) newDist else dist,
      triplet => {
        //Distance accumulator
        if (triplet.srcAttr._3 + triplet.attr < triplet.dstAttr._3) {

          val newDist = (triplet.dstAttr._1, triplet.srcAttr._2 :+ triplet.dstAttr._1, triplet.srcAttr._3 + triplet.attr)

          Iterator((triplet.dstId, newDist))
        } else {
          Iterator.empty
        }
      },
      (a, b) => if(a._3 < b._3) a else b
    )
      .mapVertices((id, res) => (res._1.getOrElse(null).asInstanceOf[ED], res._2.map(_.getOrElse(null).asInstanceOf[ED]), res._3))

    sssp
  }


  def pairMinDistance[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId) = {

    var g2 = graph.mapVertices( (vid,vd) => (false, if (vid == sourceId) 0.0 else Double.PositiveInfinity))

    var stop = false

    while( !stop ) {

      val newDistances = g2.aggregateMessages[Double](
        ctx => {

          if(!ctx.srcAttr._1 && ctx.srcAttr._2 < Double.PositiveInfinity) {
            ctx.sendToDst(ctx.srcAttr._2 + ctx.attr)
          }
        },
        (a,b) => math.min(a,b))

      g2 = g2.outerJoinVertices(newDistances){(_, vd, newSum) =>

        val changed = if(newSum.isEmpty) {
          if(vd._2 < Double.PositiveInfinity) {
            true
          } else {
            false
          }
        } else {
          if(vd._2 < Double.PositiveInfinity) {
            !(vd._2 < newSum.getOrElse(Double.PositiveInfinity))
          } else {
            false
          }
        }
        val newDist = math.min(vd._2, newSum.getOrElse(Double.PositiveInfinity))

        (changed, newDist)
      }

//      stop = g2
//        .vertices
//        .filter {e =>
//          (e._1 != sourceId && e._2._2 < Double.PositiveInfinity && !e._2._1 ) ||
//            (e._1 == targetId && e._2._2 < Double.PositiveInfinity)
//        }
//        .groupBy(e => e._2._2)
//        .min()(Ordering.by(_._1))._2
//        .exists(ed => ed._1 == targetId)

      stop = g2
        .vertices
        .filter {e =>
          (e._1 != sourceId && e._2._2 < Double.PositiveInfinity && !e._2._1 ) ||
            (e._1 == targetId && e._2._2 < Double.PositiveInfinity)
        }
        .groupBy(e => e._2._2)
        .min()(Ordering.by(_._1))._2
        .exists(ed => ed._1 == targetId)
    }

    graph.outerJoinVertices(g2.vertices)((vid, vd, dist) =>
      (vd, dist.getOrElse((false, Double.PositiveInfinity))._2))
  }


  def pairMinPath[ED](graph: Graph[ED, Double], sourceId: VertexId, targetId: VertexId) = {

    var g2 = graph.mapVertices( (vid,vd) =>
      if(vid == sourceId)
        (false, 0.0, vd, List(vd))
      else
        (false, Double.PositiveInfinity, vd, List.empty[ED])
    )

    var stop = false

    while( !stop ) {

      val newDistances = g2.aggregateMessages[(Double, List[ED])](
        ctx => {
          if(!ctx.srcAttr._1 && ctx.srcAttr._2 < Double.PositiveInfinity) {
//            ctx.sendToDst(ctx.srcAttr._2 + ctx.attr)
            ctx.sendToDst( (ctx.srcAttr._2 + ctx.attr, ctx.srcAttr._4) )
          }
        },
        (a,b) => if(a._1 < b._1) a else b
      )

      g2 = g2.outerJoinVertices(newDistances){(id, vd, newSum) =>

        val nd = newSum.getOrElse((Double.PositiveInfinity), List.empty[ED])

        val changed = if(newSum.isEmpty) {
          if(vd._2 < Double.PositiveInfinity) {
            true
          } else {
            false
          }
        } else {
          if(vd._2 < Double.PositiveInfinity) {
            !(vd._2 < nd._1)
          } else {
            false
          }
        }

        val newVD = if(vd._2 < nd._1) (changed, vd._2, vd._3, vd._4) else (changed, nd._1, vd._3, nd._2 :+ vd._3)

        newVD
      }

      stop = g2
        .vertices
        .filter {e =>
          (e._1 != sourceId && e._2._2 < Double.PositiveInfinity && !e._2._1 ) ||
            (e._1 == targetId && e._2._2 < Double.PositiveInfinity)
        }
        .groupBy(e => e._2._2)
        .min()(Ordering.by(_._1))._2
        .exists(ed => ed._1 == targetId)
    }

    graph.outerJoinVertices(g2.vertices) { (vid, vd, dist) =>
      val distTuple = dist.getOrElse((false, Double.PositiveInfinity, null, List.empty[ED]))
      (vd, distTuple._2, distTuple._4)
    }
  }

}
