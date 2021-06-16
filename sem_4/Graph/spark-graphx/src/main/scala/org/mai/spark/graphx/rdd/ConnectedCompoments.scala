package org.mai.spark.graphx.rdd

import org.apache.spark.graphx.{Graph, VertexId}

trait ConnectedCompoments {
  def run[VD](graph: Graph[VD, Int]) = {
    val initialGraph = graph.mapVertices((vid, _) => vid.toLong)

    val result = initialGraph.pregel(Long.MaxValue) (
      (_, id, newId) => math.min(id, newId),
      triplet => {
        if (triplet.srcAttr != triplet.dstAttr) {
          if (triplet.srcAttr > triplet.dstAttr)
            Iterator((triplet.srcId, triplet.dstAttr.toLong))
          else
            Iterator((triplet.dstId, triplet.srcAttr.toLong))
        } else {
          Iterator.empty
        }
      }
      ,
      (a, b) => math.min(a, b)
    )

    result
  }
}
