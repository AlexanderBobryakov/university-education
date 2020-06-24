package org.mai.dep810.scala.stackoverflow

import java.time.LocalDateTime

import scalikejdbc._

abstract class Entity(id: Int)

case class Post(
                 id: Int,
                 title: String,
                 body: String,
                 score: Int,
                 viewcount: Int,
                 answercount: Int,
                 commentcount: Int,
                 owneruserid: Int,
                 lasteditoruserid: Int,
                 acceptedanswerid: Int,
                 creationdate: LocalDateTime,
                 lasteditdate: LocalDateTime,
                 lastactivitydate: LocalDateTime,
                 tags: Seq[String]) extends Entity(id)

case class Comment(
                    id: Int,
                    postid: Int,
                    score: Int,
                    text: String,
                    creationdate: LocalDateTime,
                    userid: Int) extends Entity(id)


case class User(
                 id: Int,
                 displayName: String,
                 location: String,
                 reputation: Int,
                 views: Int,
                 upVotes: Int,
                 downVotes: Int,
                 accountId: Int,
                 creationDate: LocalDateTime,
                 lastAccessDate: LocalDateTime) extends Entity(id)

object User extends SQLSyntaxSupport[User] {
  override def connectionPoolName: Any = stackOverflowDB

  override def tableName: String = "users"

  def apply(u: ResultName[User])(rs: WrappedResultSet): Unit = {
    new User(
      rs.int(u.id),
      rs.string(u.displayName),
      rs.string(u.location),
      rs.int(u.reputation),
      rs.int(u.views),
      rs.int(u.upVotes),
      rs.int(u.downVotes),
      rs.int(u.accountId),
      rs.localDateTime(u.creationDate),
      rs.localDateTime(u.lastAccessDate)
    )
  }
}

object Post extends SQLSyntaxSupport[Post] {
  override def connectionPoolName: Any = stackOverflowDB

  override def tableName: String = "posts"
  //  override val columns = Seq(
  //    "id",
  //    "title",
  //    "body",
  //    "score",
  //    "view_count",
  //    "answer_count",
  //    "comment_count",
  //    "owner_user_id",
  //    "last_editor_user_id",
  //    "accepted_answer_id",
  //    "creation_date",
  //    "last_edit_date",
  //    "last_activity_date",
  //    "tags"
  //  )
//    override val nameConverters = Map(
//      "id" -> "id",
//      "title" -> "title",
//      "body" -> "body",
//      "score" -> "score",
//      "view_count" -> "viewCount",
//      "answer_count" -> "answerCount",
//      "comment_count" -> "commentCount",
//      "owner_user_id" -> "ownerUserId",
//      "last_editor_user_id" -> "lastEditorUserId",
//      "accepted_answer_id" -> "acceptedAnswerId",
//      "creation_date" -> "creationDate",
//      "last_edit_date" -> "lastEditDate",
//      "last_activity_date" -> "lastActivityDate",
//      "tags" -> "tags"
//    )

//  def apply(u: ResultName[Post])(rs: WrappedResultSet): Unit = {
//    new Post(
//      rs.int(u.id),
//      rs.string(u.title),
//      rs.string(u.body),
//      rs.int(u.score),
//      rs.int(u.viewCount),
//      rs.int(u.answerCount),
//      rs.int(u.commentCount),
//      rs.int(u.ownerUserId),
//      rs.int(u.lastEditorUserId),
//      rs.int(u.acceptedAnswerId),
//      rs.localDateTime(u.creationDate),
//      rs.localDateTime(u.lastEditDate),
//      rs.localDateTime(u.lastActivityDate),
//      null
//    )
//  }
}

object Comment extends SQLSyntaxSupport[Comment] {
  override def connectionPoolName: Any = stackOverflowDB

  override def tableName: String = "comments"
}

case class Config (
                    commandLoad: String = "",
                    commandClean: String = "",
                    commandInit: String = "",
                    commandExtract: String = "",
                    path: String = "",
                    file: String = "",
                    append: Boolean = false,
                    dropTables: Boolean = false,
                    forse: Boolean = false,
                    query: String = ""
                  )