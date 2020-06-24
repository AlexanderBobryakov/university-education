package org.mai.dep810.scala.stackoverflow

import java.time.LocalDateTime

import scalikejdbc._

import scala.xml.XML

trait DataLoader {
  def basePath: String

  private[this] val filesToFunc = Map[String, String => Seq[Entity]](
    "Users.xml" -> loadUsers,
    "Posts.xml" -> loadPosts,
    "Comments.xml" -> loadComments
  )

  def loadEntities(): Seq[Entity] = {
    filesToFunc
      .flatMap{
        case (file, func) =>
          val path = basePath+"/"+file
          func(path)
      }
      .toSeq
  }

  def loadData(): (Seq[User], Seq[Post], Seq[Comment]) = {
    splitEntities(loadEntities())
  }

  //split entities by type
  private[this] def splitEntities(entities: Seq[Entity]): (Seq[User], Seq[Post], Seq[Comment]) = {
    entities.foldLeft((Seq[User](), Seq[Post](), Seq[Comment]())) { case((users, posts, comments), entity) =>
      entity match {
        case u: User => (u+:users, posts, comments)
        case p: Post => (users, p+:posts, comments)
        case c: Comment => (users, posts, c+:comments)
      }
    }
  }

  private[this] def loadUsers(path: String): Seq[User] = {
    val usersXml = XML.load(path)

    for(
      userRow <- usersXml \\ "row"
    ) yield {
      User(
        (userRow \@ "Id").toInt,
        userRow \@ "DisplayName",
        userRow \@ "location",
        matchInt(userRow \@ "Reputation"),
        matchInt(userRow \@ "Views"),
        matchInt(userRow \@ "UpVotes"),
        matchInt(userRow \@ "DownVotes"),
        matchInt(userRow \@ "AccountId"),
        parseDate(userRow \@ "CreationDate"),
        parseDate(userRow \@ "LastAccessDate")
      )
    }
  }

  private[this] def loadPosts(path: String): Seq[Post] = {
    val postsXml = XML.load(path)

    for(
      postRow <- postsXml \\ "row"
    ) yield {
      Post(
        matchInt(postRow \@ "Id"),
        postRow \@ "Title",
        postRow \@ "Body",
        matchInt(postRow \@ "Score"),
        matchInt(postRow \@ "ViewCount"),
        matchInt(postRow \@ "AnswerCount"),
        matchInt(postRow \@ "CommentCount"),
        matchInt(postRow \@ "OwnerUserId"),
        matchInt(postRow \@ "LastEditorUserId"),
        matchInt(postRow \@ "AcceptedAnswerId"),
        parseDate(postRow \@ "CreationDate"),
        parseDate(postRow \@ "LastEditDate"),
        parseDate(postRow \@ "LastActivityDate"),
        (postRow \@ "Tags").stripPrefix("&lt;").stripSuffix("&gt;").split("&gt;&lt;").toSeq
      )
    }
  }

  private[this] def loadComments(path: String): Seq[Comment] = {
    val commentsXml = XML.load(path)

    for(
      row <- commentsXml \\ "row"
    ) yield {
      Comment(
        matchInt(row \@ "Id"),
        matchInt(row \@ "PostId"),
        matchInt(row \@ "Score"),
        row \@ "Text",
        parseDate(row \@ "CreationDate"),
        matchInt(row \@ "UserId")
      )
    }
  }

  private[this] def matchInt(s: String): Int = {
    val intMatch = "(\\d+)".r
    s match {
      case intMatch(i) => i.toInt
      case _ => Int.MinValue
    }
  }

  private[this] def parseDate(s: String): LocalDateTime = {
    if(s == "")
    null
    else
    LocalDateTime.parse(s)
  }
}



