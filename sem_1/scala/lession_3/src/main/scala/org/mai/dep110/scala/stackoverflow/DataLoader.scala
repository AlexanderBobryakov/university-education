package org.mai.dep110.scala.stackoverflow

import java.time.LocalDateTime

import scala.xml.XML

trait DataLoader {
  def basePath: String

  private[this] val filesToFunc = Map[String, String => Seq[Entity]](
    "Users.xml" -> loadUsers,
    "Posts.xml" -> loadPosts,
    "Comments.xml" -> loadComments,
    "Votes.xml" -> loadVotes,
    "Badges.xml" -> loadBadges,
    "Tags.xml" -> loadTags
  )

  def loadData(): Seq[Entity] = {
    filesToFunc
      .flatMap{
        case (file, func) =>
          val path = basePath+"/"+file
          func(path)
      }
      .toSeq
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
        userRow \@ "AboutMe",
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

  private[this] def loadVotes(path: String): Seq[Vote] = {
    val votesXml = XML.load(path)

    for(
      row <- votesXml \\ "row"
    ) yield {
      Vote(
          matchInt(row \@ "Id"),
          matchInt(row \@ "PostId"),
          matchInt(row \@ "VoteTypeId"),
          parseDate(row \@ "CreationDate"))
    }
  }

  private[this] def loadBadges(path: String): Seq[Badge] = {
    val badgesXml = XML.load(path)

    for(
      row <- badgesXml \\ "row"
    ) yield {
      Badge(
        matchInt(row \@ "Id"),
        matchInt(row \@ "UserId"),
        row \@ "Name",
        parseDate(row \@ "Date"),
        matchInt(row \@ "Class"))
    }
  }

  private[this] def loadTags(path: String): Seq[Tag] = {
    val tagsXml = XML.load(path)

    for(
      row <- tagsXml \\ "row"
    ) yield {
      Tag(
          matchInt(row \@ "Id"),
          row \@ "TagName",
          matchInt(row \@ "Count"),
          matchInt(row \@ "ExcerptPostId"),
          matchInt(row \@ "WikiPostId"))
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

abstract class Entity(id: Int)

case class User(
                 id: Int,
                 displayName: String,
                 location: String,
                 about: String,
                 reputation: Int,
                 views: Int,
                 upVotes: Int,
                 downVotes: Int,
                 accountId: Int,
                 creationDate: LocalDateTime,
                 lastAccessDate: LocalDateTime) extends Entity(id)

case class Post(
               id: Int,
               title: String,
               body: String,
               score: Int,
               viewCount: Int,
               answerCount: Int,
               commentCount: Int,
               ownerUserId: Int,
               lastEditorUserId: Int,
               acceptedAnswerId: Int,
               creationDate: LocalDateTime,
               lastEditDate: LocalDateTime,
               lastActivityDate: LocalDateTime,
               tags: Seq[String]) extends Entity(id)

case class Comment(
                    id: Int,
                    postId: Int,
                    score: Int,
                    text: String,
                    creationDate: LocalDateTime,
                    userId: Int) extends Entity(id)

case class Vote(
                 id: Int,
                 postId: Int,
                 voteTypeId: Int,
                 creationDate: LocalDateTime) extends Entity(id)

case class Badge(
                  id: Int,
                  userId: Int,
                  name: String,
                  date: LocalDateTime,
                  badgeClass: Int
                ) extends Entity(id)

case class Tag(
                id: Int,
                tagName: String,
                count: Int,
                excerptPostId: Int,
                wikiPostId: Int
              ) extends Entity(id)