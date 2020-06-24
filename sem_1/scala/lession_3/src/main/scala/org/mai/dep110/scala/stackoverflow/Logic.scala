package org.mai.dep110.scala.stackoverflow

import scala.util.matching.Regex

object Logic {

  //obtain all commetns from entities
  def getComments(entities: Seq[Entity]): Seq[Comment] = {
    entities
      .filter(p => {
        p match {
          case _: Comment => true
          case _ => false
        }
      })
      .map(p => p.asInstanceOf[Comment])
  }

  //split entities by type
  def splitEntities(entities: Seq[Entity]): (Seq[User], Seq[Post], Seq[Comment], Seq[Vote], Seq[Badge], Seq[Tag]) = {
    entities
      .foldLeft(Tuple6[Seq[User], Seq[Post], Seq[Comment], Seq[Vote], Seq[Badge], Seq[Tag]]
        (Seq[User](), Seq[Post](), Seq[Comment](), Seq[Vote](), Seq[Badge](), Seq[Tag]())) { (accamulator, entity) =>
        entity match {
          case user: User => accamulator.copy(_1 = accamulator._1 :+ user) //(accamulator._1 :+ entity.asInstanceOf[User] , accamulator._2, accamulator._3, accamulator._4, accamulator._5, accamulator._6)
          case post: Post => (accamulator._1, accamulator._2 :+ post, accamulator._3, accamulator._4, accamulator._5, accamulator._6)
          case _: Comment => (accamulator._1, accamulator._2, accamulator._3 :+ entity.asInstanceOf[Comment], accamulator._4, accamulator._5, accamulator._6)
          case _: Vote => (accamulator._1, accamulator._2, accamulator._3, accamulator._4 :+ entity.asInstanceOf[Vote], accamulator._5, accamulator._6)
          case _: Badge => (accamulator._1, accamulator._2, accamulator._3, accamulator._4, accamulator._5 :+ entity.asInstanceOf[Badge], accamulator._6)
          case _: Tag => (accamulator._1, accamulator._2, accamulator._3, accamulator._4, accamulator._5, accamulator._6 :+ entity.asInstanceOf[Tag])
        }
      }
  }

  //populate fields owner, lastEditor, tags with particular users from Seq[Post] and tags from Seq[Tag]
  def enreachPosts(posts: Seq[Post], users: Seq[User], tags: Seq[Tag]): Seq[EnreachedPost] = {
    posts.map(post => {
      val userOptional = users.find(user => (user.id == post.ownerUserId))
      val editorOptional = users.find(user => post.lastEditorUserId.equals(user.id))
      val tagsSeq = tags.filter(tag => {
        var filter = false
        for (elem <- post.tags) {
          if (elem.contains(tag.tagName)) filter = true
        }
        filter
      })
      EnreachedPost(post, userOptional.orNull, editorOptional.orNull, tagsSeq)
    }).toSeq
  }

  //populate fields post and owner with particular post from Seq[Post] and user from Seq[User]
  def enreachComments(comments: Seq[Comment], posts: Seq[Post], users: Seq[User]): Seq[EnreachedComment] = {
    comments.map(comment => {
      val post = posts.find(p => p.id == comment.postId)
      val owner = users.find(user => user.id == comment.userId)
      EnreachedComment(comment, post.orNull, owner.orNull)
    }).toSeq
  }

  //find all links (like http://example.com/examplePage) in aboutMe field
  def findAllUserLinks(users: Seq[User]): Seq[(User, Seq[String])] = {
    users.map(user => {
      // ищем все линки по патерну
//      val pattern: Regex = "/^(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?$//".r
//      val pattern: Regex = "/(http[s]?):\\/\\/([\\w\\.-]+)([^?$]+)?([^#$]+)?[#]?([^$]+)?/".r
      val pattern: Regex = "(?i)(?:(?:https?|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?!(?:10|127)(?:\\.\\d{1,3}){3})(?!(?:169\\.254|192\\.168)(?:\\.\\d{1,3}){2})(?!172\\.(?:1[6-9]|2\\d|3[0-1])(?:\\.\\d{1,3}){2})(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))\\.?)(?::\\d{2,5})?(?:[/?#]\\S*)?".r
      val matches = pattern.findAllMatchIn(user.about)
      val seq = matches.map(m => m.toString()).toSeq
      (user, seq)
    }).toSeq
  }

  //find all users with the reputation bigger then reputationLimit with particular badge
  def findTopUsersByBadge(users: Seq[User], basges: Seq[Badge], badgeName: String, reputationLimit: Int): Seq[User] = {
    // отбираем id пользователей с badge по имени
    val userIds = basges.filter(b => b.name.equals(badgeName))
      .foldLeft(Seq[Int](0)) { (ac, badge) =>
        ac :+ badge.userId
      }
    val usersnew = users.filter(user => {
      user.reputation > reputationLimit
    })
      .filter(user => {
        userIds.contains(user.id)
      })
    usersnew
  }

}

case class EnreachedPost(
                          post: Post,
                          owner: User,
                          lastEditor: User,
                          tags: Seq[Tag]
                        )

case class EnreachedComment(
                             comment: Comment,
                             post: Post,
                             owner: User
                           )
