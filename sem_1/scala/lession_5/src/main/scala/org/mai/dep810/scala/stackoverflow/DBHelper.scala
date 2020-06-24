package org.mai.dep810.scala.stackoverflow

import scalikejdbc._

trait DBHelper {

  def db: Symbol

  def createTables = NamedDB(db) autoCommit { implicit session =>

    //create users table
    sql"""
         create table users(
          id Int,
          display_name varchar(1000),
          location varchar(1000),
          reputation Int,
          views Int,
          up_votes Int,
          down_votes Int,
          account_id Int,
          creation_date timestamp,
          last_access_date timestamp
         )
       """.execute().apply()

    sql"""
         create table posts(
          id Int,
          title varchar(999999),
          body varchar(999999),
          score Int,
          viewCount Int,
          answerCount Int,
          commentCount Int,
          ownerUserId Int,
          lastEditorUserId Int,
          acceptedAnswerId Int,
          creationDate timestamp,
          lastEditDate timestamp,
          lastActivityDate timestamp,
          tags varchar(999999)
         )
       """.execute().apply()

    //create commetns table
    sql"""
         create table comments(
          id Int,
          postId Int,
          score Int,
          text varchar(1000),
          creationDate timestamp,
          userid Int
         )
       """.execute().apply()
  }

  def dropTables = NamedDB(db) autoCommit { implicit session =>
    //drop users table
    sql"drop table if exists users".execute().apply()

    //drop posts table
    sql"drop table if exists posts".execute().apply()

    //create commetns table
    sql"drop table if exists comments".execute().apply()
  }

  def clearData = NamedDB(db) autoCommit { implicit session =>

    //delete from users
    sql"delete from users".execute().apply()

    //delete from posts
    sql"delete from posts".execute().apply()

    //delete from comments
    sql"delete from comments".execute().apply()

  }

  def saveData(users: Seq[User], posts: Seq[Post], comments: Seq[Comment]) = NamedDB(db) autoCommit {implicit session =>

    //save users
    val u = User.column
    users.foreach{ user =>
      withSQL(
        insert.into(User).namedValues(
          u.id -> user.id ,
          u.displayName -> user.displayName ,
          u.location -> user.location ,
          u.reputation -> user.reputation ,
          u.views -> user.views ,
          u.upVotes -> user.upVotes ,
          u.downVotes -> user.downVotes ,
          u.accountId -> user.accountId ,
          u.creationDate -> user.creationDate ,
          u.lastAccessDate -> user.lastAccessDate ,
        )
      ).update.apply()
    }

    //save posts
    val p = Post.column
    posts.foreach{ post =>
      withSQL(
        insert.into(Post).namedValues(
          p.id -> post.id,
          p.title -> post.title,
          p.body -> post.body,
          p.score -> post.score,
          p.viewcount -> post.viewcount,
          p.answercount -> post.answercount,
          p.commentcount -> post.commentcount,
          p.owneruserid -> post.owneruserid,
          p.lasteditoruserid -> post.lasteditoruserid,
          p.acceptedanswerid -> post.acceptedanswerid,
          p.creationdate -> post.creationdate,
          p.lasteditdate -> post.lasteditdate,
          p.lastactivitydate -> post.lastactivitydate,
//          p.tags ->
        )
      ).update.apply()
    }


    //save comments
    val c = Comment.column
    comments.foreach{ comment =>
      withSQL(
        insert.into(Comment).namedValues(
          c.id -> comment.id,
          c.postid -> comment.postid,
          c.score -> comment.score,
          c.text -> comment.text,
          c.creationdate -> comment.creationdate,
          c.userid -> comment.userid,
        )
      ).update.apply()
    }
  }

  def extract(query: String): List[String] = NamedDB(db) readOnly { implicit session =>

    session
      .list(query){rs =>

        if(rs.row == 1) {
          List(
            (1 to rs.metaData.getColumnCount).map(i => rs.metaData.getColumnName(i)).mkString(","),
            (1 to rs.metaData.getColumnCount).map(i => if(rs.metaData.getColumnTypeName(i) == "VARCHAR") s""""${rs.string(i)}"""" else rs.string(i)).mkString(",")
          )
        } else {
          List(
            (1 to rs.metaData.getColumnCount).map(i => if(rs.metaData.getColumnTypeName(i) == "VARCHAR") s""""${rs.string(i)}"""" else rs.string(i)).mkString(",")
          )
        }
      }.flatten

  }

}


