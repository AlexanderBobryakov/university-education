package ru.mai.dep806.bigdata.spark;

import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Serializable;
import scala.Tuple2;

public class Bobryakov_Spark {
    private static final String ANSWER = "ANSWER";
    private static final String QUESTION = "QUESTION";

    public static void main(String[] args) {
        String postsPath = args[0];
        String usersPath = args[1];
        String result = args[2];

        // Создание спарк-конфигурации и контекста
        SparkConf sparkConf = new SparkConf().setAppName("ABobryakov: Spark");
        sparkConf.set("spark.sql.shuffle.partitions", "16");
        /*sparkConf
                // 4 executor per instance of each worker
                .set("spark.executor.instances", "3")
                // 5 cores on each executor
                .set("spark.executor.cores", "3")
                // количество тасков на выходе (по умолч. 200)
                .set("spark.sql.shuffle.partitions", "10");*/
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // RDD для пользователей
        JavaPairRDD<String, String> users = sc.textFile(usersPath)
                .map(line -> XmlUtils.parseXmlToMap(line))
                .filter(user -> {
                    String location = user.get("Location");
                    return filterLocation(location);
                })
                .mapToPair(user -> new Tuple2<>(user.get("Id"), " Name: " + user.get("DisplayName") + " Location:" + user.get("Location")));

        // посты
        JavaPairRDD<Long, Iterable<PostData>> postJoinPost = sc.textFile(postsPath)
                .map(XmlUtils::parseXmlToMap)
                .filter(post -> {
                    String id = post.get("Id");
                    String parentId = post.get("ParentId");
                    String typeId = post.get("PostTypeId");
                    String ownerUserId = post.get("OwnerUserId");
                    String tags = post.get("Tags");
                    if (StringUtils.isNotBlank(typeId)
                            && Integer.parseInt(typeId) == 1
                            && StringUtils.isNotBlank(id)
                            && filterQuestionTags(tags)) {
                        // QUESTION
                        return true;
                    } else if (StringUtils.isNotBlank(typeId)
                            && StringUtils.isNotBlank(parentId)
                            && StringUtils.isNotBlank(ownerUserId)
                            && Integer.parseInt(typeId) == 2) {
                        // ANSWER
                        return true;
                    }

                    return false;
                })
                .map(post -> {
                    String id = post.get("Id");
                    String parentId = post.get("ParentId");
                    String typeId = post.get("PostTypeId");
                    String ownerUserId = post.get("OwnerUserId");
                    String score = post.get("Score");
                    PostData postData = new PostData();
                    if (Integer.parseInt(typeId) == 1) {
                        // QUESTION
                        postData.setType(QUESTION);
                        postData.setId(Long.parseLong(id));
                        postData.setOwnerUserId("");
                        postData.setScore(0);
                        return postData;
                    } else if (Integer.parseInt(typeId) == 2) {
                        // ANSWER
                        postData.setType(ANSWER);
                        postData.setId(Integer.parseInt(parentId));
                        postData.setOwnerUserId(ownerUserId);
                        postData.setScore(Integer.parseInt(score));
                        return postData;
                    } else {
                        throw new RuntimeException("Неихвестный тип");
                    }
                })
                .groupBy(PostData::getId);

        postJoinPost.filter(tuple -> {
                    // должен быть и вопрос и ответ
                    boolean existQ = false;
                    boolean existA = false;
                    for (PostData postData : tuple._2) {
                        if (postData.getType().equals(QUESTION)) {
                            existQ = true;
                        } else {
                            existA = true;
                        }
                    }
                    return existA && existQ;
                })
                .aggregateByKey(
                        new PostJoinData(),  // Объект для джойна с юзерами (ownerId - score)
                        (postJoinData, postData) -> {
                            int score = 0;
                            String ownerId = "";
                            for (PostData post : postData) {
                                if (post.getType().equals(ANSWER)) {
                                    score += post.getScore();
                                    ownerId = post.getOwnerUserId();
                                }
                            }
                            postJoinData.setScore(score);
                            postJoinData.setOwnerUserId(ownerId);
                            return postJoinData;
                        },
                        (postJoinData, postJoinData2) -> {
                            /*System.out.println("T|" + (postJoinData == null) + " " + (postJoinData2 == null));
                            if (postJoinData != null) {
                                System.out.println("A|" + postJoinData.getOwnerUserId() + "_" + postJoinData.getScore());
                            }
                            if (postJoinData2 != null) {
                                System.out.println("B|" + postJoinData2.getOwnerUserId() + "_" + postJoinData2.getScore());
                            }*/
                            PostJoinData res = new PostJoinData();
                            res.setOwnerUserId(postJoinData.getOwnerUserId());
                            int score = 0;
                            if (postJoinData != null) {
                                score += postJoinData.getScore();
                            }
                            if (postJoinData2 != null) {
                                score += postJoinData2.getScore();
                            }
                            res.setScore(score);
                            return res;
                        }
                )
                .mapToPair(tuple -> new Tuple2<>(tuple._2.getOwnerUserId(), tuple._2.getScore()))
                .join(users)
                .groupByKey()
                .mapToPair(tuple -> {
//                    tuple._1 STR  OwnerId - ключ для джойна
//                    tuple._2._1 INT  Score   (value от постов)
//                    tuple._2._2 STR  location+name   (value от юзеров)
                    int score = 0;
                    String locationAndName = "";
                    for (Tuple2<Integer, String> tup : tuple._2) {
                        score += tup._1;
                        locationAndName = tup._2;
                    }
                    return new Tuple2<>(score, "ID: " + tuple._1 + locationAndName);
                })
                .sortByKey(false)
                .map(pair -> pair._2 + "  Score: " + pair._1)
                .saveAsTextFile(result);
    }

    static class PostJoinData implements Serializable {
        String ownerUserId;
        int score = 0;

        public String getOwnerUserId() {
            return ownerUserId;
        }

        public void setOwnerUserId(String ownerUserId) {
            this.ownerUserId = ownerUserId;
        }

        public int getScore() {
            return score;
        }

        public void setScore(int score) {
            this.score = score;
        }
    }

    static class PostData implements Serializable {
        long id;
        String ownerUserId;
        String type;
        int score;

        public long getId() {
            return id;
        }

        public void setId(long id) {
            this.id = id;
        }

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public String getOwnerUserId() {
            return ownerUserId;
        }

        public void setOwnerUserId(String ownerUserId) {
            this.ownerUserId = ownerUserId;
        }

        public int getScore() {
            return score;
        }

        public void setScore(int score) {
            this.score = score;
        }
    }

    private static boolean filterQuestionTags(String tags) {
        return StringUtils.isNotBlank(tags)
                && !StringUtils.containsIgnoreCase(tags, "cript")
                && StringUtils.containsIgnoreCase(tags, "java");
    }

    private static boolean filterLocation(String location) {
        return StringUtils.isNotBlank(location)
                && (StringUtils.containsIgnoreCase(location, "oscow")
                || StringUtils.containsIgnoreCase(location, "russia"));
    }

}
