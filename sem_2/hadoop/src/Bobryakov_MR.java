package ru.mai.dep806.bigdata.mr;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

public class Bobryakov_MR extends Configured implements Tool {
    private static final String ANSWER = "ANSWER";
    private static final String QUESTION = "QUESTION";
    private static final String POST = "POST";
    private static final String USER = "USER";

    static final String[] POST_FIELDS = new String[]{
            "Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "CreationDate", "DeletionDate",
            "Score", "ViewCount", "OwnerUserId", "OwnerDisplayName", "LastEditorUserId",
            "LastEditorDisplayName", "LastEditDate", "LastActivityDate", "Title", "Tags", "AnswerCount",
            "CommentCount", "FavoriteCount", "ClosedDate", "CommunityOwnedDate"
    };

    static final String[] USER_FIELDS = new String[]{
            "Id", "Reputation", "CreationDate", "DisplayName", "LastAccessDate", "Location", "Views",
            "UpVotes", "DownVotes", "Age", "AccountId"
    };

    // Данные для Join Posts (Answer-Question)
    private static class PostData implements Writable {
        Text ownerUserId;
        Text type;
        int score;

        public PostData() {
            ownerUserId = new Text();
            type = new Text();
            score = 0;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            ownerUserId.write(dataOutput);
            type.write(dataOutput);
            dataOutput.writeInt(score);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            ownerUserId.readFields(dataInput);
            type.readFields(dataInput);
            score = dataInput.readInt();
        }

        public Text getOwnerUserId() {
            return ownerUserId;
        }

        public void setOwnerUserId(Text ownerUserId) {
            this.ownerUserId = ownerUserId;
        }

        public Text getType() {
            return type;
        }

        public void setType(Text type) {
            this.type = type;
        }

        public int getScore() {
            return score;
        }

        public void setScore(int score) {
            this.score = score;
        }
    }
    private static class PostsMapper extends Mapper<Object, Text, LongWritable, PostData> {
        private LongWritable outKey = new LongWritable();
        private PostData outValue = new PostData();

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Map<String, String> row = XmlUtils.parseXmlRow(value.toString());
            String parentId = row.get("ParentId");
            String id = row.get("Id");
            String postTypeId = row.get("PostTypeId");
            String ownerUserId = row.get("OwnerUserId");
            String tags = row.get("Tags");
            // обработка ANSWER
            if (StringUtils.isNotBlank(parentId)
                    && StringUtils.isNotBlank(ownerUserId)
                    && Integer.parseInt(postTypeId) == 2
            ) {
                outKey.set(Long.parseLong(parentId));
                outValue.setOwnerUserId(new Text(ownerUserId));
                String score = row.get("Score");
                if (StringUtils.isNotBlank(score)) {
                    outValue.setScore(Integer.parseInt(score));
                }
                outValue.setType(new Text(ANSWER));
                context.write(outKey, outValue);
            }
            // обработка QUESTION
            if (StringUtils.isNotBlank(id)
                    && Integer.parseInt(postTypeId) == 1
                    && filterQuestionTags(tags)
            ) {
                outKey.set(Long.parseLong(id));
                outValue.setType(new Text(QUESTION));
                outValue.setScore(0);
                outValue.setOwnerUserId(new Text());
                context.write(outKey, outValue);
            }
        }

        private boolean filterQuestionTags(String tags) {
            return StringUtils.isNotBlank(tags)
                    && !StringUtils.containsIgnoreCase(tags, "cript")
                    && StringUtils.containsIgnoreCase(tags, "java");
        }
    }
    private static class PostsJoinPostsReducer extends Reducer<LongWritable, PostData, LongWritable, LongWritable> {
        @Override
        protected void reduce(LongWritable key, Iterable<PostData> values, Context context) throws IOException, InterruptedException {
            PostData question = null;
            List<PostData> answers = new ArrayList<>();
            // key - id поста, являющегося QUESTION
            for (PostData val : values) {
                if (val.getType().toString().equals(QUESTION)) {
                    question = new PostData();
                    question.setOwnerUserId(val.getOwnerUserId());
                    question.setScore(val.getScore());
                    question.setType(val.getType());
                } else {
                    PostData temp = new PostData();
                    temp.setOwnerUserId(val.getOwnerUserId());
                    temp.setScore(val.getScore());
                    temp.setType(val.getType());
                    answers.add(temp);
                }
            }
            if (question != null && !answers.isEmpty()) {
                // для джойна с юзерами key - long(ownerId)
                // value - score ответа
                for (PostData answer : answers) {
                    if (StringUtils.isNotBlank(answer.getOwnerUserId().toString())) {
                        LongWritable userId = new LongWritable();
                        userId.set(Long.parseLong(answer.getOwnerUserId().toString()));
                        LongWritable score = new LongWritable();
                        score.set(Long.parseLong(answer.getScore() + ""));
                        context.write(userId, score);
                    }
                }

            }
        }
    }

    // Классы для процесса User join Posts
    private static class UserPostData implements Writable {
        Text displayName;
        Text location;
        Text type;
        long score;

        public UserPostData() {
            displayName = new Text();
            location = new Text();
            type = new Text();
            score = 0;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            displayName.write(dataOutput);
            location.write(dataOutput);
            type.write(dataOutput);
            dataOutput.writeLong(score);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            displayName.readFields(dataInput);
            location.readFields(dataInput);
            type.readFields(dataInput);
            score = dataInput.readLong();
        }

        public Text getDisplayName() {
            return displayName;
        }

        public void setDisplayName(Text displayName) {
            this.displayName = displayName;
        }

        public Text getLocation() {
            return location;
        }

        public void setLocation(Text location) {
            this.location = location;
        }

        public Text getType() {
            return type;
        }

        public void setType(Text type) {
            this.type = type;
        }

        public long getScore() {
            return score;
        }

        public void setScore(long score) {
            this.score = score;
        }
    }
    private static class UserMapper extends Mapper<Object, Text, LongWritable, UserPostData> {
        private LongWritable outKey = new LongWritable();
        private UserPostData outValue = new UserPostData();

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Map<String, String> row = XmlUtils.parseXmlRow(value.toString());

            String keyString = row.get("Id");
            String location = row.get("Location");
            if (StringUtils.isNotBlank(keyString) && filterLocation(location) && !location.isEmpty()) {
                outKey.set(Long.parseLong(keyString));
                String ownerDisplayName = row.get("DisplayName");
                if (StringUtils.isNotBlank(ownerDisplayName)) {
                    outValue.setDisplayName(new Text(ownerDisplayName));
                }
                outValue.setLocation(new Text(location));
                outValue.setScore(0);
                outValue.setType(new Text(USER));
                context.write(outKey, outValue);
            }
        }

        private boolean filterLocation(String location) {
            return StringUtils.isNotBlank(location)
                    && (StringUtils.containsIgnoreCase(location, "oscow")
                        || StringUtils.containsIgnoreCase(location, "russia"));
        }
    }
    private static class PostJoinedMapper extends Mapper<LongWritable, LongWritable, LongWritable, UserPostData> {
        private LongWritable outKey = new LongWritable();
        private UserPostData outValue = new UserPostData();

        @Override
        protected void map(LongWritable key, LongWritable value, Context context) throws IOException, InterruptedException {
            //  key - ownerId сджойненных ответов из пред. шага
            //  value = score поста
            if (key != null && value != null) {
//                outKey.set(Long.parseLong(key.toString()));
                outValue.setDisplayName(new Text());
                outValue.setLocation(new Text());
                outValue.setScore(value.get());
                outValue.setType(new Text(POST));
                context.write(key, outValue);
            }
        }
    }

    // Класс для результата User Join Posts
    private static class Result {
        long id;
        String name;
        String location;
        String type;
        long score;

        public String myToString() {
            String name = (this.name == null || this.name.isEmpty()) ? "-" : this.name;
            String location = (this.location == null || this.location.isEmpty()) ? "-" : this.location;
            String type = (this.type == null || this.type.isEmpty()) ? "-" : this.type;
            return " DisplayName:" + name + " Location:" + location + " Score:" + score;
        }

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

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getLocation() {
            return location;
        }

        public void setLocation(String location) {
            this.location = location;
        }

        public long getScore() {
            return score;
        }

        public void setScore(long score) {
            this.score = score;
        }
    }
    private static class UsersJoinPostsReducer extends Reducer<LongWritable, UserPostData, Text, Text> {

        private static List<Result> sortedList = new ArrayList<Result>() {
            @Override
            public boolean add(Result result) {
                super.add(result);
                sortedList.sort((o1, o2) -> Long.compare(o2.getScore(), o1.getScore()));
                return true;
            }
        };

        @Override
        protected void reduce(LongWritable key, Iterable<UserPostData> values, Context context) throws IOException, InterruptedException {
            // key - id юзера
            // values - UserPostData со всеми ответами по которым нужен sum(score) + сам человек (его инфа)

            Result userInfo = null;
            List<Result> answers = new ArrayList<>();

            // key - id поста, являющегося QUESTION
            for (UserPostData value : values) {
                String s = value.getType().toString();
                if (s.trim().equals(USER)) {
                    userInfo = new Result();
                    userInfo.setId(key.get());
                    userInfo.setName(value.getDisplayName().toString());
                    userInfo.setLocation(value.getLocation().toString());
                    userInfo.setScore(value.getScore());
                    userInfo.setType(value.getType().toString());
                } else if (s.trim().equals(POST)) {
                    Result ans = new Result();
                    ans.setId(0);
                    ans.setName("");
                    ans.setLocation("");
                    ans.setScore(value.getScore());
                    ans.setType(value.getType().toString());
                    answers.add(ans);
                } else {
                    throw new RuntimeException("Неизвестный тип " + s + "_" + s.length());
                }
            }

            if (userInfo != null && !answers.isEmpty()) {
                int sumScore = 0;
                for (Result answer : answers) {
                    sumScore += answer.getScore();
                }
                //
                userInfo.setScore(sumScore);
//                context.write(key, userInfo);
                System.out.println("R| " + userInfo.getId() + " " + userInfo.myToString());
                sortedList.add(userInfo);
                if (sortedList.size() > 100) {
                    sortedList.remove(sortedList.size() - 1);
                }
                // key - id юзера
                // value - объект с информацией о нем (имя, score, location, type(USER))
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Result userInfo : sortedList) {
                context.write(new Text(userInfo.getId() + ""), new Text(userInfo.myToString()));
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        Path postsPath = new Path(args[0]);  // где лежат посты
        Path userPath = new Path(args[1]);   // где лежат юзеры
        Path postsJoinPostsPath = new Path(args[2]);  //
        Path result = new Path(args[3]);

        // Создаем новую задачу (Job), указывая ее название
        Job job = Job.getInstance(getConf(), "ABobryakov: Posts join Posts. Step 1/2");
        // Указываем архив с задачей по имени класса в этом архиве
        job.setJarByClass(Bobryakov_MR.class);
        // Указываем класс Редьюсера
        job.setReducerClass(PostsJoinPostsReducer.class);
        // Кол-во тасков
        job.setNumReduceTasks(10);
        // Типы для мапперов
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(PostData.class);
        // Тип ключа на выходе
        job.setOutputKeyClass(LongWritable.class);
        // Тип значения на выходе
        job.setOutputValueClass(LongWritable.class);
        // Пути к входным файлам, формат файла и мэппер
        MultipleInputs.addInputPath(job, postsPath, TextInputFormat.class, PostsMapper.class);
        // Путь к файлу на выход (куда запишутся результаты)
        FileOutputFormat.setOutputPath(job, postsJoinPostsPath);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        // Запускаем джобу и ждем окончания ее выполнения
        boolean success = job.waitForCompletion(true);

        if (success) {
            Job job2 = Job.getInstance(getConf(), "ABobryakov: Users join Posts + Sort (Top 100). Step 2/2");
            // Указываем архив с задачей по имени класса в этом архиве
            job2.setJarByClass(Bobryakov_MR.class);
            // Указываем класс Редьюсера
            job2.setReducerClass(UsersJoinPostsReducer.class);
            // Кол-во тасков
            job2.setNumReduceTasks(1);
            // Типы для выхода мапперов
            job2.setMapOutputKeyClass(LongWritable.class);
            job2.setMapOutputValueClass(UserPostData.class);
            // Тип ключа на выходе
            job2.setOutputKeyClass(Text.class);
            // Тип значения на выходе
            job2.setOutputValueClass(Text.class);
            // Пути к входным файлам, формат файла и мэппер
            MultipleInputs.addInputPath(job2, userPath, TextInputFormat.class, UserMapper.class);
            MultipleInputs.addInputPath(job2, postsJoinPostsPath, SequenceFileInputFormat.class, PostJoinedMapper.class);

            // Путь к файлу на выход (куда запишутся результаты)
            FileOutputFormat.setOutputPath(job2, result);
//            job2.setOutputFormatClass(SequenceFileOutputFormat.class);

            // Запускаем джобу и ждем окончания ее выполнения
            success = job2.waitForCompletion(true);

        }
        // Возвращаем ее статус в виде exit-кода процесса
        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        // Let ToolRunner handle generic command-line options
        int result = ToolRunner.run(new Configuration(), new Bobryakov_MR(), args);

        System.exit(result);
    }
}
