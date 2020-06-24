java -cp target/lession12-1.0.1-SNAPSHOT.jar org.mai.dep810.scala.stackoverflow.Main init --forse

java -cp target/lession12-1.0.1-SNAPSHOT.jar org.mai.dep810.scala.stackoverflow.Main clean --dropTables

java -cp target/lession12-1.0.1-SNAPSHOT.jar org.mai.dep810.scala.stackoverflow.Main init --forse

java -cp target/lession12-1.0.1-SNAPSHOT.jar org.mai.dep810.scala.stackoverflow.Main load --path stackoverflow/

java -cp target/lession12-1.0.1-SNAPSHOT.jar org.mai.dep810.scala.stackoverflow.Main extract --query "select display_name, reputation, creation_date from users where views > 100" --file users.csv