name := "mllib-naivebayes-demo"

version := "1.0"

scalaVersion := "2.10.4"

scalacOptions ++= Seq("-encoding", "UTF-8")

javacOptions ++= Seq("-encoding", "UTF-8")

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.0.1",
  "org.apache.spark" %% "spark-mllib"  % "1.0.1"
)

mainClass in (Compile, run) := Some("albert.demo.mllib.NaiveBayesScalaDemo")
