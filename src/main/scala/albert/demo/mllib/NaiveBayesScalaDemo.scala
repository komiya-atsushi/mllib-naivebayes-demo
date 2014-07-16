package albert.demo.mllib

import albert.demo.DataSet
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

/**
 * Naive Bayes によるスパムフィルタリングのデモ (Scala 版)。
 *
 * @author KOMIYA Atsushi
 */
object NaiveBayesScalaDemo {
  /** 全体での出現回数がこの指定値を下回る単語を切り捨てるようにします */
  val WORD_COUNT_LOWER_THRESHOLD = 8

  /** K-fold cross-validation の K に相当します */
  val NUM_FOLDS = 5

  def main(args: Array[String]) {
    DataSet.SMSSpamCollection.prepareIfNeed()

    val sc = new SparkContext("local", "demo")

    val preparedData = sc.textFile("SMSSpamCollection")
      .filter(line => line.startsWith("ham") || line.startsWith("spam"))
      .map(_.toLowerCase.replaceAll("[,.!?\"]", " ").split("\\s+"))
      .map(wordsInSms => (wordsInSms.head, wordsInSms.tail))

    // 単語を特徴ベクトルのインデックスに変換できるように準備します
    val wordIndexes = preparedData.flatMap(_._2)
      .map((_, 1))
      .reduceByKey(_ + _)
      .filter(_._2 > WORD_COUNT_LOWER_THRESHOLD)
      .keys
      .zipWithIndex()
      .collectAsMap()

    // LabeledPoint に変換します
    val dataSet = preparedData.map { case (label, words) =>
      val wordCounts = words
        .filter(wordIndexes.contains)
        .map(wordIndexes(_).toInt)
        .groupBy(_.toInt)
        .mapValues(_.length.toDouble)

      LabeledPoint(
        if ("ham".equals(label)) 1.0 else -1.0,
        Vectors.sparse(wordIndexes.size, wordCounts.toSeq)
      )
    }

    // 訓練データとテストデータの組を NUM_FOLDS 個用意します (K-fold cross-validation)
    val foldedDataSet = MLUtils.kFold(dataSet, NUM_FOLDS, System.currentTimeMillis().toInt)

    val result = foldedDataSet.map { case (trainingSet, testingSet) =>
      // モデルを構築します
      val model = NaiveBayes.train(trainingSet)

      // テストデータので分類をしつつ、その結果を評価します
      testingSet.map { t =>
        val predicted = model.predict(t.features)

        if (predicted == 1.0) {
          if (t.label == 1.0) "TN" else "FN"
        } else {
          if (t.label == -1.0) "TP" else "FP"
        }
      }.countByValue()
    }

    // 精度を算出します
    val totalCount = dataSet.count()

    val truePositiveCount = result.map(_("TP")).reduce(_ + _).toDouble
    val trueNegativeCount = result.map(_("TN")).reduce(_ + _).toDouble
    val falsePositiveCount = result.map(_("FP")).reduce(_ + _).toDouble
    val falseNegativeCount = result.map(_("FN")).reduce(_ + _).toDouble

    val accuracy = (truePositiveCount + trueNegativeCount) / totalCount
    val falsePositiveRate = falsePositiveCount / (falsePositiveCount + trueNegativeCount)
    val falseNegativeRate = falseNegativeCount / (falseNegativeCount + truePositiveCount)

    println("accuracy: " + accuracy)
    println("false positive rate: " + falsePositiveRate)
    println("false negative rate: " + falseNegativeRate)
  }
}
