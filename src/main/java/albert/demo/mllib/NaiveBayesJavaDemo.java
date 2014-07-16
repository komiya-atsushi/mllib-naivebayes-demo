package albert.demo.mllib;

import albert.demo.DataSet;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.stream.Collector;
import java.util.stream.Collectors;

/**
 * Naive Bayes によるスパムフィルタリングのデモ (Java 版)。
 *
 * @author KOMIYA Atsushi
 */
public class NaiveBayesJavaDemo {
    /** 全体での出現回数がこの指定値を下回る単語を切り捨てるようにします */
    static final int WORD_COUNT_LOWER_THRESHOLD = 8;

    /** 訓練データとして利用するデータの割合 (0 < TRAINING_SET_RATIO < 1) を指定します */
    static final double TRAINING_SET_RATIO = 0.5;

    public static void main(String[] args) {
        DataSet.SMSSpamCollection.prepareIfNeed();

        JavaSparkContext sc = new JavaSparkContext("local", "demo");

        JavaPairRDD<String, List<String>> rdd = sc.textFile("SMSSpamCollection")
                .mapToPair(line -> {
                    String[] elements = line.toLowerCase().replaceAll("[,.!?\"]", " ").split("\\s+");
                    return new Tuple2<>(elements[0], Arrays.stream(elements, 1, elements.length).collect(Collectors.toList()));
                });

        // 単語を特徴ベクトルのインデックスに変換できるように準備します
        Map<String, Integer> wordIndexes = rdd.flatMap(Tuple2::_2)
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey((countLeft, countRight) -> countLeft + countRight)
                .filter(wordCount -> wordCount._2() > WORD_COUNT_LOWER_THRESHOLD)
                .keys()
                .collect()
                .stream()
                .collect(Collector.of(
                        HashMap::new,
                        (BiConsumer<Map<String, Integer>, String>) (map, word) -> map.put(word, map.size()),
                        (BinaryOperator<Map<String, Integer>>) (left, right) -> {
                            left.putAll(right);
                            return left;
                        }));

        //  LabeledPoint に変換します
        JavaRDD<LabeledPoint> dataSet = rdd
                .map(record -> {
                    Map<Integer, Long> wordCounts = record._2()
                            .stream()
                            .filter(wordIndexes::containsKey)
                            .collect(Collectors.groupingBy(wordIndexes::get, Collectors.counting()));

                    Vector feature = Vectors.sparse(wordIndexes.size(),
                            wordCounts.entrySet()
                                    .stream()
                                    .map(entry -> new Tuple2<>(entry.getKey(), (double) entry.getValue()))
                                    .collect(Collectors.toList()));

                    return new LabeledPoint("ham".equals(record._1()) ? 1.0 : -1.0, feature);
                });

        long maxTrainingSetIndex = (long) (dataSet.count() * TRAINING_SET_RATIO);

        // 訓練データとテストデータに分割します
        JavaRDD<LabeledPoint> trainingSet = dataSet
                .zipWithIndex()
                .filter(t -> t._2() <= maxTrainingSetIndex)
                .map(Tuple2::_1);
        JavaRDD<LabeledPoint> testingSet = dataSet
                .zipWithIndex()
                .filter(t -> t._2() > maxTrainingSetIndex)
                .map(Tuple2::_1);

        // モデルを構築します
        NaiveBayesModel model = NaiveBayes.train(trainingSet.rdd());

        // テストデータでの分類をしつつ、その結果を評価します
        JavaRDD<String> result = testingSet.map(t -> {
            double predicted = model.predict(t.features());

            if (predicted == 1.0) {
                return t.label() == 1.0 ? "TN" : "FN";
            }

            return t.label() == -1.0 ? "TP" : "FP";
        });

        // 精度を算出します
        Map<String, Long> counts = result.countByValue();
        long totalCount = testingSet.count();

        double truePositiveCount = longValue(counts.get("TP"));
        double trueNegativeCount = longValue(counts.get("TN"));
        double falsePositiveCount = longValue(counts.get("FP"));
        double falseNegativeCount = longValue(counts.get("FN"));

        double accuracy = (truePositiveCount + trueNegativeCount) / totalCount;
        double falsePositiveRate = falsePositiveCount / (falsePositiveCount + trueNegativeCount);
        double falseNegativeRate = falseNegativeCount / (falseNegativeCount + truePositiveCount);

        System.out.println("accuracy: " + accuracy);
        System.out.println("false positive rate: " + falsePositiveRate);
        System.out.println("false negative rate: " + falseNegativeRate);
    }

    static long longValue(Long val) {
        if (val == null) {
            return 0;
        }
        return val;
    }
}
