package jp.ac.utokyo.modernbigdata;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

// $example on$
import java.util.Arrays;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Scanner;
import java.io.Writer;
import java.io.IOException;
import scala.Tuple2;
import java.util.Map;
import java.util.HashMap;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.Statistics;
// $example off$

public class TestSpark {

    private static Scanner scanner = new Scanner(System.in);

    public void SparkCorr () 
    {
      System.out.println("Please enter the output name");
        String outfile = scanner.nextLine();

        SparkConf conf = new SparkConf().setAppName("JavaCorrelationsExample").setMaster("local[2]").set("spark.executor.memory","1g");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        try {
            Writer out = new BufferedWriter(new FileWriter(outfile));
            // $example on$
            JavaDoubleRDD seriesX = jsc.parallelizeDoubles(
            Arrays.asList(1.0, 2.0, 3.0, 3.0, 5.0));  // a series
      
          // must have the same number of partitions and cardinality as seriesX
            JavaDoubleRDD seriesY = jsc.parallelizeDoubles(
            Arrays.asList(11.0, 22.0, 33.0, 33.0, 555.0));
      
          // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method.
          // If a method is not specified, Pearson's method will be used by default.
            double correlation = Statistics.corr(seriesX.srdd(), seriesY.srdd(), "pearson");
      
          // note that each Vector is a row and not a column
            JavaRDD<Vector> data = jsc.parallelize(
                Arrays.asList(
              Vectors.dense(1.0, 10.0, 100.0),
              Vectors.dense(2.0, 20.0, 200.0),
              Vectors.dense(5.0, 33.0, 366.0)
            )
            );
      
          // calculate the correlation matrix using Pearson's method.
          // Use "spearman" for Spearman's method.
          // If a method is not specified, Pearson's method will be used by default.
            Matrix correlMatrix = Statistics.corr(data.rdd(), "pearson");
          // $example off$
            out.write("Correlation is: " + correlation + "\n");
            out.write("Correlation matrix: " + "\n" + correlMatrix.toString());
            out.close();

        } catch (IOException ex) {
            System.out.println("Error reading file");
        }
    
        jsc.stop();
    }

    public void BinaryClassificationMetrics() 
    {
      System.out.println("Please enter the input filename");
      String input = scanner.nextLine();
      System.out.println("Please enter the output filename");
      String outfile = scanner.nextLine();
      SparkConf conf = new SparkConf().setAppName("JavaCorrelationsExample").setMaster("local[*]").set("spark.executor.memory","1g");
      SparkContext sc = new SparkContext(conf);

      try {
        Writer out = new BufferedWriter(new FileWriter(outfile));
        // String path = "data/mllib/sample_binary_classification_data.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, input).toJavaRDD();
        
        // Split initial RDD into two parts
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        // Run training alogrithm to build the model
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                                          .setNumClasses(2)
                                          .run(training.rdd());
        model.clearThreshold();

        // Compute raw scores on the test set
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(
          p -> new Tuple2<>(model.predict(p.features()), p.label()));
        
        // Get evaluation metrics
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();

        // Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();        

        // F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();

        // Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();

        // Threshold
        JavaRDD<Double> threshold = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();

        out.write("The result of Logisitic Regression Model\n");
        out.write("Precision by threshold: \n" + precision.collect() + "\n\n");
        out.write("Recall by threshold: \n" + recall.collect() + "\n\n");
        out.write("F1 Score by threshold: \n" + f1Score.collect() + "\n\n");
        out.write("F2 Score by threshold: \n" + f2Score.collect() + "\n\n");
        out.write("Precision-recall curve: \n" + prc.collect() + "\n\n");
        out.write("ROC curve: \n" + roc.collect() + "\n\n");
        out.write("Area under precision-recall curve: \n" + metrics.areaUnderPR() + "\n\n");
        out.write("Area under ROC: \n" + metrics.areaUnderROC());
        out.close();

        model.save(sc, "target/tmp/LogisticRegressionModel");
        LogisticRegressionModel.load(sc, "target/tmp/LogisticRegressionModel");
      } catch (IOException ex) {
        System.out.println("Error reading file");
      }
      sc.stop();
    }

    public void RandomForestExample() {
      System.out.println("Please enter the input filename");
      String input = scanner.nextLine();
      System.out.println("Please enter the output filename");
      String output = scanner.nextLine();
      SparkConf conf = new SparkConf().setAppName("AppRandomForest").setMaster("local[*]").set("spark.executor.memory", "1g");
      SparkContext sc = new SparkContext(conf);

      try {
        Writer out = new BufferedWriter(new FileWriter(output));
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, input).toJavaRDD();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];
        
        int numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numTrees = 10;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        int maxDepth = 10;
        int maxBins = 100;
        int seed = 12345;

        RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses, 
          categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
        
        JavaPairRDD<Object, Object> predictionAndLabel = 
          testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        
        double testErr = predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testData.count();
        
        // Get evaluation metrics
        BinaryClassificationMetrics metrics_rf = new BinaryClassificationMetrics(predictionAndLabel.rdd());

        // Get presion by threshold
        JavaRDD<Tuple2<Object, Object>> presions = metrics_rf.precisionByThreshold().toJavaRDD(); 

        // Get Recall by threshold
        JavaRDD<?> recall = metrics_rf.recallByThreshold().toJavaRDD();

        // Get F score by threshold
        JavaRDD<?> f1score = metrics_rf.fMeasureByThreshold().toJavaRDD();
        JavaRDD<?> f2score = metrics_rf.fMeasureByThreshold(2.0).toJavaRDD();

        // Get Presion-recall curve
        JavaRDD<?> prc = metrics_rf.pr().toJavaRDD();

        // Get Threshold
        JavaRDD<Double> threshold = presions.map(t -> Double.parseDouble(t._1().toString()));

        // Get ROC curve
        JavaRDD<?> roc = metrics_rf.roc().toJavaRDD();

        out.write("The general information of model: \n" + model.toString() + "\n\n");
        out.write("Test Error: \n" + testErr + "\n\n");
        out.write("Presion by threshold: \n" + presions.collect() + "\n\n");
        out.write("Recall by threshold: \n" + recall.collect() + "\n\n");
        out.write("F1 Score by threshold: \n" + f1score.collect() + "\n\n");
        out.write("F2 Score by threshold: \n" + f2score.collect() + "\n\n");
        out.write("Presion-recall curve: \n" + prc.collect() + "\n\n");
        out.write("ROC curve: \n" + roc.collect() + "\n\n");
        out.write("Area under presion-recall curve: \n" + metrics_rf.areaUnderPR() + "\n\n");
        out.write("Area under ROC: \n" + metrics_rf.areaUnderROC() + "\n\n");
        out.write("Learned classification forest model: \n" + model.toDebugString() + "\n\n");
        out.close();

        model.save(sc, "target/tmp/RandomForestModel");
        RandomForestModel.load(sc, "target/tmp/RandomForestModel");
      } catch (IOException ex) {
        System.out.println("Error reading file");
      }

      sc.stop();
    }

    public static void main(String[] args) {
      TestSpark test = new TestSpark();
      // test.SparkCorr();   
      // test.BinaryClassificationMetrics();
      test.RandomForestExample();
    }
}
