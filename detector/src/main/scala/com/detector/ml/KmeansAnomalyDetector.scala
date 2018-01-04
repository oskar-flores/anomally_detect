package com.detector.ml

import java.text.SimpleDateFormat
import java.util.Calendar

import com.detector.batch.core.{calculateDistance, clusteringScore, maxByCentroid, write2file}
import main.scala.com.detector.Main.{DataSchema, TestPath}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf


class KmeansAnomalyDetector(private val spark: SparkSession, var data: DataFrame) {

  import spark.implicits._

  // Select only numerical features
  val CategoricalColumns = Seq("label", "protocol_type", "service", "flag")


  /**
    * Check if a point is an anomaly
    *
    * If the score of a point is higher than the maximum of the cluster
    * in which it belongs, it is an anomaly.
    *
    * UDF run on "dist" column and "prediction"
    *
    * @param max Map containing the maximal value of each cluster
    * @return 1 if the data is an anomaly, else 0
    */
  def checkAnomaly(max: Map[Int, Double]) = udf((distance: Double, k: Int) => if (distance > max(k)) 1 else 0)

  /**
    * Get all the anomalies of a test set
    *
    * @param pipeline  The pipeline used for the preprocessing
    * @param data      The test data
    * @param centroids The centroids found on the training data
    * @param max       Maximal value of each centroid
    * @return A DataFrame containing the anomalies
    */
  def getAnomalies(pipeline: PipelineModel, data: DataFrame, centroids: Array[Vector], max: Map[Int, Double]) = {
    val predictDF = pipeline.transform(data)

    val distanceDF = predictDF.withColumn("dist", calculateDistance(centroids)(predictDF("features"), predictDF("prediction"))).checkpoint()
    val anomalies = distanceDF.withColumn("anomaly", checkAnomaly(max)(distanceDF("dist"), distanceDF("prediction"))).checkpoint()
    anomalies.filter($"anomaly" > 0)
  }

  /**
    * Anomaly detection on test set
    *
    * Get the maximal value of each cluster, and check for each point
    * if its value is higher than the maximal, if this is the case, this is an anomaly.
    *
    * @param dataDF             Data of the training
    * @param pipelineModel      Pipeline model used with the training
    * @param totalClusterNumber Number of clusters
    * @return A DataFrame containing the anomalies
    */
  def anomalyDectection(dataDF: DataFrame, pipelineModel: PipelineModel, totalClusterNumber: Int): DataFrame = {
    // Load the data into the schema created previously
    val dataTestDF = spark.read.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .schema(DataSchema)
      .load(TestPath)

    val testDF = dataTestDF.drop("label")
    testDF.cache()

    // Prediction
    val cluster = pipelineModel.transform(dataDF)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // Get the centroids
    val centroids = kmeansModel.clusterCenters

    // Get the maximal distance for each cluster (on the training data)
    val max = maxByCentroid(centroids, cluster, totalClusterNumber)

    // Detect anomalies on the test data
    val anomalies = getAnomalies(pipelineModel, testDF, centroids, max)
    testDF.unpersist()
    anomalies
  }

  /**
    * K-means with only numerical features, without normalization
    *
    * @param clusterNumber Number of cluster
    */
  def kmeansSimple(clusterNumber: Int): Unit = {
    println(s"Running kmeansSimple ($clusterNumber)")
    val startTime = System.nanoTime()
    // Remove the label column
    val dataDF = this.data.drop("label")
    dataDF.cache()
    val numericalColumns = dataDF.columns.diff(CategoricalColumns)

    // Creation of vector with features
    val assembler = new VectorAssembler()
      .setInputCols(numericalColumns)
      .setOutputCol("features")

    val kmeans = new KMeans()
      .setK(clusterNumber)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(1L)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, kmeans))

    val pipelineModel = pipeline.fit(dataDF)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // Prediction
    val cluster = pipelineModel.transform(dataDF)

    // Get the centroids
    val centroids = kmeansModel.clusterCenters

    // Calculate the score
    val score = clusteringScore(centroids, cluster, clusterNumber)

    write2file(score, startTime, "K-means (" + clusterNumber + ") simple")

    // Anomaly detection
    val anomalies = this.anomalyDectection(dataDF, pipelineModel, clusterNumber)
    // Save results to json file
    val format = new SimpleDateFormat("yyyyMMddHHmm")
    Thread.sleep(1000)
    anomalies.write.json("anomalies_" + format.format(Calendar.getInstance().getTime) + "_" + clusterNumber + ".json")
    dataDF.unpersist()
  }

  /**
    * K-means using categorical features, without normalization
    *
    * Categorical features are encoded using the One-Hot encoder.
    *
    * @param clusterNumber Number of cluster
    */
  def kmeansOneHotEncoder(clusterNumber: Int): Unit = {
    println(s"Running kmeansOneHotEncoder ($clusterNumber)")
    val startTime = System.nanoTime()
    // Remove the label column
    val dataDF = this.data.drop("label")
    dataDF.cache()

    // Indexing categorical columns
    val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(s"${c}_index")
    ).toArray

    // Encoding previously indexed columns
    val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new OneHotEncoder()
        .setInputCol(s"${c}_index")
        .setOutputCol(s"${c}_vec")
        .setDropLast(false)
    ).toArray

    // Creation of list of columns for vector assembler (with only numerical columns)
    val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

    // Creation of vector with features
    val assembler = new VectorAssembler()
      .setInputCols(assemblerColumns)
      .setOutputCol("features")

    val kmeans = new KMeans()
      .setK(clusterNumber)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(1L)

    val pipeline = new Pipeline()
      .setStages(indexer ++ encoder ++ Array(assembler, kmeans))

    val pipelineModel = pipeline.fit(dataDF)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // Prediction
    val cluster = pipelineModel.transform(dataDF)
    dataDF.unpersist()

    // Get the centroids
    val centroids = kmeansModel.clusterCenters

    // Calculate the score
    val score = clusteringScore(centroids, cluster, clusterNumber)

    write2file(score, startTime, "K-means (" + clusterNumber + ") with one-hot encoder")
  }

  /**
    * K-means using categorical features, with normalization
    *
    * Categorical features are encoded using the One-hot encoder.
    * One-hot encoder will map a column of label indices to a column of binary vectors.
    * Normalization is done using the standard deviation
    *
    * @param k Number of cluster
    */
  def kmeansOneHotEncoderWithNormalization(k: Int): Unit = {
    println(s"Running kmeansOneHotEncoderWithNormalization ($k)")
    val startTime = System.nanoTime()
    // Remove the label column
    val dataDF = this.data.drop("label")
    dataDF.cache()

    // Indexing categorical columns
    val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(s"${c}_index")
    ).toArray

    // Encoding previously indexed columns
    val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new OneHotEncoder()
        .setInputCol(s"${c}_index")
        .setOutputCol(s"${c}_vec")
        .setDropLast(false)
    ).toArray

    // Creation of list of columns for vector assembler (with only numerical columns)
    val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

    // Creation of vector with features
    val assembler = new VectorAssembler()
      .setInputCols(assemblerColumns)
      .setOutputCol("featuresVector")

    // Normalization using standard deviation
    val scaler = new StandardScaler()
      .setInputCol("featuresVector")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans()
      .setK(k)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(1L)

    val pipeline = new Pipeline()
      .setStages(indexer ++ encoder ++ Array(assembler, scaler, kmeans))

    val pipelineModel = pipeline.fit(dataDF)

    // Prediction
    val cluster = pipelineModel.transform(dataDF)
    dataDF.unpersist()

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // Get the centroids
    val centroids = kmeansModel.clusterCenters

    // Calculate the score
    val score = clusteringScore(centroids, cluster, k)
    write2file(score, startTime, "K-means (" + k + ") with one-hot encoder with normalization")
  }

  /**
    * Bisecting K-means using categorical features, with normalization
    *
    * With the Bisecting K-means, al observations start in one cluster
    * and split are performed recursively in a "top-down" approach.
    *
    * Categorical features are encoded using the One-hot encoder.
    * One-hot encoder will map a column of label indices to a column of binary vectors.
    * Normalization is done using the standard deviation
    *
    * @param k Number of cluster
    */
  def bisectingKmeansOneHotEncoderWithNormalization(k: Int): Unit = {
    println(s"Running bisectingKmeansOneHotEncoderWithNormalization ($k)")
    val startTime = System.nanoTime()
    // Remove the label column
    val dataDF = this.data.drop("label")
    dataDF.cache()

    // Indexing categorical columns
    val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(s"${c}_index")
    ).toArray

    // Encoding previously indexed columns
    val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
      c => new OneHotEncoder()
        .setInputCol(s"${c}_index")
        .setOutputCol(s"${c}_vec")
        .setDropLast(false)
    ).toArray

    // Creation of list of columns for vector assembler (with only numerical columns)
    val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

    // Creation of vector with features
    val assembler = new VectorAssembler()
      .setInputCols(assemblerColumns)
      .setOutputCol("featuresVector")

    // Normalization using standard deviation
    val scaler = new StandardScaler()
      .setInputCol("featuresVector")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new BisectingKMeans()
      .setK(k)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(1L)

    val pipeline = new Pipeline()
      .setStages(indexer ++ encoder ++ Array(assembler, scaler, kmeans))

    val pipelineModel = pipeline.fit(dataDF)

    // Prediction
    val cluster = pipelineModel.transform(dataDF)
    dataDF.unpersist()

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[BisectingKMeansModel]

    // Get the centroids
    val centroids = kmeansModel.clusterCenters

    // Calculate the score
    val score = clusteringScore(centroids, cluster, k)

    write2file(score, startTime, "Bisecting K-means (" + k + ") with one-hot encoder with normalization")
  }
}

