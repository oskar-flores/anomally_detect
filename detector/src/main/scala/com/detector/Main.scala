package main.scala.com.detector

/**
  * Anomaly Detection in Network Traffic with different clustering algorithm.
  *
  * The implementation is done using the DataFrame-based API of SparkMLlib.
  *
  *
  * Basic implementation is based on the chapter 5 (Anomaly Detection in Network Traffic with K-means clustering)
  * of the book Advanced Analytics with Spark.
  *
  * Algorithms:
  *
  *  - K-means
  *
  * Categorical features are transformed into numerical features using one-hot encoder.
  * Afterwards, all features are normalized.
  *
  *
  * Metrics used:
  *
  *  - Sum of distances between points and their centroids
  *
  *
  * Anomaly detection is done as follow:
  *
  *   - Find the maximal value of each cluster, those will be the thresholds
  *   - For a new point, calculate its score (distance), if it is more than the threshold of its cluster,
  * this is an anomaly
  *
  * Datasource: https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data
  * Test set:  http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html (corrected.gz)
  */

import com.detector.ml.KmeansAnomalyDetector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

object Main {

  val DataPath = "data/kddcup.data.corrected"
  val TestPath = "data/test.data.corrected"

  // Fraction of the dataset used (1.0 for the full dataset)
  val Fraction = 1.0

  // Schema of data from json file
  // This should be different depending the schema. Worts thinking on load as a DI case class
  val DataSchema = StructType(Array(
    StructField("duration", IntegerType, nullable = true),
    StructField("protocol_type", StringType, nullable = true),
    StructField("service", StringType, nullable = true),
    StructField("flag", StringType, nullable = true),
    StructField("src_bytes", IntegerType, nullable = true),
    StructField("dst_bytes", IntegerType, nullable = true),
    StructField("land", IntegerType, nullable = true),
    StructField("wrong_fragment", IntegerType, nullable = true),
    StructField("urgent", IntegerType, nullable = true),
    StructField("hot", IntegerType, nullable = true),
    StructField("num_failed_logins", IntegerType, nullable = true),
    StructField("logged_in", IntegerType, nullable = true),
    StructField("num_compromised", IntegerType, nullable = true),
    StructField("root_shell", IntegerType, nullable = true),
    StructField("su_attempted", IntegerType, nullable = true),
    StructField("num_root", IntegerType, nullable = true),
    StructField("num_file_creations", IntegerType, nullable = true),
    StructField("num_shells", IntegerType, nullable = true),
    StructField("num_access_files", IntegerType, nullable = true),
    StructField("num_outbound_cmds", IntegerType, nullable = true),
    StructField("is_host_login", IntegerType, nullable = true),
    StructField("is_guest_login", IntegerType, nullable = true),
    StructField("count", IntegerType, nullable = true),
    StructField("srv_count", IntegerType, nullable = true),
    StructField("serror_rate", DoubleType, nullable = true),
    StructField("srv_serror_rate", DoubleType, nullable = true),
    StructField("rerror_rate", DoubleType, nullable = true),
    StructField("srv_rerror_rate", DoubleType, nullable = true),
    StructField("same_srv_rate", DoubleType, nullable = true),
    StructField("diff_srv_rate", DoubleType, nullable = true),
    StructField("srv_diff_host_rate", DoubleType, nullable = true),
    StructField("dst_host_count", IntegerType, nullable = true),
    StructField("dst_host_srv_count", IntegerType, nullable = true),
    StructField("dst_host_same_srv_rate", DoubleType, nullable = true),
    StructField("dst_host_diff_srv_rate", DoubleType, nullable = true),
    StructField("dst_host_same_src_port_rate", DoubleType, nullable = true),
    StructField("dst_host_srv_diff_host_rate", DoubleType, nullable = true),
    StructField("dst_host_serror_rate", DoubleType, nullable = true),
    StructField("dst_host_srv_serror_rate", DoubleType, nullable = true),
    StructField("dst_host_rerror_rate", DoubleType, nullable = true),
    StructField("dst_host_srv_rerror_rate", DoubleType, nullable = true),
    StructField("label", StringType, nullable = true)))

  def main(args: Array[String]): Unit = {
    // Creation of configuration and session
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("NetworkAnomalyDetection")
      .set("spark.driver.memory", "6g")

    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoints/")

    val spark = SparkSession
      .builder()
      .appName("NetworkAnomalyDetection")
      .getOrCreate()

    // Load the data into the schema created previously
    val rawDataDF = spark.read
      .option("header", "false")
      .option("inferSchema", "true")
      .schema(DataSchema)
      .csv(DataPath)

    val dataDF = rawDataDF.sample(withReplacement = false, Fraction, 42)
    println("Size of dataset=" + dataDF.count + " (total=" + rawDataDF.count + ")")
    val detector = new KmeansAnomalyDetector(spark, dataDF)

    //Can choose between two of this, probably needs to separate int the generation fro the usage
    //Check time between and select one of them.

    // K-means
    // K-means simple is also doing anomaly detections.
    (20 to 100 by 10).map(k => (k, detector.kmeansOneHotEncoderWithNormalization(k)))

  }
}
