package com.detector.batch

import java.io.{File, PrintWriter}
import java.text.SimpleDateFormat
import java.time.ZonedDateTime
import java.util.Calendar

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row, SaveMode}

import scala.collection.mutable.ArrayBuffer


package object core {

  /**
    * Saves the data from a DataFrame to the specified path
    *
    * @param dataFrame     The dataframe containing the data to save
    * @param mode          The write mode, it defaults to "overwrite"
    * @param numPartitions The number of partitions to split the file
    * @param path          The path to save the file
    * @param format        The format to use, it defaults to "parquet"
    */
  def save(dataFrame: DataFrame, mode: SaveMode = SaveMode.Overwrite, numPartitions: Int = 200, path: String, format: String = "parquet"): Unit = {
    dataFrame
      .coalesce(numPartitions)
      .write
      .format(format)
      .mode(mode)
      .save(path)
  }

  def saveCsv(dataFrame: DataFrame, mode: SaveMode = SaveMode.Overwrite, numPartitions: Int = 200, path: String, delimiter: String = "|", header: String = "false"): Unit = {
    dataFrame
      .coalesce(numPartitions)
      .write
      .format("com.databricks.spark.csv")
      .option("delimiter", delimiter)
      .option("header", header)
      .mode(mode)
      .save(path)
  }

  /**
    * Loads the Data from an array of paths and returns a Dataframe
    *
    * @param context Hive enviroment that manages the call
    * @param paths   Array of paths used to load data
    * @param format  The format to use, it defaults to "parquet"
    * @return
    */
  def load(context: HiveContext, paths: Seq[String], format: String = "parquet"): DataFrame = {
    context
      .read
      .format(format)
      .load(paths: _*)
  }

  def load(context: HiveContext, path: String, format: String): DataFrame = {
    load(context, paths = Seq(path), format)
  }

  def load(context: HiveContext, path: String): DataFrame = {
    load(context, paths = Seq(path))
  }

  /**
    * Gets an Array of paths given a base path and a number of days to get
    *
    * @param basePath        Path of folders to get
    * @param daysToSubstract number of days to get backwards
    * @param baseDate        Date of reference
    * @return
    */
  def getFoldersWithPrefix(basePath: String, daysToSubstract: Int, baseDate: ZonedDateTime): Seq[String] = {
    var paths = new ArrayBuffer[String](daysToSubstract)

    for (i <- 1 to daysToSubstract) {
      val path = f"$basePath/${baseDate.minusDays(i).getYear}/${baseDate.minusDays(i).getMonthValue}%02d/${baseDate.minusDays(i).getDayOfMonth}%02d/*"
      paths += path
    }
    paths
  }

  /**
    * Calculate the Euclidean distance between a data point and its centroid
    *
    * @param centroid Vector with the components of the centroid
    * @param data     Vector with the components of the data point
    * @return The distance between the data point and the centroid
    */
  def distance(centroid: Vector, data: Vector): Double =
  // Tranforming vector to array of double since operations
  // on vector are not implemented
    math.sqrt(centroid.toArray.zip(data.toArray)
      .map(p => p._1 - p._2).map(d => d * d).sum)

  /**
    * Apply the Euclidean distance between all points belonging to a centroid and the centroid in question
    *
    * @param centroid     Vector with the components of the centroid
    * @param dataCentroid All data points (as Vector) belonging to the centroid
    * @return An array of double containing all the distance of a cluster (data with same centroid)
    */
  def distanceAllCluster(centroid: Vector, dataCentroid: Array[DenseVector]): Array[Double] = {
    dataCentroid.map(d => distance(centroid, d))
  }

  /**
    * Calculate the score of a cluster
    *
    * For each k, select data belonging to the centroid
    * and calculating the distance.
    *
    * @param centroids Array containing all the centroids
    * @param data      Dataset used
    * @param k         Number of cluster
    * @return The mean of the score from all cluster
    */
  def clusteringScore(centroids: Array[Vector], data: DataFrame, k: Int): Double = {
    val score = (0 until k).map { k =>
      val dataCentroid = data.filter(col("prediction") === k)
        .select("features")
        .collect()
        .map {
          // Get the feature vectors in dense format
          case Row(v: Vector) => v.toDense
        }
      val s = distanceAllCluster(centroids(k), dataCentroid)
      if (s.length > 0)
        s.sum / s.length
      else
        s.sum // Sum will be 0 if no element in cluster
    }
    if (score.nonEmpty)
      score.sum / score.length
    else
      score.sum
  }

  /**
    * Get the maximum value of each centroid
    *
    * @param centroids Array containing all the centroids
    * @param data      DataFrame containing the data points
    * @param k         The number of cluster
    * @return A Map with k as the key and its maximum value as value
    */
  def maxByCentroid(centroids: Array[Vector], data: DataFrame, k: Int): Map[Int, Double] = {

    val max = (0 until k).map { k =>
      val dataCentroid = data.filter(col("prediction") === k)
        .select("features")
        .collect()
        .map {
          // Get the feature vectors in dense format
          case Row(v: Vector) => v.toDense
        }
      val dist = distanceAllCluster(centroids(k), dataCentroid)
      if (dist.isEmpty) {
        (k, 0.0)
      }
      else
        (k, dist.max)
    }
    max.toMap
  }

  /**
    * Calculate the distance between a point and its centroid
    *
    * This is an udf and must be run on a DataFrame.
    * Usage of currying in order to pass other parameters.
    *
    * The columns of the DataFrame to use: "features" and "prediction"
    * Uses the prediction column to know in which centroid the point belongs.
    *
    * @param centroids Centroids
    * @return
    */
  def calculateDistance(centroids: Array[Vector]) = udf((v: Vector, k: Int) => {
    math.sqrt(centroids(k).toArray.zip(v.toArray)
      .map(p => p._1 - p._2).map(d => d * d).sum)
  })

  /**
    * Write the result of a run into a file
    *
    * Filename is create dynamically with the current date and the algorithm used.
    *
    * @param score     Score already calculated
    * @param startTime Start time of the computation
    * @param technique String with the name of the algorithm/preprocessing used
    */
  def write2file(score: Double, startTime: Long, technique: String): Unit = {
    val format = new SimpleDateFormat("yyyyMMddHHmm")
    val pw = new PrintWriter(new File("results" + format.format(Calendar.getInstance().getTime) +
      "_" + technique.replaceAll(" ", "_") + ".txt"))
    try {
      println(technique)
      pw.write(s"$technique\n")
      println(s"Score=$score")
      pw.write(s"Score=$score\n")
      val duration = (System.nanoTime - startTime) / 1e9d
      println(s"Duration=$duration")
      pw.write(s"Duration=$duration\n")
    } finally {
      pw.close()
    }
  }

}
