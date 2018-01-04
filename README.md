
Basic implementation is based on the chapter 5 (Anomaly Detection in Network Traffic with K-means clustering)
of the book Advanced Analytics with Spark.

Algorithms:

 - K-means

Categorical features are transformed into numerical features using one-hot encoder.
Afterwards, all features are normalized.

Metrics used:

 - Sum of distances between points and their centroids


Anomaly detection is done as follow:

  - Find the maximal value of each cluster, those will be the thresholds
  - For a new point, calculate its score (distance), if it is more than the threshold of its cluster,
this is an anomaly

Datasource: https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data
Test set:  http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html (corrected.gz)
