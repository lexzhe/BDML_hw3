package scala.example

import org.apache.spark
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.SparkSession

class Main2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("linregsgd")
      .master("local[*]")
      .getOrCreate()

    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
  }
}
