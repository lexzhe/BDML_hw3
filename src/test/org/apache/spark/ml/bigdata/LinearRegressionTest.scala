//@Author lexzhe
package org.apache.spark.ml.bigdata

import breeze.linalg.DenseVector
import com.google.common.io.Files
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{ Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val eps = 0.0001
  lazy val testData: Dataset[_] = LinearRegressionTest._testData
  lazy val actual_labels: DenseVector[Double] = LinearRegressionTest._labels
  lazy val coefficients: DenseVector[Double] = DenseVector(LinearRegressionTest._coefficients.toArray)

  private def validateModel(output: DataFrame): Unit = {
    output.show()
    val predicted_labels = output.select("prediction").collect()
    // Check if labels are equal
    predicted_labels.length should be(actual_labels.length)
    for (i <- predicted_labels.indices) {
      predicted_labels.apply(i).getDouble(0) should be(actual_labels(i) +- eps)
    }
  }

  private def validateCoefficients(actual: DenseVector[Double]): Unit = {
    for (i <- 0 until coefficients.length) {
      actual(i) should be(coefficients(i) +- eps)
    }
  }

  "Model" should "predict function without fit" in {
    val model = new LinearRegressionModel(coefficients)
    validateModel(model.transform(testData))
  }

  "Regressor" should "have correct weights (coefficients) after fit" in {
    val estimator = new LinearRegression("testing1")
    val model = estimator.train(testData)
    validateCoefficients(model.weights)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression("testing3")
    ))
    val model = pipeline.fit(testData)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reReadModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reReadModel.transform(testData))
  }

  "Regressor" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression("testing2")
    ))
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reReadRegressor = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reReadRegressor.fit(testData).stages(0).asInstanceOf[LinearRegressionModel]

    validateCoefficients(model.weights)
    validateModel(model.transform(testData))
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _coefficients: Vector = Vectors.dense(1.5, 0.3, -0.7)

  lazy val randomizer = new scala.util.Random(1)
  // DF with random doubles
  lazy val df: DataFrame = spark.createDataFrame(
    spark.sparkContext.parallelize(
      Seq.fill(100000) {(randomizer.nextDouble, randomizer.nextDouble, randomizer.nextDouble)}
    )
  ).toDF("x1", "x2", "x3")

  // Assembler to convert 3 columns of type Double to 1 column of type Vector[Double]
  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x1", "x2", "x3"))
    .setOutputCol("features")

  lazy val makeLabelColumn: UserDefinedFunction = udf { x: Vector =>
    x.asBreeze dot _coefficients.asBreeze
  }
  lazy val _testData: DataFrame = {
    assembler.transform(df).withColumn("label", makeLabelColumn(col("features")))
  }

  lazy val _labels: DenseVector[Double] = DenseVector(_testData.select("label").collect().map(_.getAs[Double](0)))

}


