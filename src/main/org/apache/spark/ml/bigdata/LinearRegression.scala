//@Author lexzhe
package org.apache.spark.ml.bigdata

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.types.{DataType, StructType}

trait LinearRegressionParams extends PredictorParams with HasMaxIter {
  // Parameters for LinearRegression Model
  private final val learningRate = new DoubleParam(this, "learningRate", "Learning rate", ParamValidators.gtEq(0))
  private final val tolerance = new DoubleParam(this, "tolerance", "Tolerance of solution")
  private final val batchSize = new IntParam(this, "batchSize", "Batch of gradient descent size")

  def getLearningRate: Double = $(learningRate)

  def getTolerance: Double = $(tolerance)

  def getBatchSize: Int = $(batchSize)

  setDefault(
    maxIter -> 1000,
    learningRate -> 1.0,
    tolerance -> 1e-6,
    batchSize -> 10000)

  override protected def validateAndTransformSchema(schema: StructType,
                                                    fitting: Boolean,
                                                    featuresDataType: DataType): StructType = {
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Regressor[Vector, LinearRegression, LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable with Logging {
  override def train(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

    val rows: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val dim = rows.first()._1.size

    val maxIter = getMaxIter
    val lr = getLearningRate
    val eps = getTolerance
    val batchSize = getBatchSize
    var weight = DenseMatrix.zeros[Double](dim, 1) // 3 * 1

    var diffNorm = Double.PositiveInfinity
    var curIter = 0

    // Stop evaluation if differential is lesser then tolerance or all iterations pass
    while (curIter < maxIter && diffNorm > eps) {
      val grad = rows.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        Iterator(
          data.sliding(batchSize, batchSize).foldLeft(new MultivariateOnlineSummarizer())(
            (summarizer, batchRows) => {
              // Matrix computation
              val rows = batchRows.map(row => row._1.asBreeze.toArray).toArray
              val matrix = new DenseMatrix(dim, batchRows.size, rows.flatten).t //batchSize * 3
              val y = new DenseVector(batchRows.map(x => x._2).toArray).asDenseMatrix.t // batchSize * 1
              val y_hat = matrix * weight - y // batchSize * 1
              val grads = matrix.t * (y_hat) / batchRows.length.toDouble // 3 * 1
              summarizer.add(mllib.linalg.Vectors.fromBreeze(grads.toDenseVector))
            }
          )
        )
      }).reduce(_ merge _)

      val diff = lr * grad.mean.asBreeze
      weight = weight - diff.toDenseVector.asDenseMatrix.t // 3 * 1
      curIter += 1
      diffNorm = norm(diff)
    }

    copyValues(new LinearRegressionModel(weight.flatten()))
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[bigdata](
                                              override val uid: String,
                                              val weights: DenseVector[Double],
                                            ) extends RegressionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[bigdata] def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("LinearRegressionModel"), weights)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights), extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val tupled = Tuple1(Vectors.fromBreeze(weights))
      sqlContext.createDataFrame(Seq(tupled)).write.parquet(path + "/weights")
    }
  }

  override def predict(features: Vector): Double = features.asBreeze dot weights
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/weights")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val weight = vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(weight)
      metadata.getAndSetParams(model)
      model
    }
  }
}
