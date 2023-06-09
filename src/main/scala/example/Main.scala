package example

import org.apache.spark.ml.bigdata.LinearRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, max, min, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructType}

import scala.collection.mutable
import scala.math

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("linregsgd")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    println("Hello, world")

    val path = "/mnt/e/BigDataMl/hw3/LinearRegression/insurance.csv"

    val schema = new StructType()
      .add("age", IntegerType, true)
      .add("sex", StringType, true)
      .add("bmi", DoubleType, true)
      .add("children", IntegerType, true)
      .add("smoker", StringType, true)
      .add("region", StringType, true)
      .add("charges", DoubleType, true)

    var df = spark.read.format("csv")
      .option("header", "true")
      .schema(schema)
      .load(path)

    val indexer = new StringIndexer()
      .setInputCols(Array("sex", "smoker", "region"))
      .setOutputCols(Array("sexIndex", "smokerIndex", "regionIndex"))

    df = indexer.fit(df).transform(df)

    val encoder = new OneHotEncoder()
      .setDropLast(false)
      .setInputCol("regionIndex")
      .setOutputCol("regionVec")

    df = encoder.fit(df).transform(df)

    val toArrUdf: UserDefinedFunction = udf { x: SparseVector =>
      x.toArray
    }
    df = df.withColumn("regionVec_arr", toArrUdf(col("regionVec")))

    val numberOfRegions = df.first().get(df.first().fieldIndex("regionVec_arr"))
      .asInstanceOf[mutable.WrappedArray[Double]].length
    println(numberOfRegions)

    df = df.select(
      col("age").cast(DoubleType)
        +: col("sexIndex")
        +: col("bmi")
        +: col("children").cast(DoubleType)
        +: col("smokerIndex")
        +: col("charges")
        +: (0 until numberOfRegions).map(i => col("regionVec_arr")(i).alias(s"col$i")): _*
    )

    df.show()
    df.printSchema()

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("age", "sexIndex", "bmi", "children", "smokerIndex", "col0", "col1", "col2", "col3"))
      .setOutputCol("features1")

    df = assembler.transform(df).withColumn("label", col("charges"))

    val scaler = new StandardScaler()
      .setInputCol("features1")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val minLabel = df.select(sql.functions.min("label")).first().getDouble(0)
    val maxLabel = df.select(sql.functions.max("label")).first().getDouble(0)

    val normalize: UserDefinedFunction = udf { x: Double =>
      (x - minLabel)/(maxLabel - minLabel)
    }

    df = scaler.fit(df).transform(df)
      .select(col("features"), normalize(col("label")).alias("label"))
//    df = labelScaler.fit(df).transform(df)


    df.show(false)
    df.printSchema()

    val Array(train, test) = df.randomSplit(Array[Double](0.9, 0.1), 18)

    val regressor = new LinearRegression("testing7")
    val model = regressor.train(train)

    val output = model.transform(test)

    output.show()
    output.printSchema()

    val predicted_labels = output.select("prediction").collect().map(_.getDouble(0))
    val actual_labels = test.select("label").collect().map(_.getDouble(0))

//    println(train.first())
//    println(test.first())
    println("coefficients: " + model.weights)
    println("First prediction: " + model.predict(test.first().getAs[Vector]("features")))
//    println("kek1")
    var total = 0.0
    for (i <- predicted_labels.indices) {
      total += math.abs(predicted_labels.apply(i) - actual_labels.apply(i))
      println(math.abs(predicted_labels.apply(i) - actual_labels.apply(i)))
    }
    println("total error:" + total / predicted_labels.length)
//    println("kek2")
  }
}
