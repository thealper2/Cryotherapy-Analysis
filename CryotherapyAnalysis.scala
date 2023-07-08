import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

object CryotherapyAnalysis {
	def main(): Unit = {
		//val spark = SparkSession.buidler
		//	.master("local[4]")
		//	.appName("CryotherapyAnalysis")
		//	.getOrCreate()

		import spark.implicits._

		val df_path = "/home/alper/Spark/Scala/CryotherapyPrediction/Cryotherapy.csv"

		var df = spark.read
			.option("header", true)
			.option("inferSchema", true)
			.csv(df_path)

		df.show(10)
		df.printSchema()

		df = df.withColumnRenamed("Result_of_Treatment", "label")

		val vector_assembler = new VectorAssembler()
			.setInputCols(Array("sex", "age", "Time", "Number_of_Warts", "Type", "Area"))
			.setOutputCol("features")

		val assembleDF = vector_assembler.transform(df)
			.select("label", "features")

		assembleDF.show(10)

		val seed = 4242
		val splits = assembleDF.randomSplit(Array(0.8, 0.2), seed)
		val (train_df, test_df) = (splits(0), splits(1))

		train_df.cache
		test_df.cache

		val dt = new DecisionTreeClassifier()
			.setImpurity("gini")
			.setMaxBins(10)
			.setMaxDepth(30)
			.setLabelCol("label")
			.setFeaturesCol("features")

		val dt_model = dt.fit(train_df)

		val evaluator = new BinaryClassificationEvaluator()
			.setLabelCol("label")

		val predictionDF = dt_model.transform(test_df)
		val result = predictionDF.select("label", "prediction", "probability")
		result.show(10)

		val accuracy = evaluator.evaluate(predictionDF)
		println("Classification Accuracy: " + accuracy)

		val preds = predictionDF.select("label", "prediction")
		val total = predictionDF.count()

		val tp = preds
			.filter($"prediction" === 0.0)
			.filter($"label" === $"prediction").count() / total.toDouble

		val tn = preds
			.filter($"prediction" === 1.0)
			.filter($"label" === $"prediction").count() / total.toDouble

		val fp = preds
			.filter($"prediction" === 1.0)
			.filter(not($"label" === $"prediction")).count() / total.toDouble

		val fn = preds
			.filter($"prediction" === 0.0)
			.filter(not($"label" === $"prediction")).count() / total.toDouble

		println("True Positive: " + tp * 100 + "%")
		println("True Negative: " + tn * 100 + "%")
		println("False Positive: " + fp * 100 + "%")
		println("False Negative: " + fn * 100 + "%")

		spark.stop()
	}
}
