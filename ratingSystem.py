from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

cd MyDrive/Colab\ Notebooks

!pip install pyspark

from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .getOrCreate()

spark

df = spark.read.csv("movielens_ratings.csv",header=True, inferSchema=True).cache()

df.printSchema()

df.show(3)

df.describe().show()

train, test = df.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')

model = als.fit(train)
predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
rmse = evaluator.evaluate(predictions)
print('RMSE:', rmse)

this_user = test.filter(test['userId'] == 12).select('userId', 'movieId')
this_user.show()


recommendation_this_user = model.transform(this_user)
recommendation_this_user.show()

recommendation_this_user.orderBy('prediction', ascending=False).show()