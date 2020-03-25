from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Q2 A") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

lines = spark.read.load("Data/ratings.csv", format="csv", inferSchema="true", header="true").cache()

from functools import reduce
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import numpy as np 
import matplotlib.pyplot as plt

rmse_list1 = []
mae_list1 =[]
rmse_list2 = []
mae_list2 =[]
rmse_list3 = []
mae_list3 =[]
f =(f_1, f_2, f_3) = lines.randomSplit([0.33, 0.33, 0.34], seed = 10)
master = f.copy()

for index, value in enumerate(f):
    f = master.copy()
    testing = f.pop(index)
    training = reduce(DataFrame.unionAll, f)
    
    als1 = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
    als2 = ALS(maxIter=5, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
    als3 = ALS(maxIter=1, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
    # Model 1      
    model1 = als1.fit(training)
    predictions1 = model1.transform(testing)
    RMSE_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    MAE_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
    rmse1 = RMSE_evaluator.evaluate(predictions1)
    mae1 = MAE_evaluator.evaluate(predictions1)
    rmse_list1.append(rmse1)
    mae_list1.append(mae1)

    print(" Model 1: Root-mean-square error = " + str(rmse1))
    print("Model 1: Mean-absolute-error = " + str(mae1))
    
    # Model 2
    model2 = als2.fit(training)
    predictions2 = model2.transform(testing)
    rmse2 = RMSE_evaluator.evaluate(predictions2)
    mae2 = MAE_evaluator.evaluate(predictions2)
    rmse_list2.append(rmse2)
    mae_list2.append(mae2)

    print(" Model 2: Root-mean-square error = " + str(rmse2))
    print("Model 2: Mean-absolute-error = " + str(mae2))
    
     # Model 3
    model3 = als3.fit(training)
    predictions3 = model3.transform(testing)
    rmse3 = RMSE_evaluator.evaluate(predictions3)
    mae3 = MAE_evaluator.evaluate(predictions3)
    rmse_list3.append(rmse3)
    mae_list3.append(mae3)

    print(" Model 3: Root-mean-square error = " + str(rmse3))
    print("Model 3: Mean-absolute-error = " + str(mae3))

# Model 1    
# Mean for Root mean squared error
np.mean(rmse_list1)
mean1= np.mean(rmse_list1)
print("Model 1 : Mean for Root mean squared error = ", (mean1))
# Mean for mean absolute error
np.mean(mae_list1)
mean2= np.mean(mae_list1)
print("Model 1 : Mean for Mean absolute error = ", (mean2))
# Standard deviation for Root mean squared error
np.std(rmse_list1)
std1= np.std(rmse_list1)
print("Model 1: Standard deviation for Root-mean-squared error = ", (std1))
#Standard deviation of Mean absolute error
np.std(mae_list1)
std2= np.std(mae_list1)
print("Model 1: Standard deviation for Mean absolute error = ",(std2))

# Model 2   
# Mean for Root mean squared error
np.mean(rmse_list2)
mean3= np.mean(rmse_list2)
print("Model 2 : Mean for Root mean squared error = ", (mean3))
# Mean for mean absolute error
np.mean(mae_list2)
mean4= np.mean(mae_list2)
print("Model 2 : Mean for Mean absolute error = ", (mean4))
# Standard deviation for Root mean squared error
np.std(rmse_list2)
std3= np.std(rmse_list2)
print("Model 2: Standard deviation for Root-mean-squared error = ", (std3))
#Standard deviation of Mean absolute error
np.std(mae_list2)
std4= np.std(mae_list2)
print("Model 2: Standard deviation for Mean absolute error = ",(std4))

# Model 3   
# Mean for Root mean squared error
np.mean(rmse_list3)
mean5= np.mean(rmse_list3)
print("Model 3 : Mean for Root mean squared error = ", (mean5))
# Mean for mean absolute error
np.mean(mae_list3)
mean6= np.mean(mae_list3)
print("Model 3 : Mean for Mean absolute error = ", (mean6))
# Standard deviation for Root mean squared error
np.std(rmse_list3)
std5= np.std(rmse_list3)
print("Model 3: Standard deviation for Root-mean-squared error = ", (std5))
#Standard deviation of Mean absolute error
np.std(mae_list3)
std6= np.std(mae_list3)
print("Model 3: Standard deviation for Mean absolute error = ",(std6))

#Matplotlib graph for Q2 A 
# set width of bar
barWidth = 0.20
 
# Bars position - X axis
r1 = np.arange(len(rmse_list1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]



plt.bar(r1, rmse_list1, width=barWidth, label='RMSE Mean')
plt.bar(r2, mae_list2, width=barWidth, label='MAE Mean')
plt.bar(r3, std5, width=barWidth, label='RMSE std')
plt.bar(r4, std6, width=barWidth, label='MAE std')

 
# Labeling
plt.ylabel('Mean and std')
plt.xlabel('Three Version of ALS')
plt.xticks([r + barWidth for r in range(len(rmse_list1))], ['ALS1', 'ALS2', 'ALS3'])

# Displaying
plt.legend(loc='upper left')
plt.savefig('Q2_figA.png')
    

spark.stop()

