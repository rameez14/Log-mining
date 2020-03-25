from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Q1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile=spark.read.text("Data/NASA_access_log_Jul95.gz").cache()

import matplotlib.pyplot as plt


# Question 1A and 1B
Hours = ['00:00:00-03:59:59','04:00:00-07:59:59','08:00:00-11:59:59;', '12:00:00-15:59:59', '16:00:00-19:59:59', '20:00:00-23:59:59']
Sum_hour={h:0 for h in Hours}
Count_hours={h:0 for h in Hours}

for hour in range(28):
    if len(str(hour)) == 1:
        query = '0'+str(hour)
    else:
        query = str(hour)
    Sum_hour[Hours[hour%6]] +=\
    logFile.filter(logFile.value.contains(query+"/Jul/1995")).count()
    Count_hours[Hours[hour%6]] += 1
Request_hour = {h:(Sum_hour[h]/Count_hours[h]) for h in Hours}
print("Question 1A and 1B:")
print(Request_hour)

#Matplotlib Graph for Question 1A and 1B
Average = list(Request_hour.values())
plt.barh(Hours,Average)
plt.xlabel('Requests')
plt.ylabel('Hours')
plt.title('Average requests for each 4 hour slots')
plt.savefig('Q1_figA.png')

# Question 1C and 1D 

request_HTML = logFile.filter(logFile.value.contains(".html")).cache()
from pyspark.sql.functions import split, regexp_extract, count
split_df = request_HTML.select(regexp_extract('value', r'^.*"\w+\s+.*/([^\s]+\.html)\s+HTTP.*"', 1).alias('path'))
HTML_path = split_df.select('path')
request_HTML = (HTML_path.groupBy('path').agg(count('path').alias("_count")).sort('_count', ascending=False))
print("Question 1C and 1D:") 
request_HTML.show(20, False)
HTML_p = [row.path for row in request_HTML.collect()]
print(".html file names:")
print(HTML_p[:20])
HTML_c= [int(row._count) for row in request_HTML.collect()]
print("Number of requests made:")
print(HTML_c[:20])


#Matplotlib Graph for Question 1C and 1D
plt.subplots()
plt.barh(HTML_p[:20], HTML_c[:20])
plt.xlabel('Requests')
plt.ylabel('.html files')
plt.title('Top 20 most requested .html files')
plt.savefig('Q1_figB.png')


spark.stop()
