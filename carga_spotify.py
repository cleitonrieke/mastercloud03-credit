# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport

# COMMAND ----------

df = spark.read.csv("file:/Workspace/Repos/cleitonrieke@gmail.com/mastercloud03-credit/spotify-2023.csv", header=True)

# COMMAND ----------

df.display()

# COMMAND ----------

df = df.withColumn('streams', F.col('streams').cast('long'))
df = df.withColumn('danceability_%', F.col('danceability_%').cast('int'))
df = df.withColumn('valence_%', F.col('valence_%').cast('int'))
df = df.withColumn('energy_%', F.col('energy_%').cast('int'))
df = df.withColumn('acousticness_%', F.col('acousticness_%').cast('int'))
df = df.withColumn('intrumentalness_%', F.col('instrumentalness_%').cast('int'))
df = df.withColumn('liveness_%', F.col('liveness_%').cast('int'))
df = df.withColumn('speechiness_%', F.col('speechiness_%').cast('int'))
df = df.withColumn('bpm', F.col('bpm').cast('int'))
df = df.withColumn('mais_tocada', (F.when((F.col('streams') > 514137424), 1).otherwise(0)).cast('int'))

# COMMAND ----------

df_pd = df.toPandas()
sns.heatmap(df_pd.corr())

# COMMAND ----------

#df_pd["streams"].max()
df_pd_maiores = df_pd[df_pd['mais_tocada'] == 1]

# COMMAND ----------

df_pd_maiores.display()

# COMMAND ----------

df_pd['streams'].min()

# COMMAND ----------

df_pd.corr(method='spearman')

# COMMAND ----------

profile = ProfileReport(df_pd)

# COMMAND ----------

report_html = profile.to_html()
displayHTML (report_html)

# COMMAND ----------

df = df.withColumnRenamed("artist(s)_name", "artist_name")
df = df.withColumnRenamed("danceability_%", "danceability")
df = df.withColumnRenamed("valence_%", "valence")
df = df.withColumnRenamed("energy_%", "energy")
df = df.withColumnRenamed("acousticness_%", "acousticness")
df = df.withColumnRenamed("instrumentalness_%", "instrumentalness")
df = df.withColumnRenamed("liveness_%", "liveness")
df = df.withColumnRenamed("speechiness_%", "speechiness")
df = df.withColumnRenamed("intrumentalness_%", "intrumentalness")
df.display()

# COMMAND ----------

df.write.saveAsTable("hive_metastore.default.spotify_gold")
