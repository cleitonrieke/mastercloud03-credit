# Databricks notebook source
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pyspark.sql import functions as F
import datetime

# COMMAND ----------

spark_df = spark.read.table("hive_metastore.default.spotify_gold")
spark_df = spark_df.withColumn('data', F.concat(F.col('released_year'), F.lit('-'), F.col('released_month'), F.lit('-'), F.col('released_day')))
df = spark_df.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

y = df["mais_tocada"]
X = df.copy().drop(columns={"mais_tocada", "track_name", "artist_name", "artist_count", "released_year", "released_month", "released_day", "in_spotify_playlists", "key", "mode", "in_spotify_charts", "in_apple_playlists", "in_deezer_playlists", "in_shazam_charts", "in_apple_charts", "in_deezer_charts"})

# COMMAND ----------

display(X)

# COMMAND ----------

display(X)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# COMMAND ----------

display(X_train), display(X_test)

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

y.value_counts(normalize=True)

# COMMAND ----------

clf = lgb.LGBMClassifier(objective='binary', class_weight='balanced')

# COMMAND ----------

clf

# COMMAND ----------

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Add more preprocessing steps as needed
    ('classifier', clf)
])

# COMMAND ----------

pipeline

# COMMAND ----------

params = {
    "objective": "binary",
    "class_weight": "balanced",
}
mlflow.log_params(params)

# COMMAND ----------

pipeline.fit(X_train, y_train)

# COMMAND ----------

mlflow.lightgbm.log_model(clf, "LightGBM_Model")

# COMMAND ----------

display(X_test.iloc[1])

# COMMAND ----------

y_test.iloc[1]

# COMMAND ----------

pipeline.predict(X_test)[1]

# COMMAND ----------

accuracy_score(y_test, pipeline.predict(X_test))

# COMMAND ----------

pipeline.predict_proba(X_test)[:, 1]

# COMMAND ----------

y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Inadimplentes reais')
plt.title('Matriz de confusão')
plt.show()

# COMMAND ----------

sns.scatterplot(data = df, x = 'valence', y = 'danceability', hue = 'streams')

# COMMAND ----------

X2 = df.copy().drop(columns={"mais_tocada", "track_name", "artist_name", "artist_count", "released_year", "released_month", "released_day", "in_spotify_playlists", "key", "mode", "in_spotify_charts", "in_apple_playlists", "in_deezer_playlists", "in_shazam_charts", "in_apple_charts", "in_deezer_charts", "streams"})

# COMMAND ----------

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, df[["streams"]], test_size=0.20, random_state=42)

# COMMAND ----------

X2_train_norm = preprocessing.normalize(X2_train)
X2_test_norm = preprocessing.normalize(X2_test)

# COMMAND ----------

kmeans = KMeans(n_clusters=3)
kmeans.fit(X2)

# COMMAND ----------

X2['cluster'] = kmeans.labels_

# COMMAND ----------

X2

# COMMAND ----------

sns.scatterplot(data = X2, x = 'bpm', y = 'energy', hue = 'cluster')
