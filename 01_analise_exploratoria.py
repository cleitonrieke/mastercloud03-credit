# Databricks notebook source
# DBTITLE 1,Carrega

import pandas as pd
df = spark.read.csv("file:/Workspace/Repos/cleitonrieke@gmail.com/Mastercloud-Trilha-03/ML/credito/data/treino.csv", header=True)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df = df.withColumn('idade', F.col('idade').cast('int'))
df = df.withColumn('salario_mensal', F.col('salario_mensal').cast('float'))
df = df.withColumn('razao_debito', F.col('razao_debito').cast('float'))
df = df.withColumn('inadimplente', F.col('inadimplente').cast('int'))
df = df.withColumn('salario_mensal', F.col('salario_mensal').cast('float'))
df = df.withColumn('numero_emprestimos_imobiliarios', F.col('numero_emprestimos_imobiliarios').cast('int'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))
df = df.withColumn('vezes_passou_de_30_59_dias', F.col('vezes_passou_de_30_59_dias').cast('float'))
df = df.withColumn('numero_linhas_crdto_aberto', F.col('numero_linhas_crdto_aberto').cast('int'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))

# COMMAND ----------

display(df)

# COMMAND ----------

df.show()

# COMMAND ----------

df2 = df.filter( (F.col('idade') > 20) & (F.col('idade') < 80) )
df2 = df.filter( (F.col('salario_mensal') > 0) & (F.col('salario_mensal') < 20000) )

# COMMAND ----------

df2.show()

# COMMAND ----------

display(df2)

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

df_pd.corr()

# COMMAND ----------

df2 = df2.withColumn("atraso",
    F.when(df['vezes_passou_de_30_59_dias'] + df['numero_de_vezes_que_passou_60_89_dias'] + df['numero_vezes_passou_90_dias'] > 0, 0)
    .otherwise(1)
)


# COMMAND ----------

df2.groupBy("atraso").agg(F.mean("inadimplente")).display()

# COMMAND ----------

df2.groupBy("numero_emprestimos_imobiliarios", "atraso").agg(F.mean("inadimplente")).display()

# COMMAND ----------

df_pd.to_parquet("file:/Workspace/Repos/cleitonrieke@gmail.com/Mastercloud-Trilha-03/ML/credito/data/treino.csv")
