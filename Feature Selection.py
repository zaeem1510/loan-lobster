# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %md
# MAGIC ##Import Libraries

# COMMAND ----------

from pyspark.ml.feature import *
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import Window
from pyspark.sql.types import *

import random

import joblib

import pandas as pd
import numpy as np
from scipy import stats
import pyspark.pandas as ps

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import random

from collections import Counter

from datetime import date

from sklearn import metrics

from xgboost.spark import SparkXGBClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ###Feature Selection

# COMMAND ----------

df1 = spark.read.table('BFL_STD_LAKE.SME_COE.dj_plcs_ml_base_reduced_all_vars').filter(col('plcs_flag')==1).sample(0.1)
df2 = spark.read.table('BFL_STD_LAKE.SME_COE.dj_plcs_ml_base_reduced_all_vars').filter(col('plcs_flag')==0).sample(0.03)

# COMMAND ----------

df = df1.union(df2)

# COMMAND ----------

# MAGIC %sql
# MAGIC select plcs_flag,count(1) from BFL_STD_LAKE.SME_COE.dj_plcs_ml_base_reduced_all_vars
# MAGIC group by all

# COMMAND ----------

# MAGIC %sql
# MAGIC describe BFL_STD_LAKE.SME_COE.dj_plcs_ml_base_reduced_all_vars

# COMMAND ----------

df = df.drop(*('id,CUSTOMER_ID','BUREAU_HIT_FLAG','BUREAUCUSTOMERID','BUREAUPRODKEY','REPORTINGDATETIME','INDVIDUAL_ID_CORP_CUSTOMER','EVER_PL_FLAG','MAX_DATE','L2Y_PL_FLAG','L2Y_DATE','L2Y_PL_SANCTION_AMT','original_amt_fin','disbursement_date','LOAN_STATUS','APPL_ID','CNT_BUSINESS','BUSINESS','BRANCH_DESC','channel_flag','final_date','etb_earliest_product_group','etb_earliest_business_type','etb_latest_product_group','etb_latest_business_type','INDIV_CORP_FLAG','CARDED_FLAG','ONLY_FD_FLAG','ONLY_MF_FLAG','CIF_FLAG','PROF_RED_FLAG','REPUTATION_FLAG','LENDING_FLAG','FOS_FLAG','UPI_FLAG','PPI_FLAG','BBPS_FLAG','DIGITAL_ACTIVE_FLAG','AF_LOGIN_BASE','ONLY_AF','rnk3in1'))

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.fillna('OTHERS', subset=['tier','PGE'])

df = df.fillna(0)

# COMMAND ----------

inputs = ['PGE','tier']
outputs = ['PGE_cat','tier_cat']

stringIndexer = StringIndexer(inputCols=inputs, outputCols=outputs)

df = stringIndexer.fit(df).transform(df)

df = df.drop(*('PGE','tier'))

# COMMAND ----------

inputs = ['PGE_cat','tier_cat']
outputs = ['PGE_ohe','tier_ohe']

ohencoder = OneHotEncoder(inputCols=inputs, outputCols=outputs, dropLast=False)

df = ohencoder.fit(df).transform(df)

df = df.drop(*('PGE_cat','tier_cat'))v

# COMMAND ----------

featuresCols = df.columns
featuresCols.remove('plcs_flag')

vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")

# COMMAND ----------

df_final = vectorAssembler.transform(df)

# COMMAND ----------

df_final = df_final.select('features','plcs_flag')

# COMMAND ----------

xgb_classifier = SparkXGBClassifier(num_workers=4, label_col="plcs_flag")
xgb_classifier_model = xgb_classifier.fit(df_final)

# COMMAND ----------

display(pd.DataFrame.from_dict(xgb_classifier_model.get_booster().get_score(importance_type='weight'),orient='index').reset_index())

# COMMAND ----------

display(pd.DataFrame(df.drop('plcs_flag').columns))

# COMMAND ----------

display(pd.DataFrame.from_dict(xgb_classifier_model.get_booster().get_score(importance_type='gain'),orient='index').reset_index())