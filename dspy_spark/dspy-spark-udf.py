# Databricks notebook source
# MAGIC %md
# MAGIC #DSPY/Spark UDF integration 
# MAGIC - Instal dependencies
# MAGIC - Set up env vars with PAT (api_key) and HOST(api_base) values
# MAGIC - Defines and runs (locally) a simple DSPy module
# MAGIC - Defines Spark DataFrame with dummy data (two records)
# MAGIC
# MAGIC ###Approach 1:
# MAGIC - Deactivates DSPy file caching and defines/runs a UDF that calls the simple DSPy module
# MAGIC
# MAGIC ###Approach 2:
# MAGIC - Activate DSPy file caching, set caching path to a DBX files directory, and defines/runs a UDF that calls the simple DSPy module
# MAGIC
# MAGIC ####NOTES: 
# MAGIC - Notebooks has been tested using serverles compute
# MAGIC - Setting the caching path to a volume is not supported

# COMMAND ----------

# MAGIC %md
# MAGIC ###Installing dependencies

# COMMAND ----------

# MAGIC %pip install -U -qqqq dspy>=3.0.4b1 mlflow
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("secret_scope", "")
dbutils.widgets.text("pat_secret_name", "")

# COMMAND ----------

import os

pat_secret_name = dbutils.widgets.get("pat_secret_name")
secret_scope = dbutils.widgets.get("secret_scope")

os.environ["PAT"] = dbutils.secrets.get(scope=secret_scope, key=pat_secret_name)
os.environ["HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ###DSPy module

# COMMAND ----------

import dspy
import mlflow
import os


class AnswerQuestion(dspy.Module):
    def __init__(self):

        mlflow.dspy.autolog()

        super().__init__()

        self.lm = dspy.LM(
            model="databricks/databricks-llama-4-maverick",
            api_key=os.environ["PAT"],
            api_base=f'https://{os.environ["HOST"]}/serving-endpoints', 
        )
        self.qa = dspy.Predict("question: str -> response: str")

    def forward(self, question: str) -> str:
        question = (
            question[-1]["content"] if not isinstance(question, str) else question
        )
        with dspy.context(lm=self.lm):
            answer = self.qa(question=question).response
        return answer


respond = AnswerQuestion()

respond("what are high memory and low memory on linux?")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Setup dummy Spark Dataframe

# COMMAND ----------

df = spark.createDataFrame( [{"question": """is applecare for iphone transferable?""", "response": """Yes, according to AppleCare's Frequently Asked Questions, both the AppleCare Protection Plan and AppleCare+ are transferable. \nFurthermore, when an individual obtains an Apple product with AppleCare, the previous owner is obliged to transfer the ownership of the AppleCare Protection Plan or AppleCare+ to the new owner if they choose to sell or give the product away.  \nThe option to transfer ownership is provided because the original owner has the ability to cancel the AppleCare protection at any time and may receive a prorated refund."""}, 
{"question": """find the total size of certain files within a directory branch""", "response": """In computing disk usage for JPEG files, you can use various commands such as \"du -ch public_html/images/*.jpg\" to check the total usage within a directory and the command \"find ./photos/john_doe -type f -name '*.jpg' -exec du -ch {} +\" which may produce multiple totals if the file list is extensive.  \nFor an efficient solution that works on different systems including Mac OS X, and gives you the total answer in gigabytes, you can implement \"find . -iname \"*.jpg\" -ls | perl -lane '$t += $F; print $t/1024/1024/1024\" GB.\" \nAdditionally, to work with any locale, you could use \"LC_ALL=C find ./photos/john_doe -type f -name '*.jpg' -exec du -ch {} +\" or \"find -type f -iname *.jpg -print0 | du -ch --files0-from=-\" are commendable commands."""}])

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Running UDF

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf
from dspy import configure_cache

configure_cache(enable_disk_cache=False)

@pandas_udf("string")
def answer_question(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for questions in batch_iter:
        answers = pd.Series([respond(question) for question in questions])
        yield answers


df = (
    df
    .select("question")
    .limit(2)
    .withColumn("answer", answer_question(F.col("question")))
)
display(df)

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf

dspy.configure_cache(disk_cache_dir="./dspy_cache", enable_disk_cache=True)

@pandas_udf("string")
def answer_question(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for questions in batch_iter:
        answers = pd.Series([respond(question) for question in questions])
        yield answers


df = (
    df
    .select("question")
    .limit(2)
    .withColumn("answer", answer_question(F.col("question")))
)
display(df)
