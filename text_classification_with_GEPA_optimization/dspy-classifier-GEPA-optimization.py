# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Optimization of a Text Classification using DSPy GEPA  (Genetic-Pareto)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package
# MAGIC

# COMMAND ----------

# MAGIC %pip install -qqqq -U dspy==3.0.1 mlflow==3.3.1 databricks-agents=1.3.0 
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Set up data
# MAGIC The following downloads the [pubmed text classification cased](https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv) dataset from Huggingface and writes a utility to ensure that your train and test split has the same labels.

# COMMAND ----------

import numpy as np
import pandas as pd
from dspy.datasets.dataset import Dataset
from pandas import StringDtype

def read_data_and_subset_to_categories() -> tuple[pd.DataFrame]:
    """
    Read the pubmed-text-classification-cased dataset. Docs can be found in the url below:
    https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv
    """

    # Read train/test split
    file_path = "https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv"
    train = pd.read_csv(file_path.format("train"))
    test = pd.read_csv(file_path.format("test"))

    train.drop('description_cln', axis=1, inplace=True)
    test.drop('description_cln', axis=1, inplace=True)

    return train, test


class CSVDataset(Dataset):
    def __init__(
        self, n_train_per_label: int = 40, n_test_per_label: int = 20, *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.n_train_per_label = n_train_per_label
        self.n_test_per_label = n_test_per_label

        self._create_train_test_split_and_ensure_labels()

    def _create_train_test_split_and_ensure_labels(self) -> None:
        """Perform a train/test split that ensure labels in `test` are also in `train`."""
        # Read the data
        train_df, test_df = read_data_and_subset_to_categories()

        train_df = train_df.astype(StringDtype())
        test_df = test_df.astype(StringDtype())

        # Sample for each label
        train_samples_df = pd.concat([
            group.sample(n=self.n_train_per_label, random_state=1) 
            for _, group in train_df.groupby('target')
        ])
        test_samples_df = pd.concat([
            group.sample(n=self.n_test_per_label, random_state=1) 
            for _, group in test_df.groupby('target')
        ])

        # Set DSPy class variables
        self._train = train_samples_df.to_dict(orient="records")
        self._test = test_samples_df.to_dict(orient="records")


# Sample a train/test split from the pubmed-text-classification-cased dataset
dataset = CSVDataset(n_train_per_label=3, n_test_per_label=10)

# Create train and test sets containing DSPy examples
train_dataset = [example.with_inputs("description") for example in dataset.train]
test_dataset = [example.with_inputs("description") for example in dataset.test]

print(f"train dataset size: \n {len(train_dataset)}")
print(f"test dataset size: \n {len(test_dataset)}")
print(f"Train labels: \n {set([example.target for example in dataset.train])}")
print(f"Sample entry: \n {train_dataset[0]}")

# COMMAND ----------

# MAGIC %md ## Set up DSPy signature and module

# COMMAND ----------

from typing import Literal
import mlflow
import dspy

# turning on autologging traces
mlflow.dspy.autolog(
    log_evals=True,
    log_compiles=True,
    log_traces_from_compile=True
)

# Create a signature for the DSPy module
class TextClassificationSignature(dspy.Signature):
    description: str = dspy.InputField()
    target: Literal[
        'CONCLUSIONS', 'RESULTS', 'METHODS', 'OBJECTIVE', 'BACKGROUND'
        ] = dspy.OutputField()


class TextClassifier(dspy.Module):
    """
    Classifies medical texts into a previously defined set of categories.
    """
    def __init__(self, lm_name: str):
        super().__init__()
        # Define the language model
        self.lm = dspy.LM(model=f"databricks/{lm_name}", max_tokens = 25000, cache=False, reasoning_effort="medium")
        # Define the prediction strategy
        self.generate_classification = dspy.Predict(TextClassificationSignature)

    def forward(self, description: str):
        """Returns the predcited category of the description text provided"""
        with dspy.context(lm=self.lm):
            return self.generate_classification(description=description)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run a Hello world example
# MAGIC
# MAGIC The following demonstrates predicting using the DSPy module and associated signature.

# COMMAND ----------


# Initilize our impact_improvement class
text_classifier = TextClassifier(lm_name="databricks-gpt-oss-20b")

print(
  text_classifier(description="This study is designed as a randomised controlled trial in which men living with HIV in Australia will be assigned to either an intervention group or usual care control group .")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluation Function  

# COMMAND ----------

import time
from databricks.agents.evals import judges

def validate_classification_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Uses Databricks AI judges to validate the prediction and return score (1.0 = corract, 0.0 = incorrect) plus feedback.
    """
    # Call correctness judge 
    judgement = judges.correctness(
        request=example.description,
        response=prediction.target,
        expected_response=example.target
    )
    # obtain score from judgement (1.0 = correct, 0.0 = incorrect)
    if judgement and judgement.value: 
        score = int(judgement.value.name == "YES")
    else:
        # if no judgement, fallback to comparing prediction to expected
        score = int(example.target == prediction.target)

    # obtain feedback from judgement
    if judgement and judgement.rationale:
        feedback = judgement.rationale
    else:
        # if no judgement, do not provide feedback  
        feedback = None
    return dspy.Prediction(score=score, feedback=feedback)

def check_accuracy(classifier, test_data: pd.DataFrame = test_dataset) -> float:
    """
    Checks the accuracy of the classifier on the test data.
    """
    scores = []
    for example in test_data:
        prediction = classifier(description=example["description"])
        score = validate_classification_with_feedback(example, prediction).score
        scores.append(score)
        
    return np.mean(scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting the small model accuracy (databricks-gpt-oss-20b)

# COMMAND ----------

small_lm_name = "databricks-gpt-oss-20b"
uncompiled_small_lm_accuracy = check_accuracy(TextClassifier(lm_name=small_lm_name))

displayHTML(f"<h1>Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting the large model accuracy (databricks-gpt-oss-120b)

# COMMAND ----------

lager_lm_name = "databricks-gpt-oss-120b"
uncompiled_large_lm_accuracy = check_accuracy(TextClassifier(lm_name=lager_lm_name))

displayHTML(f"<h1>Uncompiled {lager_lm_name} accuracy: {uncompiled_large_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md ## Optimization (using GEPA)

# COMMAND ----------

import uuid

# defining an UUID to identify the optimized module
id = str(uuid.uuid4())
print(f"id: {id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using GEPA with Claude-Sonnet-4 to evolve the instructions based on the AI Judge feedback

# COMMAND ----------

small_lm_name = "databricks-gpt-oss-20b"
reflection_lm_name = "databricks-claude-sonnet-4-5"

gepa = dspy.GEPA(
    metric=validate_classification_with_feedback,
    auto="light",
    reflection_minibatch_size=20,
    reflection_lm=dspy.LM(f"databricks/{reflection_lm_name}", max_tokens=25000),
    num_threads=16,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_{id}"):
    compiled_gepa = gepa.compile(
        TextClassifier(lm_name=small_lm_name),
        trainset=train_dataset, #reminder: Only passing 15 training sets! 
    )

compiled_gepa.save(f"compiled_gepa_{id}.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting prompt optimized small model accuracy (databricks-gpt-oss-20b)

# COMMAND ----------

text_classifier_gepa = TextClassifier(lm_name=small_lm_name)
text_classifier_gepa.load(f"compiled_gepa_{id}.json")

compiled_small_lm_accuracy = check_accuracy(text_classifier_gepa)
displayHTML(f"<h1>Compiled {small_lm_name} accuracy: {compiled_small_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Inspect Optimized Prompt

# COMMAND ----------

print(text_classifier_gepa.lm.history[-1]["messages"][0]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Recap
# MAGIC - Implemented a text classifier using DSPy
# MAGIC - Tested GPT OSS 20b and 120b
# MAGIC - Optimized the prompt for GPT OSS 20b with only 15 observations
# MAGIC - Optimized prompt with GTP OSS 20b outperforms un-optimized GTP OSS 120b
# MAGIC
# MAGIC ###Key points:
# MAGIC - GEPA optimizer leverages AI Judge feedback
# MAGIC - Experimentation is Key
# MAGIC - Rule of thumb: start with newer optimizers 

# COMMAND ----------


