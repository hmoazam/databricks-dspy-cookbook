# Databricks notebook source
# MAGIC %md
# MAGIC #Advanced DSPy Integrations
# MAGIC
# MAGIC In this section (and mainly because it does not nicely fit anywhere), you will learn about the following:
# MAGIC
# MAGIC 1. DSPy optimizers
# MAGIC 2. DSPy's MCP Integration 
# MAGIC 3. DSPy with Databricks AI Bridge  
# MAGIC 4. DSPy with ai_query
# MAGIC
# MAGIC Use each of these capabilities as necessary for your use case. These are not required to successfully build with DSPy but help when building on Databricks 

# COMMAND ----------

# MAGIC %md
# MAGIC #DSPy Prompt Optimizers 
# MAGIC
# MAGIC Iterating through prompts manually is tedious. Without an automated and grounded/objective way of iterating development of prompts, it becomes nearly impossible to maintain prompts over time, especially post production. 
# MAGIC
# MAGIC **Automated improvement** - Instead of manually tweaking prompts through trial-and-error, DSPy systematically optimizes them based on your metrics and training examples. This can save significant time and often discovers better prompts than manual engineering.
# MAGIC
# MAGIC **Data-driven optimization** - The optimizers learn from your specific examples and use cases, tailoring prompts to your actual needs rather than generic best practices.
# MAGIC Complex pipeline optimization - When you have multi-step LLM workflows (retrieval → reasoning → generation), DSPy can optimize the entire pipeline together, which is much harder to do manually.
# MAGIC
# MAGIC **Reproducible and systematic** - Unlike ad-hoc prompt engineering, DSPy provides a programmatic, repeatable process for improving your LLM applications.
# MAGIC
# MAGIC **Handling prompt brittleness** - Optimizers can find more robust prompts that work across different examples, reducing the brittleness common with hand-crafted prompts. 
# MAGIC
# MAGIC DSPy optimizers are particularly useful when you:
# MAGIC
# MAGIC 1. Have clear metrics and evaluation data
# MAGIC 2. Need to optimize complex, multi-step LLM pipelines
# MAGIC 3. Want to adapt prompts for different models (DSPy can re-optimize when you switch models)
# MAGIC 4. Have spent significant time manually tweaking prompts without great results
# MAGIC 5. Need consistent performance across diverse inputs
# MAGIC
# MAGIC ##Cost Value of Prompt Optimizers 
# MAGIC
# MAGIC The Databricks Mosaic AI Research Team released a blog post highlighting how they achieve 90x cost savings by using GEPA, a prompt optimizer on their AI workflows. It highlights how we can find significant performance gains just from optimizing prompts on smaller LLMs. Check out the blog here: https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization
# MAGIC
# MAGIC If costs are stopping you from going to production, it is essentially mandatory to do prompt optimization so that you are enabled to use smaller LLMs. 
# MAGIC
# MAGIC For example, below is a cost breakdown of using GPT-OSS 20B vs Claude Sonnet 4.5

# COMMAND ----------

# MAGIC %md
# MAGIC #Prompt Optimization Demo: GEPA 
# MAGIC
# MAGIC We will use the same optimizer using in the Mosaic AI research blog post to highlight how powerful optimizers are. In this section, you will optimize a GPT OSS 20B model with a Claude Sonnet 4.5 model as the teacher LLM. We will compare the 20B model to the 120B and Claude to see how well 20B does pre and post optimization

# COMMAND ----------

# MAGIC %pip install --upgrade dspy mlflow databricks-agents
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

# MAGIC %md
# MAGIC #Set up the DSPy module and signature for testing 

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
# MAGIC #Let's test that it works

# COMMAND ----------


# Initilize our impact_improvement class
text_classifier = TextClassifier(lm_name="databricks-gpt-oss-20b")

print(
  text_classifier(description="This study is designed as a randomised controlled trial in which men living with HIV in Australia will be assigned to either an intervention group or usual care control group .")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Make an Evaluation Function
# MAGIC
# MAGIC Now we need an evaluation function to ensure that we provide correct feedback to guide the models in the right direction. GEPA accepts numeric and text feedback which allows us to integrate AI Judges. AI judges enable us to dynamically react to the performance of the smaller language model and provide more direct, relevant feedback, especially when the AI judge is grounded in our data 

# COMMAND ----------

import time
from databricks.agents.evals import judges

def validate_classification_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Uses Dtabricks AI judges to validate the prediction and return score (1.0 = corract, 0.0 = incorrect) plus feedback.
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
# MAGIC #Let's test GPT-OSS 20B with this Evaluation Function

# COMMAND ----------

small_lm_name = "databricks-gpt-oss-20b"
uncompiled_small_lm_accuracy = check_accuracy(TextClassifier(lm_name=small_lm_name))

displayHTML(f"<h1>Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC #Now let's test GPT-OSS 120B

# COMMAND ----------

lager_lm_name = "databricks-gpt-oss-120b"
uncompiled_large_lm_accuracy = check_accuracy(TextClassifier(lm_name=lager_lm_name))

displayHTML(f"<h1>Uncompiled {lager_lm_name} accuracy: {uncompiled_large_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC #Now Claude's turn!

# COMMAND ----------

lager_lm_name = "databricks-claude-sonnet-4"
uncompiled_large_lm_accuracy = check_accuracy(TextClassifier(lm_name=lager_lm_name))

displayHTML(f"<h1>Uncompiled {lager_lm_name} accuracy: {uncompiled_large_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC ### We can see that 20B just cannot compete with the bigger, frontier models. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Time to run GEPA
# MAGIC
# MAGIC Now we have our baseline with the larger, frontier models. We can optimize GPT-OSS 20B to see how well it does compared to the larger, frontier models. 
# MAGIC
# MAGIC If you need to read more about GEPA, check out the resources here: 
# MAGIC 1. GEPA Paper: https://arxiv.org/pdf/2507.19457 
# MAGIC 2. DSPy GEPA Tutorials: https://dspy.ai/api/optimizers/GEPA/overview/ 

# COMMAND ----------

import uuid

# defining an UUID to identify the optimized module
id = str(uuid.uuid4())
print(f"id: {id}")

# COMMAND ----------

small_lm_name = "databricks-gpt-oss-20b"
reflection_lm_name = "databricks-claude-sonnet-4"

gepa = dspy.GEPA(
    metric=validate_classification_with_feedback,
    auto="light",
    reflection_minibatch_size=15,
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
# MAGIC #Let's try it again
# MAGIC
# MAGIC You can see the optimized prompt is saved as a json. We can load this json and use this with a model. Let's try this again with GPT-OSS 20B 

# COMMAND ----------

text_classifier_gepa = TextClassifier(lm_name=small_lm_name)
text_classifier_gepa.load(f"compiled_gepa_{id}.json")

compiled_small_lm_accuracy = check_accuracy(text_classifier_gepa)
displayHTML(f"<h1>Compiled {small_lm_name} accuracy: {compiled_small_lm_accuracy}</h1>")

# COMMAND ----------

# MAGIC %md
# MAGIC #Look at that score! 
# MAGIC
# MAGIC We managed to improve GPT-OSS 20B's performance by 12 points, beating GPT-OSS 120B. 
# MAGIC
# MAGIC If this was your use case, you may be more comfortable deploying GPT-OSS 20B instead of Claude Sonnet, which is about 60x to 75x to use than Claude Sonnet. 
# MAGIC
# MAGIC Now you have some options. 
# MAGIC 1. You use a model that is 60x to 75x cheaper and faster in latency than Claude 4 Sonnet at the cost of 6 points if that's acceptable. 
# MAGIC 2. You use a model that is 20x to 22x cheaper, faster in latency AND BEATS GPT-OSS 120B. This may make hosting the model more achievable 
# MAGIC
# MAGIC Ideally in a production use case, you will want to host your model. Now you have more wiggle room in doing so! 

# COMMAND ----------

# MAGIC %md
# MAGIC #You can inspect the prompt below! 
# MAGIC
# MAGIC It's not a significant change from what we started with but we now have an automated way to find huge gains in performance!

# COMMAND ----------

print(text_classifier_gepa.lm.history[-1]["messages"][0]["content"])

# COMMAND ----------

