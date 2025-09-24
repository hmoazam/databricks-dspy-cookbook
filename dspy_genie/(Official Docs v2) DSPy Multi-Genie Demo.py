# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy a multi-agent system with Genie and DSPy
# MAGIC This notebook demonstrates how to build a multi-agent system using Mosaic AI Agent Framework and [DSPy](https://dspy.ai/), where [Genie](https://www.databricks.com/product/business-intelligence/ai-bi-genie) is one of the agents. In this notebook, you:
# MAGIC
# MAGIC 1. Author a multi-agent system using DSPy.
# MAGIC 2. Wrap the DSPy agent with MLflow ChatAgent to ensure compatibility with Databricks features.
# MAGIC 3. Manually test the multi-agent system's output.
# MAGIC 4. Log and deploy the multi-agent system.
# MAGIC
# MAGIC ## Why use a Genie agent?
# MAGIC Multi-agent systems consist of multiple AI agents working together, each with specialized capabilities. As one of those agents, Genie allows users to interact with their structured data using natural language.
# MAGIC
# MAGIC Unlike SQL functions which can only run pre-defined queries, Genie has the flexibility to create novel queries to answer user questions.
# MAGIC
# MAGIC ##Prerequisites 
# MAGIC
# MAGIC 1. Create a Genie Space following the instruction in this [notebook]($./setup_genie) (See Databricks documentation for more details ([AWS](https://docs.databricks.com/aws/en/genie/set-up) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/genie/set-up))).
# MAGIC 2. Address all TODOs in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies

# COMMAND ----------

# MAGIC %pip install -qqqq --upgrade dspy mlflow databricks-sdk databricks-agents pydantic uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # DSPy Basics

# COMMAND ----------

# MAGIC %md
# MAGIC We will use DSPy and create python functions that our agent will use as tools. 
# MAGIC
# MAGIC All we need is the following: 
# MAGIC 1. An LLM that we can configure via dspy.configure
# MAGIC 2. A dspy.Signature, that defines how the LLM should accomplish the task. A signature structures your inputs and outputs, enforces typing and provides additional instructions. 
# MAGIC 3. A dspy.Module to convert the signature into a prompt. Depending on the module, you can enable agents (dspy.ReAct), chain of thought or reasoning (dspy.CoT) or just a simple predict (dspy.Predict). The module's output will be a prediction where we can programatically access the outputs

# COMMAND ----------

# MAGIC %md
# MAGIC #Let's deploy this agent using the Databricks AI Agent Framework

# COMMAND ----------

# MAGIC %md
# MAGIC We start by setting up mlflow experiment path

# COMMAND ----------

import mlflow

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

mlflow_experiment_path = f"/Users/{current_user}/dspy_multi_genie"
mlflow.set_experiment(experiment_name=mlflow_experiment_path)

# Get the experiment ID to use in the next step
experiment_id = mlflow.tracking.fluent._get_experiment_id()
print(experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO: Assign a PAT Token to DATABRICKS_TOKEN. 
# MAGIC
# MAGIC Follow these [instructions](https://docs.databricks.com/aws/en/dev-tools/auth/pat#databricks-personal-access-tokens-for-workspace-users) to generate a PAT Token.

# COMMAND ----------

import os

os.environ["DATABRICKS_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = 'SET_TOKEN_HERE'

# COMMAND ----------

# MAGIC %md
# MAGIC We will now use `%%writefile` to create a python file containing our agents code.

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC
# MAGIC import mlflow
# MAGIC from typing import Any, Generator, Optional
# MAGIC from databricks.sdk.service.dashboards import GenieAPI
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc.model import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC import dspy
# MAGIC import uuid
# MAGIC import datetime
# MAGIC import os
# MAGIC
# MAGIC # Autolog DSPy traces to MLflow
# MAGIC mlflow.dspy.autolog()
# MAGIC
# MAGIC # Set up DSPy with a Databricks-hosted LLM
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
# MAGIC dspy.settings.configure(lm=lm)
# MAGIC
# MAGIC
# MAGIC ######################################
# MAGIC ## Create our Genie Selector Signature
# MAGIC ######################################
# MAGIC # This signature will be used to determine which Genie Space based on the request and utilize the relevant Genie Space using self-reasoning to select the proper tool.
# MAGIC
# MAGIC # Deciding on your module’s inputs, outputs, and their types is the most critical part of creating a signature. This definition forms the contract that other components in your multi-agent pipeline will rely on.
# MAGIC
# MAGIC class genie_selector_agent(dspy.Signature):
# MAGIC   """
# MAGIC   Given a question, determine which genie space tool to call, send the exact question to the tool and answer the question given the response from the tool.
# MAGIC   """ 
# MAGIC   # **NOTE** : You can add details to the prompt about each tool to increase the accuracy of the agent. Both the function name and prompt inform the agent what the tool is used for.
# MAGIC
# MAGIC   question: str = dspy.InputField()
# MAGIC   response: str = dspy.OutputField() 
# MAGIC   sql_query_output:  list = dspy.OutputField()
# MAGIC
# MAGIC
# MAGIC #####################################
# MAGIC ## Create our DSPY Chat Agent Class
# MAGIC #####################################
# MAGIC class DSPyChatAgent(ChatAgent):     
# MAGIC     def __init__(self):
# MAGIC       self.w = WorkspaceClient(
# MAGIC         host=os.getenv("DATABRICKS_HOST"),
# MAGIC         token=os.getenv("DATABRICKS_TOKEN")
# MAGIC       )
# MAGIC
# MAGIC       self.genie_selector_agent = genie_selector_agent
# MAGIC
# MAGIC       self.multi_genie_agent = dspy.ReAct(self.genie_selector_agent, 
# MAGIC                                           tools=[self.australian_sales_space, self.us_sales_space], 
# MAGIC                                           max_iters=1)
# MAGIC
# MAGIC
# MAGIC
# MAGIC     #######################################################
# MAGIC     ## Create a Function that will call the 1st Genie Space
# MAGIC     #######################################################
# MAGIC
# MAGIC     def australian_sales_space(self, question):
# MAGIC       """This genie space is used to query data about Australia Sales. This genie space takes in a request and returns the relevant data."""
# MAGIC
# MAGIC       # TODO add GENIE_SPACE_ID and a description for this space
# MAGIC       # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
# MAGIC       genie_space_id = "01f063657dfe1464aa0eec353df246f5"
# MAGIC
# MAGIC
# MAGIC       # Start a conversation
# MAGIC       conversation = self.w.genie.start_conversation_and_wait(
# MAGIC           space_id=genie_space_id,
# MAGIC           content=f"{question} always limit to one result",
# MAGIC           timeout=datetime.timedelta(minutes=20)
# MAGIC       )
# MAGIC
# MAGIC       response = self.w.genie.get_message_attachment_query_result(
# MAGIC         space_id=genie_space_id,
# MAGIC         conversation_id=conversation.conversation_id,
# MAGIC         message_id=conversation.message_id,
# MAGIC         attachment_id=conversation.attachments[0].attachment_id
# MAGIC       )
# MAGIC
# MAGIC       return response.statement_response.result.data_array
# MAGIC     
# MAGIC     #######################################################
# MAGIC     ## Create a Function that will call the 2nd Genie Space
# MAGIC     #######################################################
# MAGIC     
# MAGIC     def us_sales_space(self, question):
# MAGIC       """This genie space is used to query data about USA Retail Sales. This genie space takes in a request and returns the relevant data."""
# MAGIC
# MAGIC       # TODO add GENIE_SPACE_ID and a description for this space
# MAGIC       # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
# MAGIC       genie_space_id = "01f06364428a132b8de538b325b2c24f"
# MAGIC
# MAGIC
# MAGIC       # Start a conversation
# MAGIC       conversation = self.w.genie.start_conversation_and_wait(
# MAGIC           space_id=genie_space_id,
# MAGIC           content=f"{question} always limit to one result",
# MAGIC           timeout=datetime.timedelta(minutes=20)
# MAGIC       )
# MAGIC
# MAGIC       response = self.w.genie.get_message_attachment_query_result(
# MAGIC         space_id=genie_space_id,
# MAGIC         conversation_id=conversation.conversation_id,
# MAGIC         message_id=conversation.message_id,
# MAGIC         attachment_id=conversation.attachments[0].attachment_id
# MAGIC       )
# MAGIC
# MAGIC       return response.statement_response.result.data_array
# MAGIC
# MAGIC
# MAGIC     def prepare_message_history(self, messages: list[ChatAgentMessage]):
# MAGIC         history_entries = []
# MAGIC         # Assume the last message in the input is the most recent user question.
# MAGIC         for i in range(0, len(messages) - 1, 2):
# MAGIC             history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
# MAGIC         return dspy.History(messages=history_entries)
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         latest_question = messages[-1].content
# MAGIC         response = self.multi_genie_agent(question=latest_question).response
# MAGIC         return ChatAgentResponse(
# MAGIC             messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
# MAGIC         )
# MAGIC
# MAGIC # Set model for logging or interactive testing
# MAGIC from mlflow.models import set_model
# MAGIC AGENT = DSPyChatAgent()
# MAGIC set_model(AGENT)

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC Interact with the agent to test its output. Since this notebook called mlflow.dspy.autolog() you can view the trace for each step the agent takes.

# COMMAND ----------

from agent import AGENT

input_example= "Who is the top buyer in Australia?"

response = AGENT.predict({"messages": [{"role": "user", "content": input_example}]})

print(response.messages[0].content)

# COMMAND ----------

from agent import AGENT

input_example= "Who is the top buyer in USA?"

response  = AGENT.predict({"messages": [{"role": "user", "content": input_example}]})

print(response.messages[0].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC Log the agent as code from the `agent.py` file that was written by the previous cell. See [MLflow - Models from Code](https://mlflow.org/docs/latest/ml/model/#models-from-code).
# MAGIC
# MAGIC ## Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling mlflow.pyfunc.log_model().
# MAGIC
# MAGIC - TODO: Under the `resources` variable, add the correct genie_space_ids.

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
)
from pkg_resources import get_distribution


# TODO : set the genie_space_id for each Genie Space you want to call
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id= "01f06364428a132b8de538b325b2c24f"),
    DatabricksGenieSpace(genie_space_id= "01f063657dfe1464aa0eec353df246f5"),
    DatabricksSQLWarehouse(warehouse_id="30d6e63b35f828c5"),
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        # input_example=input_example,
        extra_pip_requirements=[
            # f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"databricks-sdk=={get_distribution('databricks-sdk').version}",
        ],
        # resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-debug#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": input_example}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering the Model
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.
# MAGIC
# MAGIC

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model.
catalog = "users"
schema = "luis_moros"
model_name = "dspy_multi_genie"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

from databricks import agents
import os

agents.deploy(
  UC_MODEL_NAME, 
  uc_registered_model_info.version, 
  tags={"endpointSource": "docs"},
  environment_vars={
    "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"), 
    "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN")
  },  
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM users.luis_moros.dspy_multi_genie_payload order by request_date desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from system.compute.warehouses where warehouse_name = "1111-default-wh"
