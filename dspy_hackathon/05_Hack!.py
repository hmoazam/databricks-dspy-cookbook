# Databricks notebook source
# MAGIC %md
# MAGIC #Objective: 
# MAGIC Create an Agent that can process unstructured data to their specifications
# MAGIC
# MAGIC Broad Ideas: 
# MAGIC
# MAGIC 1. Process a PDF and convert to structured Data
# MAGIC 2. Create Vector Search Index off of it 
# MAGIC 3. Use Genie Spaces to link data 
# MAGIC 4. Identify necessary information 
# MAGIC 5. Define a function to create necessary visuals or take necessary actions
# MAGIC
# MAGIC You agent should do the following: 
# MAGIC 1. Be able to hand off to another Agent 
# MAGIC 2. Have access to multiple tools 
# MAGIC 3. Demonstrate switching between LLMs 
# MAGIC 4. Use a combination of Python Code and calls to the Agent to improve your answer 
# MAGIC
# MAGIC UIs are not necessary but highly encouraged

# COMMAND ----------

# MAGIC %md
# MAGIC #Don't have an idea? 
# MAGIC
# MAGIC Here's one: 
# MAGIC
# MAGIC Goal: Make an agent that uses a genie space to query stock data when dates on when the stock price dropped. You want the agent to use this structured data to query an external web search tool (hosted as a UC function) based on this information. 
# MAGIC
# MAGIC If possible, try to spin up a vector search index. 
# MAGIC
# MAGIC Some data is provided for you below to use in a genie space. 
# MAGIC
# MAGIC There is also an example python function to call a UC function

# COMMAND ----------

# MAGIC %pip install --upgrade dspy openai litellm "mlflow[databricks]>=3.1.0" "databricks-connect>=16.1" unitycatalog-ai[databricks] databricks-sdk databricks-vectorsearch databricks-agents databricks-dspy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

df = pd.read_csv("./financial.csv")
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").option("delta.columnMapping.mode", "name").saveAsTable('you_delta_table here')

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def web_search(query):
  """this tool is to query a function to query the web"""
  
  client = DatabricksFunctionClient(execution_mode="local")
  result = client.execute_function(
    "catalog.schema.your_uc_function",
    parameters={"query": query}
  )
  return result.value

# COMMAND ----------

# MAGIC %md
# MAGIC #Recommendations for ETL 
# MAGIC 1. Use ai_parse_document for PDF
# MAGIC Use Databricks' ai_parse_document function to automatically extract text, tables, and structured data from your PDF files. This AI-powered tool understands document layout and converts PDFs into clean, structured JSON format. It works much better than basic text extraction for complex documents.
# MAGIC 2. Store Information into Delta Tables. We need existing Delta Tables to write back to existing tables. 
# MAGIC Take the parsed PDF data and save it into a Delta Table, which is Databricks' optimized storage format. Delta Tables provide reliable data storage with features like automatic versioning and schema management. This becomes your clean, queryable data source for the next steps.
# MAGIC 3. Create Vector Search Index
# MAGIC Convert your text data into vector embeddings and create a searchable index that enables semantic search. This allows you to find documents based on meaning rather than just keywords. The vector index is essential for building AI applications like chatbots or document Q&A systems.
# MAGIC 4. There are some provided notebooks that you can use to parse PDFs more quickly and reliably than ai_parse_document if you wish
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Section 2: Functions
# MAGIC 1. Create X amount of managed functions to complete task. This can be Genie Spaces, Model serving Endpoints, Agent Bricks
# MAGIC 2. Create regular python functions that can execute code and create visuals. 
# MAGIC 3. Optional: Try out Managed MCP and incorporate that as a tool

# COMMAND ----------

# MAGIC %md
# MAGIC #Section 3: Deployment
# MAGIC
# MAGIC 1. Use Agents.deploy to deploy your agent to an agent endpoint 
# MAGIC 2. Make sure to install mlflow 3.0 to take advantage of the latest experiment and traces tracking. 
# MAGIC 3. (Optional) Deploy to a Databricks apps UI
# MAGIC
# MAGIC Databricks is taking a model as code approach for deploying agents since there are so many difference pieces that can be defined in many different ways. For maximum compatibility, we recommend making a agent.py file to deploy as a model. 
# MAGIC
# MAGIC The workflow is shown below. You will take advantange of the typical mlflow capabilities to deploy this

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC
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
# MAGIC
# MAGIC # Autolog DSPy traces to MLflow
# MAGIC mlflow.dspy.autolog()
# MAGIC
# MAGIC # Set up DSPy with a Databricks-hosted LLM
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
# MAGIC dspy.settings.configure(lm=lm)
# MAGIC
# MAGIC ######################################
# MAGIC ## Create our Signature. Make as many as you need
# MAGIC ######################################
# MAGIC class genie_selector_agent(dspy.Signature):
# MAGIC   """
# MAGIC   Given the sql_instructions, determine which genie space tool to call, send the exact sql_instruction text to the tool and answer the question given the response from the tool.
# MAGIC   """ 
# MAGIC   sql_instruction: str = dspy.InputField()
# MAGIC   response: str = dspy.OutputField() 
# MAGIC   sql_query_output:  list = dspy.OutputField()
# MAGIC
# MAGIC ######################################
# MAGIC ## Create custom modules 
# MAGIC ######################################
# MAGIC
# MAGIC #this is entire up to you if you want to prepackage some modules together to collective complete a task
# MAGIC
# MAGIC ######################################
# MAGIC ## Create our ChatAgent. This is an object MLflow needs to recognize what kind of model this is. You'll notice its design is very similar to a custom module
# MAGIC ######################################
# MAGIC
# MAGIC class DSPyChatAgent(ChatAgent):     
# MAGIC     def __init__(self): #instantiate the agents or signatures that you need
# MAGIC       self.genie_selector_agent = genie_selector_agent
# MAGIC       self.multi_genie_agent = dspy.ReAct(self.genie_selector_agent, tools=[self.hls_patient_genie, self.investment_portfolio_genie], max_iters=1)
# MAGIC
# MAGIC     ######################################
# MAGIC     ## Define your tools below within the ChatAgent so that the class knows these exist. You can define these outside the class as well if you like
# MAGIC     ######################################
# MAGIC
# MAGIC     def hls_patient_genie(self, sql_instruction):
# MAGIC
# MAGIC       w = WorkspaceClient()
# MAGIC       genie_space_id = "01effef4c7e113f9b8952cf568b49ac7"
# MAGIC
# MAGIC       # Start a conversation
# MAGIC       conversation = w.genie.start_conversation_and_wait(
# MAGIC           space_id=genie_space_id,
# MAGIC           content=f"{sql_instruction} always limit to one result"
# MAGIC       )
# MAGIC
# MAGIC       response = w.genie.get_message_attachment_query_result(
# MAGIC         space_id=genie_space_id,
# MAGIC         conversation_id=conversation.conversation_id,
# MAGIC         message_id=conversation.message_id,
# MAGIC         attachment_id=conversation.attachments[0].attachment_id
# MAGIC       )
# MAGIC
# MAGIC       return response.statement_response.result.data_array
# MAGIC
# MAGIC     def investment_portfolio_genie(self, sql_instruction):
# MAGIC
# MAGIC       w = WorkspaceClient()
# MAGIC       genie_space_id = "01f030d91cc6165d88aaee122a274294"
# MAGIC
# MAGIC       # Start a conversation
# MAGIC       conversation = w.genie.start_conversation_and_wait(
# MAGIC           space_id=genie_space_id,
# MAGIC           content=f"{sql_instruction} always limit to one result"
# MAGIC       )
# MAGIC
# MAGIC       response = w.genie.get_message_attachment_query_result(
# MAGIC         space_id=genie_space_id,
# MAGIC         conversation_id=conversation.conversation_id,
# MAGIC         message_id=conversation.message_id,
# MAGIC         attachment_id=conversation.attachments[0].attachment_id
# MAGIC       )
# MAGIC
# MAGIC       return response.statement_response.result.data_array
# MAGIC   
# MAGIC     #very basic memory implementation
# MAGIC     def prepare_message_history(self, messages: list[ChatAgentMessage]):
# MAGIC         history_entries = []
# MAGIC         # Assume the last message in the input is the most recent user question.
# MAGIC         for i in range(0, len(messages) - 1, 2):
# MAGIC             history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
# MAGIC         return dspy.History(messages=history_entries)
# MAGIC
# MAGIC     ######################################
# MAGIC     ## This predict method is where the interaction first starts. If you want to change what happens here, you can. It must return ChatAgentResponse to be compatible with agents.deploy
# MAGIC     ######################################
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         latest_question = messages[-1].content
# MAGIC         response = self.multi_genie_agent(sql_instruction=latest_question).response
# MAGIC         return ChatAgentResponse(
# MAGIC             messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
# MAGIC         )
# MAGIC
# MAGIC # Set model for logging or interactive testing
# MAGIC from mlflow.models import set_model
# MAGIC AGENT = DSPyChatAgent()
# MAGIC set_model(AGENT)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log your Agent with passthrough authentication

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)
from pkg_resources import get_distribution


# TODO : set the genie_space_id for each Genie Space you want to call
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id= "01f0357a714f14b39ec53dfeb7c916b5"),
    DatabricksGenieSpace(genie_space_id= "01f0357a519d17cd96ad784b8afce762"),
    DatabricksVectorSearchIndex(index_name="jai_behl.ias.knowledge_base")
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        # input_example=input_example,
        extra_pip_requirements=[f"databricks-connect=={get_distribution('databricks-connect').version}"],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register your Agent to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model.
catalog = "jai_behl"
schema = "ias"
model_name = "dspy_multi_genie"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC #Dev Time

# COMMAND ----------

