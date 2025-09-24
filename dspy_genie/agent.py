
import mlflow
from typing import Any, Generator, Optional
from databricks.sdk.service.dashboards import GenieAPI
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import dspy
import uuid
import datetime
import os

# Autolog DSPy traces to MLflow
mlflow.dspy.autolog()

# Set up DSPy with a Databricks-hosted LLM
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
dspy.settings.configure(lm=lm)


######################################
## Create our Genie Selector Signature
######################################
# This signature will be used to determine which Genie Space based on the request and utilize the relevant Genie Space using self-reasoning to select the proper tool.

# Deciding on your module’s inputs, outputs, and their types is the most critical part of creating a signature. This definition forms the contract that other components in your multi-agent pipeline will rely on.

class genie_selector_agent(dspy.Signature):
  """
  Given a question, determine which genie space tool to call, send the exact question to the tool and answer the question given the response from the tool.
  """ 
  # **NOTE** : You can add details to the prompt about each tool to increase the accuracy of the agent. Both the function name and prompt inform the agent what the tool is used for.

  question: str = dspy.InputField()
  response: str = dspy.OutputField() 
  sql_query_output:  list = dspy.OutputField()


#####################################
## Create our DSPY Chat Agent Class
#####################################
class DSPyChatAgent(ChatAgent):     
    def __init__(self):
      self.w = WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST"),
        token=os.getenv("DATABRICKS_TOKEN")
      )

      self.genie_selector_agent = genie_selector_agent

      self.multi_genie_agent = dspy.ReAct(self.genie_selector_agent, 
                                          tools=[self.australian_sales_space, self.us_sales_space], 
                                          max_iters=1)



    #######################################################
    ## Create a Function that will call the 1st Genie Space
    #######################################################

    def australian_sales_space(self, question):
      """This genie space is used to query data about Australia Sales. This genie space takes in a request and returns the relevant data."""

      # TODO add GENIE_SPACE_ID and a description for this space
      # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
      genie_space_id = "01f063657dfe1464aa0eec353df246f5"


      # Start a conversation
      conversation = self.w.genie.start_conversation_and_wait(
          space_id=genie_space_id,
          content=f"{question} always limit to one result",
          timeout=datetime.timedelta(minutes=20)
      )

      response = self.w.genie.get_message_attachment_query_result(
        space_id=genie_space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
        attachment_id=conversation.attachments[0].attachment_id
      )

      return response.statement_response.result.data_array
    
    #######################################################
    ## Create a Function that will call the 2nd Genie Space
    #######################################################
    
    def us_sales_space(self, question):
      """This genie space is used to query data about USA Retail Sales. This genie space takes in a request and returns the relevant data."""

      # TODO add GENIE_SPACE_ID and a description for this space
      # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
      genie_space_id = "01f06364428a132b8de538b325b2c24f"


      # Start a conversation
      conversation = self.w.genie.start_conversation_and_wait(
          space_id=genie_space_id,
          content=f"{question} always limit to one result",
          timeout=datetime.timedelta(minutes=20)
      )

      response = self.w.genie.get_message_attachment_query_result(
        space_id=genie_space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
        attachment_id=conversation.attachments[0].attachment_id
      )

      return response.statement_response.result.data_array


    def prepare_message_history(self, messages: list[ChatAgentMessage]):
        history_entries = []
        # Assume the last message in the input is the most recent user question.
        for i in range(0, len(messages) - 1, 2):
            history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
        return dspy.History(messages=history_entries)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        latest_question = messages[-1].content
        response = self.multi_genie_agent(question=latest_question).response
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
        )

# Set model for logging or interactive testing
from mlflow.models import set_model
AGENT = DSPyChatAgent()
set_model(AGENT)
