
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

# Autolog DSPy traces to MLflow
mlflow.dspy.autolog()

# Set up DSPy with a Databricks-hosted LLM
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
dspy.settings.configure(lm=lm)

class genie_selector_agent(dspy.Signature):
  """
  Given the sql_instructions, determine which genie space tool to call, send the exact sql_instruction text to the tool and answer the question given the response from the tool.
  """ 
  sql_instruction: str = dspy.InputField()
  response: str = dspy.OutputField() 
  sql_query_output:  list = dspy.OutputField()

class DSPyChatAgent(ChatAgent):     
    def __init__(self):
      self.genie_selector_agent = genie_selector_agent
      self.multi_genie_agent = dspy.ReAct(self.genie_selector_agent, tools=[self.hls_patient_genie, self.investment_portfolio_genie], max_iters=1)

    def hls_patient_genie(self, sql_instruction):

      w = WorkspaceClient()
      genie_space_id = "01effef4c7e113f9b8952cf568b49ac7"

      # Start a conversation
      conversation = w.genie.start_conversation_and_wait(
          space_id=genie_space_id,
          content=f"{sql_instruction} always limit to one result"
      )

      response = w.genie.get_message_attachment_query_result(
        space_id=genie_space_id,
        conversation_id=conversation.conversation_id,
        message_id=conversation.message_id,
        attachment_id=conversation.attachments[0].attachment_id
      )

      return response.statement_response.result.data_array

    def investment_portfolio_genie(self, sql_instruction):

      w = WorkspaceClient()
      genie_space_id = "01f030d91cc6165d88aaee122a274294"

      # Start a conversation
      conversation = w.genie.start_conversation_and_wait(
          space_id=genie_space_id,
          content=f"{sql_instruction} always limit to one result"
      )

      response = w.genie.get_message_attachment_query_result(
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
        response = self.multi_genie_agent(sql_instruction=latest_question).response
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
        )

# Set model for logging or interactive testing
from mlflow.models import set_model
AGENT = DSPyChatAgent()
set_model(AGENT)
