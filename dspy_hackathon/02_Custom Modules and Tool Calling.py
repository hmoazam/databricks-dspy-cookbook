# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade dspy==3.0.0b4

# COMMAND ----------

import dspy
import mlflow
# llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)
mlflow.dspy.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC #Section 1: Creating your own modules 
# MAGIC
# MAGIC Now that you've created your own signatures and have learned the various ways to use modules, it's time to start building. 
# MAGIC
# MAGIC DSPy builds upon these core concepts to give you the most flexibility when designing your applications. Because of its pure python approach, you can integrate any Python library and any Python logic within DSPy. This is where we create custom DSPy Modules.
# MAGIC
# MAGIC Fundamentally, a custom DSPy module looks like the structure below:

# COMMAND ----------

class MyProgram(dspy.Module):
    
    def __init__(self, ...):
        # Define attributes and sub-modules here
        {constructor_code}

    def forward(self, input_name1, input_name2, ...):
        # Implement your program's logic here
        {custom_logic_code}

# COMMAND ----------

# MAGIC %md
# MAGIC It's very similar to Pytorch and how that framework approaches Neural Network development. It inherits dspy.Module that contains the `__call__` method. 
# MAGIC
# MAGIC We can best illustrate the benefit of this by using python functions in conjunction with our DSPy signatures. RAG does not always have to be a function/tool call (more on this later). We can always call the LLM first, call a tool, then augment the pulled information with another LLM call. By using pure Python logic inbetween our LLM calls, we can benefit from the following: 
# MAGIC
# MAGIC 1. **Less Black Box approach**. Unlike the black box nature of an LLM, we know exactly what our Python code is doing
# MAGIC 2. **More reliable**. Unsure if your LLM is retrieving the right information? You can use Python to ensure you get the right information from the right tools
# MAGIC 3. **Smaller Language Models**. You need powerful LLMs that do function calling very well. By utilizing Python logic, we can afford to use smaller LLMs and save on cost while increasing performance
# MAGIC
# MAGIC Let's review one of DSPy's examples to understand how we can use signatures together and use their outputs downstream in Python logic and future LLM calls to orchestrate Agentic logic: 

# COMMAND ----------

import dspy
######################################
## Let's make our signatures
######################################
#Create the first signature that converts the question into a query
class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

#Create the second signature that answers the original question with the added context
class QuestionAndAnswer(dspy.Signature):
    """Answer the question based on the provided context"""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    identified_number: int = dspy.OutputField() 
    
######################################
## Now let's define a tool/python function that we will use to do too/function calling
######################################

#The tool we will use to find more context based on the user's question
def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return [x["text"] for x in results]

######################################
## Put it altogher in your own custom module!
######################################

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator) #The first signature and first LLM call 
        self.answer_generator = dspy.ChainOfThought(QuestionAndAnswer) #the 2nd signature and second LLM call but using ChainOfThought this time

    def forward(self, question, **kwargs):
        with dspy.context(lm=dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct')): #We can change what LLM we use for a specific DSPy call. 
            query = self.query_generator(question=question).response #we use the first signature to convert the question from the user into a query. We use the attribute access to get only the query, not the dspy.Prediction
        context = search_wikipedia(query)[0] #The query created by the LLM is used to serach wikipedia
        return self.answer_generator(question=question, context=context).answer #the context retrieved from wikipedia is then sent to the 2nd inline signature call to create the final answer

# COMMAND ----------

#Now let's execute the code
rag = RAG() #We instantiate the packaged up dspy.Module 
print(rag(question="Is Lebron James the basketball GOAT?"))

# COMMAND ----------

# MAGIC %md
# MAGIC With the custom RAG module, we can now do two LLM calls with two distinct DSPy Signatures packaged up into one module. You can take this custom module and put it in another module and continue developing modules to put these pieces together. 
# MAGIC
# MAGIC We did not use any Agentic Logic here. We simply used the DSPy outputs downstream to feed Python functions and other LLM calls to accomplish our task. But, in essence, we accomplished RAG, just without the vector search and Agentic reasoning part

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1: Make your own custom module: Stock Analyzer 
# MAGIC
# MAGIC In this task, we will be making a stock analyzer agent. You will receive an article about a specific company and you will try to pull relevant stock information using a python function pinging the Yahoo Finance API. 
# MAGIC
# MAGIC This Python function is provided for you below. The article to provide to the LLM is provided below as well.
# MAGIC
# MAGIC Your goal is create one module that accepts the article, identifies all the companies in the article, creates a list of their stock tickets, sends this list to the yahoo Python function and then sends all the results to a final LLM call to analyze the stock activity of these companies. 
# MAGIC
# MAGIC I will leave it up to you on how you want the module to provide the final response 
# MAGIC
# MAGIC Check your work using MLflow Traces as the created stock ticker could be incorrect
# MAGIC
# MAGIC Limitations: 
# MAGIC 1. Start with Llama-8B. You should be able to accomplish this task with just llama-8b. The more powerful models will almost definitely work. If you're having trouble, feel free to bump up the model to Llama-70B
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Yahoo Finance Tool
import yfinance as yf
from typing import Dict, Optional

def get_stock_data_yahoo(symbol: str) -> Optional[Dict]:
    """
    Fetch stock data from Yahoo Finance using yfinance
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Dictionary containing stock data or None if error
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get current info
        info = ticker.info
        
        # Get recent price data
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol.upper(),
                'price': round(latest_price, 2),
                'company_name': info.get('longName', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'previous_close': info.get('previousClose', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A')
            }
        else:
            print(f"No data found for symbol: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# COMMAND ----------

article = """
The U.S. Space Force has awarded Boeing a $2.8 billion contract for secure satellite communications under its Evolved Strategic SATCOM (ESS) program.

The program aims to enhance national defense communications with two initial satellites, with future options including expansion and Arctic capabilities, supporting operations from key facilities such as Vandenberg Space Force Base (VBG).

Boeing Wins $2.8 Billion New Space Force SATCOM Contract
By Horizon206 – Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=134697393
Boeing Secures ESS Contract
Boeing outpaced Northrop Grumman to secure the contract, marking a critical milestone in the development of the U.S. military’s next-generation satellite communication systems. This $2.8 billion deal funds two satellites, with the U.S. Space Force retaining the option to procure two more under the broader $12 billion ESS initiative, Defense News reported.


freestar
Advertisement

freestar
The ESS system is designed to replace the aging Advanced Extremely High Frequency (AEHF) satellite constellation. It will provide enhanced survivability, resilience, and cybersecurity to address growing threats in space.

Boeing’s proposal stood out for its innovative architecture and capacity to deliver guaranteed communication in high-threat environments.

According to Kay Sears, VP and GM of Boeing’s Space, Intelligence, and Weapons Systems division, the system is engineered to meet evolving national security needs with unmatched reliability.

The U.S. Space Force has awarded Boeing a $2.8 billion contract for secure satellite communications under its Evolved Strategic SATCOM (ESS) program.
Photo: By Clemens Vasters from Viersen, Germany, Germany – Northrop Grumman B-2 Spirit, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=50405787
Long-Term Strategic Goals
The contract runs through 2033 and represents a major portion of the Space Force’s long-term SATCOM evolution strategy.


freestar
Beyond the first satellites, ESS may include Arctic-specific capabilities to support operations in high-latitude regions, a growing area of interest for defense planners.

The Space Force has also emphasized its intent to pivot toward a “family of systems” strategy. This approach will ensure incremental capability upgrades delivered on faster timelines, especially for anti-jamming and protected tactical communication functions.


NASA HQ PHOTO | Credit: (NASA/Joel Kowsky)
Cancellation of PTS-R and Shift in Strategy
In a related move, the Protected Tactical SATCOM–Resilient (PTS-R) program has been officially canceled. The Space Force said this shift reflects a new strategy focusing on incremental capability delivery via existing frameworks like the Protected Tactical Waveform (PTW).

While PTS-R has been discontinued, several core components of the SATCOM architecture remain in development. These include:


freestar
Protected Tactical SATCOM–Global
Protected Tactical Enterprise Service
Enterprise Management and Control
Air Force-Army Anti-Jam Modem
Initial prototypes of the Protected Tactical SATCOM program are expected to launch in 2026, reinforcing the broader shift to modular, adaptive systems with lower risk and reduced procurement costs."""

# COMMAND ----------

import dspy
import json
import mlflow

llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

class TickerIdentifier(dspy.Signature):
    """TODO"""
    article: TODO
    stock_ticker: list[dict] TODO

class StockAnalysis(dspy.Signature):
    """TODO"""
    stock_information: TODO
    answer: TODO

class StockAnalyzer(dspy.Module):
    def __init__(self):
        self.stock_list_creator = TODO
        self.answer_generator = TODO

    def forward(self, article, **kwargs):        
        stock_list = TODO
        yahoo_result = TODO
        return TODO
      
stock_assistant = StockAnalyzer()
print(stock_assistant(article=article))

# COMMAND ----------

#Answer Key
import dspy
import json
import mlflow
llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-gemma-3-12b', cache=False)
# llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

class TickerIdentifier(dspy.Signature):
    """Read the entire article and find all companies in the article. Then create their stock ticker"""
    article: str = dspy.InputField()
    stock_ticker: list[dict] = dspy.OutputField(desc="Example: {'company': 'name', 'ticker': 'ticker_name'}")

class StockAnalysis(dspy.Signature):
    """Analyze the provided companies stock information and determine market health"""
    stock_information: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="be descriptive and talk about each company")

class StockAnalyzer(dspy.Module):
    def __init__(self):
        self.stock_list_creator = dspy.ChainOfThought(TickerIdentifier)
        # self.stock_list_creator = dspy.Predict(TickerIdentifier)
        self.answer_generator = dspy.Predict(StockAnalysis)

    def forward(self, article, **kwargs):        
        stock_list = self.stock_list_creator(article=article).stock_ticker 
        yahoo_result = [get_stock_data_yahoo(stock['ticker']) for stock in stock_list]
        return self.answer_generator(stock_information=yahoo_result).answer
      
stock_assistant = StockAnalyzer()
print(stock_assistant(article=article))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Additional Exercise
# MAGIC Consider the following: 
# MAGIC
# MAGIC 1. How would you do this without DSPy? Try to make a prompt to get the stock tickers off of the article as a list using llama-8b in the playground
# MAGIC 2. What's the benefit of separating out the LLM calls like this? 
# MAGIC 3. Could you use a weaker model like Llama-8B or would you need to rely on the power of Claude or GPT-4.1 to accomplish this task?  

# COMMAND ----------

# MAGIC %md
# MAGIC #Section 2: Function Calling with DSPy
# MAGIC
# MAGIC Sometimes, you just simply don't know what kind of input is going to be provided for your Agents or GenAI applications. As such, we do need to rely on the reasoning capabilities of the LLM to determine which function to call next to find the most relevant information or execute a specific task to complete the request. 
# MAGIC
# MAGIC Technically, you've been doing function calling in the above section. That is because function calling requires a minimum of 2 LLM calls. The process is as follows:
# MAGIC 1. The first call determines what function to use to answer the question. 
# MAGIC 2. Then, the function needs to be executed. 
# MAGIC 3. The second call takes the results of this function and completes the request
# MAGIC
# MAGIC In the last exercise, you took the outputs from the first LLM call, used a python function to process the outputs and then sent the processed infromation to a final LLM call. This was possible because you knew what to expect and could handle it programmatically. DSPy makes this easier by providing typing expectations from the signature. As you likely saw from the playground exercise, consistently getting the same output was likely extremely difficult when using weaker models. 
# MAGIC
# MAGIC However, Function Calling heavily depends on the capabilities of the underlying model. You will likely be unable to accomplish function calling use cases with a model like Llama-8B unless it was specifically trained to do this well. The model needs to be able to recognize when to call a tool and know when a tool finished executing to move on to the next step.You'll see in the 8B example how it struggles to move on after an initial function call. 
# MAGIC
# MAGIC We usually default to at least Llama-70B to accomplish function calling tasks. 
# MAGIC
# MAGIC There are two ways to do function calling on DSPy
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1: dspy.ReAct
# MAGIC
# MAGIC This is the easiest method and packages up the function call into a module called dspy.ReAct. It does come at the cost of some latency due to the additional LLM calls this does on your behalf. 
# MAGIC
# MAGIC Here is an example from DSPy below: 

# COMMAND ----------

llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

#As we saw with the math example in the first notebook, LLMs are bad at math. It's better if we could just use good old python to execute the math equation. 
def evaluate_math(expression: str) -> float:  
    return eval(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer) #5761.3 is the answer

# COMMAND ----------

# MAGIC %md
# MAGIC Let's walk through this example: 
# MAGIC 1. The LLM got a question that was missing someinformation: The year of birth of David Gregory. So it decided to use the wikipedia tool first. 
# MAGIC 2. After receiving that information, it then called the math function to do the calculation.
# MAGIC 3. Although it got the right answer, we can review the MLflow Trace to see that the LLM actually called the evaluate_math function multiple times as it was not getting the right answer at all. It executed this function 10 more times. 
# MAGIC
# MAGIC When using function calling, and this applies to all LLMs, the way the function is defined and described is incredibly important. The name of the function, docstring, parameters and so forth all tell the LLMs when and how to use the function. Thus, that does require the LLM to be powerful enough to find the right function and use it. 
# MAGIC
# MAGIC Let's try this again but using a more powerful model like Llama-70B

# COMMAND ----------

# llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

#As we saw with the math example in the first notebook, LLMs are bad at math. It's better if we could just use good old python to execute the math equation. 
def evaluate_math(expression: str) -> float:  
    return eval(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)

# COMMAND ----------

# MAGIC %md
# MAGIC Great, in the MLflow Trace, we can see that switching the model GREATLY helped in calling only the necessary tools and correctly. 
# MAGIC
# MAGIC Now let's try asking an entirely different question. For this question, we don't want to call evaluate_math at all. In a real life scenario, we simply could not predict what kind of question would be asked so we have to leave it to the LLM to decide the best course of action

# COMMAND ----------

pred = react(question="IS Lebron James the Goat")
print(pred.answer)

# COMMAND ----------

# MAGIC %md
# MAGIC In the MLflow Trace, we can see evaluate_math was not called. If we had tried to code that in ourselves in a custom module, we would encounter an error or need to do extensive exception handling to make sure we accomodate for whatever input comes in. 
# MAGIC
# MAGIC This is where Agents shine, being able to handle the uncertainty that comes in from user inputs, unstructured data and figuring out how to accomplish the task 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2: Use DSPy.Tool
# MAGIC
# MAGIC DSPy gives you the flexibility to use dspy.Tool as a type within your signature. This returns a dspy.ToolCalls that contains what the LLM decided to send to the tool. So, you can actually check what the LLM decided to send to the Tool, adjust it if needed, then execute the tool, before sending the output to another LLM call. This is very similar to what you were doing in Section 1, except, the LLM decides what tool to call next and what that output should be. 
# MAGIC
# MAGIC Dspy.Tool is a core component of dspy.ReAct. But, because DSPy let's you use this outside of ReAct, you lose some functionality like calling multiple functions. This is on you to implement. It has incredbily high benefits if you are very comfortable in python development. 
# MAGIC
# MAGIC Let's see it in action below using the same example above:

# COMMAND ----------

# llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)
                 
#As we saw with the math example in the first notebook, LLMs are bad at math. It's better if we could just use good old python to execute the math equation. 
def evaluate_math(expression: str) -> float:  
    return eval(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

class ToolCallingQnA(dspy.Signature):
  """Answer the question using tools as necessary"""
  question: str = dspy.InputField()
  search_wikipedia: dspy.Tool = dspy.InputField()
  evaluate_math: dspy.Tool = dspy.InputField()  
  answer: str = dspy.OutputField()
  tool_call_output: dspy.ToolCalls = dspy.OutputField()

react = dspy.Predict(ToolCallingQnA) #Notice how we use dspy.Predict instead of dspy.ReAct

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?", search_wikipedia=dspy.Tool(search_wikipedia), evaluate_math=dspy.Tool(evaluate_math))
                      
#Very crude execution of each function. You would need to execute this recursively until the tool calls are complete
if pred.tool_call_output.tool_calls[0].name == 'search_wikipedia':
  wiki_result = search_wikipedia(pred.tool_call_output.tool_calls[0].args['query'])
  pred = react(question=f"What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle? Wikipedia Context: {wiki_result}", evaluate_math=dspy.Tool(evaluate_math))

if pred.tool_call_output.tool_calls[0].name == 'evaluate_math':
  math_evaluation = evaluate_math(pred.tool_call_output.tool_calls[0].args['expression'])
  print(math_evaluation)

# COMMAND ----------

# MAGIC %md
# MAGIC While a bit crude in implenetation, I'm able to control what tools it has access to after each one completes and manipulate the inputs and outputs before sending it to the next LLM call. 
# MAGIC
# MAGIC When comparing this version with its equivalent dspy.ReAct version (earlier in cell 15) using Llama-8B, I'm able to execute this nearly 5s faster or more than double the speed than using dspy.ReAct and I avoid the excessive function calls it was making. It's something to consider if performance is a big concern.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2: Make a DSPy Agent or a Function Calling Enabled Signature
# MAGIC
# MAGIC Given we have a search_wikipedia function as well as a yahoo stock function, let's create a Stock and Wikipedia Agent that uses both or one sources to answer questions. 
# MAGIC
# MAGIC Your goal is to provide both as tools to find the following information: 
# MAGIC
# MAGIC 1. The same company stock information 
# MAGIC 2. Information about said companies on wikipedia 
# MAGIC 3. Define a new python function that uses spark.sql to write information about said companies to a delta table with the following columns: 
# MAGIC   
# MAGIC    a. Company Name
# MAGIC    
# MAGIC    b. Company Stock Ticker
# MAGIC    
# MAGIC    c. Company Stock Summary 
# MAGIC    
# MAGIC    d. Company Wikipedia Summary 
# MAGIC    
# MAGIC    It's up to you if you want to let the LLM execute this query or you execute this query
# MAGIC
# MAGIC 4. Provide an end summary stating this was all completed
# MAGIC
# MAGIC Limitations: Llama-70B must be used.
# MAGIC
# MAGIC Input example: What news is affecting Boeing's stock and financial health? 
# MAGIC
# MAGIC

# COMMAND ----------

import wikipedia

llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

class stock_wikipedia_agent(dspy.Signature): 
  """TODO"""
  question: #TODO 
  response: #TODO


def yfinance(TODO):  #Hint this was created for you earlier 
    # TODO

def search_wikipedia(TODO): #Hint use AI to make a more comprehensive wikipedia tool that actually searches wikipedia. 
    #TODO

react = dspy.ReAct(stock_wikipedia_agent, tools=[yfinance, search_wikipedia])

pred = react(question="What news is affecting Boeing's stock and financial health? ")
print(pred.response)

# COMMAND ----------

import wikipedia

llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)

class stock_wikipedia_agent(dspy.Signature): 
  """A stock analysis agent"""
  question: str = dspy.InputField() 
  response: str = dspy.OutputField()

def get_stock_data_yahoo(symbol: str) -> Optional[Dict]:
    """
    Fetch stock data from Yahoo Finance using yfinance
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Dictionary containing stock data or None if error
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get current info
        info = ticker.info
        
        # Get recent price data
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol.upper(),
                'price': round(latest_price, 2),
                'company_name': info.get('longName', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'previous_close': info.get('previousClose', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A')
            }
        else:
            print(f"No data found for symbol: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def search_wikipedia(query, num_results=10, get_suggestions=False):
    """
    Search Wikipedia for articles matching the query.
    
    Args:
        query (str): The search term
        num_results (int): Number of search results to return (default: 10)
        get_suggestions (bool): Whether to include search suggestions (default: False)
    
    Returns:
        dict: Search results with titles and suggestions (if requested)
    """
    try:
        if get_suggestions:
            results, suggestion = wikipedia.search(query, results=num_results, suggestion=True)
            return {
                'results': results,
                'suggestion': suggestion,
                'query': query
            }
        else:
            results = wikipedia.search(query, results=num_results)
            return {
                'results': results,
                'query': query
            }
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            'results': e.options[:num_results],
            'query': query,
            'note': 'Multiple options found - showing disambiguation pages'
        }
    except Exception as e:
        return {
            'error': str(e),
            'query': query
        }

react = dspy.ReAct(stock_wikipedia_agent, tools=[get_stock_data_yahoo, search_wikipedia])

pred = react(question="What news is affecting Boeing's stock and financial health? ")
print(pred.response)

# COMMAND ----------

# MAGIC %md
# MAGIC #Task 3: Create a custom module that does function calling 
# MAGIC
# MAGIC Now that you have a function calling signature and two tools that it can call, how would you put this into a custom module? 

# COMMAND ----------

class stock_wikipedia_module(dspy.Module): 
  def __init__(self):
    agent_signature = #TODO 

  def yfinance(self, #TODO): 
  
  def search_wikipedia(self, #TODO):
                       
  def forward(#TODO): 
              

run_agent = stock_wikipedia_module()
result = run_agent(TODO=TODO)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC #Workshop: Create a Databricks Genie Space Calling Agent
# MAGIC
# MAGIC For this activity, I will leave the team to create a single DSPy module that has access to two Genie Spaces. They should be unique in the data that they can query. 
# MAGIC
# MAGIC Your goal is to create an agent that can alternate asking the two genie spaces for answers to the questions. 
# MAGIC
# MAGIC You will need to do the following: 
# MAGIC 1. Create two Genie Spaces either with existing data or new data
# MAGIC 2. Create two Python Functions that can query those two Genie Spaces. 
# MAGIC 3. Use DSPy to do function/tool calling and use these two Genie Spaces to answer question 
# MAGIC 4. Use MLflow Traces to test and review the accuracy of your Agent 
# MAGIC 5. Implement some kind of memory for your Agent so that it remembers or has some kind of history it can access to know what's been talked about in the past 
# MAGIC
# MAGIC Resources: 
# MAGIC 1. Accessing Genie via Databricks AI bridge: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_ai_bridge.html 
# MAGIC 2. Accessing Genie via Databricks SDK: https://databricks-sdk-py.readthedocs.io/en/stable/workspace/dashboards/genie.html
# MAGIC 3. Genie Conversation API Documentation: https://docs.databricks.com/aws/en/genie/conversation-api
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade git+https://github.com/stanfordnlp/dspy.git openai litellm "mlflow[databricks]>=3.1.0" "databricks-connect>=16.1" unitycatalog-ai[databricks] databricks-sdk databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dspy
import mlflow
# llm = dspy.LM('databricks/databricks-meta-llama-3-1-8b-instruct', cache=False)
llm = dspy.LM('databricks/databricks-meta-llama-3-3-70b-instruct', cache=False)
# llm = dspy.LM('databricks/databricks-claude-3-7-sonnet', cache=False)
dspy.configure(lm=llm)
mlflow.dspy.autolog()

# COMMAND ----------

import pandas as pd

df = pd.read_csv("./financial.csv")
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").option("delta.columnMapping.mode", "name").saveAsTable('you_delta_table here')

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def stock_info_genie(stock_question):
  """Pull stock information"""
  w = WorkspaceClient()
  genie_space_id = "01f06e37644a130182f2644a3fa0fc9c" #replace this with your genie space ID that you created

  conversation = w.genie.start_conversation_and_wait(
      space_id=genie_space_id,
      content=stock_question
  )

  response = w.genie.get_message_attachment_query_result(
    space_id=genie_space_id,
    conversation_id=conversation.conversation_id,
    message_id=conversation.message_id,
    attachment_id=conversation.attachments[0].attachment_id
  )

  return response.statement_response.result.data_array

# COMMAND ----------

class genie_question_answer(dspy.Signature): 
  """answers questions about stocks"""
  question: str = dspy.InputField() 
  response: str = dspy.OutputField() 

genie_llm = dspy.ReAct(genie_question_answer, tools=[stock_info_genie], max_iters=1)

# COMMAND ----------

result = genie_llm(question="when did apple's stock drop by more than 5%?")
print(result.response)

# COMMAND ----------

# MAGIC %md
# MAGIC #Example Signatures and Modules 
# MAGIC
# MAGIC Below is a list of signatures and modules I've created for customers and demos that you can use as a reference.

# COMMAND ----------

# MAGIC %md
# MAGIC these signatures were used altogether in a complete agent system from my DAIS presentation

# COMMAND ----------

from typing import List, Any

class memoryHistory(dspy.BaseType):
  history: List[dict] 
  last_message: List[str]
  summary_so_far: str
  # placeholder: str

  def format(self) -> list[dict[str, Any]]:
    return [
      {
      "type": "memory", 
      "memory": {
        "history": self.history, 
        "message": self.last_message, 
        "summary": self.summary_so_far,
        "placeholder": self.placeholder,
        }
      }
      ]

class text_summarizer_extraction(dspy.Signature): 
  """Agent to summarize the ocr output and find keywords based on the original query."""

  ocr_input: str = dspy.InputField()
  original_query: str = dspy.InputField()
  memory_so_far: memoryHistory = dspy.InputField(desc="a history of the workflow so far")
  response: str = dspy.OutputField()
  summary_so_far: str = dspy.OutputField()
  keywords: str = dspy.OutputField()
  next_agent_or_tool: Literal["text_processing_agent", "patient_lookup_genie_agent", "final_agent"] = dspy.OutputField() 

class genie_agent(dspy.Signature): 
  """Agent to use Databricks Genie Space to find information about a patient. It creates a question based on the provided keywords in patient_information or memory_history to query the genie_space with only the patient's name. Then, it takes the genie_output, makes a text_query based on insurance type, insurance name and keyterms like deductible found in both genie_outputs and original_query and sends the text_query to patient_insurance_lookup"""

  patient_information: str = dspy.InputField(desc="Find the patient's name")
  original_query: str = dspy.InputField()
  memory_so_far: memoryHistory = dspy.InputField(desc="a history of the workflow so far")
  genie_output: str = dspy.OutputField()
  insurance_details: str = dspy.OutputField()
  response: str = dspy.OutputField()
  summary_so_far: str = dspy.OutputField()
  deductible: str = dspy.OutputField(desc="this is the result of patient_insurance_lookup")
  next_agent_or_tool: Literal["text_processing_agent", "patient_lookup_genie_agent", "final_agent"] = dspy.OutputField() 

class final_agent(dspy.Signature):
  """Agent to convert the collected information and write to a delta table based on the original_query.""" 

  original_query: str = dspy.InputField()
  genie_output: list = dspy.InputField() 
  ocr_summary: str = dspy.InputField() 
  deductible: str = dspy.InputField()
  completed_response: str = dspy.OutputField()

class document_analyzer(dspy.Signature):
  """Agent to analyze the document provided by reviewing the outputs of the model and determining if there's enough information to go to the next agent or try analyzing the document again with a different vision model""" 

  vision_model_output: str = dspy.InputField()
  response: str = dspy.OutputField() 
  next_agent_or_tool: Literal["text_processing_agent", "patient_lookup_genie_agent", "final_agent"] = dspy.OutputField() 

class insurance_finder(dspy.Signature):
  """Find the relevant information based on the text_query within the image"""

  image: dspy.Image = dspy.InputField()
  text_query: str = dspy.InputField()
  deductible: str = dspy.OutputField()
  other_information: str = dspy.OutputField()

# COMMAND ----------

# MAGIC %md
# MAGIC MCP Server Signature and Code

# COMMAND ----------

import dspy
import os
import aiohttp
from mcp.server.fastmcp import FastMCP
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import asyncio
from databricks.sdk.core import Config
import asyncio
import nest_asyncio

nest_asyncio.apply()

lm = dspy.LM("databricks/<model-serving-endpoint-name>")

# token = dbutils.secrets.get(scope="groq_key", key="service_secret_pat")

config = Config()
token = config.oauth_token().access_token
#Change the URL to the URL of your MCP Server or Databricks App if a custom MCP: https://docs.databricks.com/aws/en/generative-ai/agent-framework/mcp#managed-mcp-servers

transport = StreamableHttpTransport(
    # url="https://genie-app-vivian-1444828305810485.aws.databricksapps.com/api/mcp/",
    url = "https://telco-operations-mcp-server-1444828305810485.aws.databricksapps.com/api/mcp/", 
    headers={"Authorization": f"Bearer {token}"}
)

class MCP_Test(dspy.Signature):
    """You are given a list of tools to handle user requests.
    Use the genie-query tool to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField()

client = Client(transport)
async def main():
    async with client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}\n\n")
        dspy_tools = []
        for tool in tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(client, tool))

        react = dspy.ReAct(MCP_Test, tools=dspy_tools)
        result = await client.call_tool(
            name="check-outage-status",
            # arguments={"query": "List top 3 distribution centers"}
            arguments={"query": "What is the total raw material demand by product?"}   
        )
        print(result)
        

asyncio.run(main())


# COMMAND ----------

# MAGIC %md
# MAGIC DSA Blog Post on interacting with a Vector Search Index and Genie Space
# MAGIC
# MAGIC Repo: https://github.com/databricks-solutions/databricks-blogposts/tree/main/2025-06-06-multi-modal-hls-DSA

# COMMAND ----------

class image_analyzer(dspy.Signature):
  """review the image and genie_patient_response to answer the text_query"""
  image: dspy.Image = dspy.InputField() 
  genie_patient_response: list = dspy.InputField()
  text_query: str = dspy.InputField()
  response: str = dspy.OutputField() 
  deductible: str = dspy.OutputField()

class patient_information_extraction(dspy.Signature):
  """This class only extracts and returns information from relevant tools based on the text_query. Include relevant information from the genie_patient_response in the keywords_for_vector_search"""
  text_query: str = dspy.InputField()
  genie_patient_response: list = dspy.OutputField()
  keywords_for_vector_search: str = dspy.OutputField(desc="string of keywords to pass to vector search")

class MultiModalPatientInsuranceAnalyzer(dspy.Module):
  def __init__(self):
    super().__init__()
    self.image_analyzer = dspy.Predict(image_analyzer)
    self.patient_information_extraction = dspy.ReAct(patient_information_extraction, tools=[self.hls_patient_genie], max_iters=1)
  
  def process_image(self, base64_string):
    image_data = base64.b64decode(base64_string) 
    pil_image = Image.open(io.BytesIO(image_data))
    dspy_image = dspy.Image.from_PIL(pil_image)
    return dspy_image
  
  def vector_search_for_patient_pdf(self, text_query):
    """Pulls matching Insurance Documents based on the text_query"""
    client = mlflow.deployments.get_deploy_client("databricks") 
    response = client.predict(
              endpoint=model_endpoint_name,
              inputs={"dataframe_split": {
                      "columns": ["text"],
                      "data": [[text_query]]
                      }
              }
            )
    text_embedding = response['predictions']['predictions']['embedding']
    index = vs_client.get_index(endpoint_name=vector_search_endpoint_name, index_name=f"{catalog}.{schema}.{index_name}")
    results = index.similarity_search(num_results=3, columns=["base64_image"], query_vector=text_embedding)
    return results['result']['data_array'][0][0]
  
  def hls_patient_genie(self, patient_name):
    """Pull Patient information based on the patient's name"""
    w = WorkspaceClient()
    genie_space_id = "01effef4c7e113f9b8952cf568b49ac7" #replace this with your genie space ID that you created

    conversation = w.genie.start_conversation_and_wait(
        space_id=genie_space_id,
        content=f"Find any details about {patient_name}. Limit your answer to one result."
    )

    response = w.genie.get_message_attachment_query_result(
      space_id=genie_space_id,
      conversation_id=conversation.conversation_id,
      message_id=conversation.message_id,
      attachment_id=conversation.attachments[0].attachment_id
    )

    return response.statement_response.result.data_array


  def forward(self, text_query: str):
    results = self.patient_information_extraction(text_query=text_query)
    base64_str = self.vector_search_for_patient_pdf(text_query=results.keywords_for_vector_search)
    dspy_image = self.process_image(base64_string=base64_str)
    return self.image_analyzer(image=dspy_image, genie_patient_response=results.genie_patient_response, text_query=text_query)

# COMMAND ----------

import os
from databricks.vector_search.client import VectorSearchClient
#make sure to pip install databricks-vectorsearch databricks-sdk

vsc = VectorSearchClient(
        workspace_url="https://e2-demo-field-eng.cloud.databricks.com/",
        personal_access_token=""
    )

index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=vectorSearchIndexName)

result = index.similarity_search(num_results=3, columns=<add the columns you want to query>, query_text=<the text you want to query)

return result['result']['data_array'][0][0]