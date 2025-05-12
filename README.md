# DSPy - Databricks Cookbook

## Overview
The purpose of this cookbook is to provide a starting template to build [DSPy](https://dspy.ai/) applications on Databricks. It covers how to get started using DSPy with Databricks features such as vector search, MLFlow, Databricks Agent Framework and Mosaic Agent Eval. 

It also  provides a guide to understanding the value of DSPy when building GenAI applications on Databricks. 

As a teaching example, we're looking at a simple RAG agent, however DSPy really excels at more complex and nuanced tasks, so we encourage you to use these notebooks as a starting point for your more advanced projects.

## Instructions

### Setup
1. Generate a PAT token and store it as a Databricks secret
1. Create a compute instance to run the notebooks, this example is not compute intensive so a general purpose instance with 2 cores is sufficient, e.g. m5d.large. However, if you are a more advanced user looking to do multi-threaded optimization, aim to use compute such that #threads = #cores.  
1. Populate the `config.yaml` file with values for all the given fields. The values will be referenced throughout all the notebooks. 
1. Navigate to the setup folder, and run the first 3 notebooks. 
1. Once the volume is created, upload the files in the data folder to the volume. 
1. Run notebooks 04 and 05 in the setup folder.

### The DSPy cookbook
1. Go through the `01_dspy_without_opt_rag_agent` notebook
1. Go through the `02_dspy_with_opt_rag_agent` notebook
1. Go through the `03_register_and_deploy` notebook
