# DSPy - Databricks Cookbook

## Overview
The purpose of this cookbook is to provide a starting template to build dspy applications on Databricks. It covers how to get started using dspy with mlflow, databricks agents, synthetic eval, and provides a guide to understand the value of dspy when building genAI applications.  As a first example we're looking at a simple RAG agent, however DSPy really excels at more complex and nuanced tasks. 

## Instructions
1. Start by completing all the prequisites below
2. Populate the 

## Prequisites
### 1) Generate a PAT token and store it as a Databricks secret

1. Log into your Databricks workspace.
1. Navigate to the User Settings:
    1. Click on your user profile icon in the top right corner of the Databricks UI.
    1. Select "Settings" from the dropdown menu.
    1. Navigate to the "Developer" section
1. Generate a Personal Access Token:
   1. Go to manage "Access Tokens" 
   1. Click on the "Generate New Token" button.
   1. You will be prompted to provide a comment and an expiration period for the token.
   1. After creating the token, make sure to copy it immediately. You won't be able to see it again once you navigate away from the page.
1. [Secure the Token](https://docs.databricks.com/aws/en/security/secrets/?language=Databricks%C2%A0CLI#manage-secret-scopes)

### 2) Create compute
<!-- TODO: Include compute specs -->


### 3) Update config.yaml
<!-- TODO -->

### 4) Run the setup notebooks


### 5) Upload sample data to the volume

