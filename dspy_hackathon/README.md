# DSPy Hackathon 2025

An introduction to DSPy - a framework for programming language models rather than prompting them.

## Overview

This repository contains educational materials and tutorials for learning **DSPy** (Declarative Self-improving Python)[1], a framework developed at Stanford that allows developers to build modular AI systems by writing compositional Python code instead of brittle prompts[1].

DSPy enables you to iterate fast on building structured AI applications and offers algorithms for optimizing prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or complex Agent loops[1].

## What is DSPy?

DSPy is a declarative framework for building modular AI software that allows you to iterate fast on structured code, rather than brittle strings[2]. Instead of manually crafting prompts, DSPy lets you focus on building AI pipelines by:

- **Programming, not prompting**: Write compositional Python code instead of fragile prompt strings[1]
- **Modular design**: Build complex systems using composable modules[3]
- **Automatic optimization**: Let DSPy optimize prompts and model weights for you[1]
- **Structured outputs**: Get reliable, structured responses from language models[4]

## Repository Contents

This repository includes three comprehensive Jupyter notebooks that progressively introduce DSPy concepts:

### 📚 Notebooks Overview

1. **`01_Intro to DSPy.ipynb`** - Introduction to DSPy fundamentals
2. **`02_Custom Modules and Tool Calling.ipynb`** - Building custom modules and integrating tools
3. **`03_Deploying with Databricks.ipynb`** - Production deployment strategies

## Key DSPy Concepts Covered

### Core Modules
- **Predict**: Basic predictor that handles key forms of learning[3]
- **ChainOfThought**: Teaches the LM to think step-by-step before responding[3]
- **ProgramOfThought**: Generates and executes Python code for complex reasoning[3]
- **ReAct**: Agent that can use tools to implement given signatures[3]
- **Retrieve**: Handles retrieval for RAG applications[3]

### Advanced Features
- **Signatures**: Define input/output interfaces for your AI modules[4]
- **Custom Modules**: Build specialized components for your use case
- **Tool Integration**: Connect external APIs and services
- **Production Deployment**: Scale your DSPy applications

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Basic understanding of Python programming

### Installation

```bash
pip install dspy
```

To install the latest development version:

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```

### Quick Example

```python
import dspy

# Configure your language model
lm = dspy.LM('gpt-3.5-turbo')
dspy.configure(lm=lm)

# Define a signature
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Create a predictor
generate_answer = dspy.ChainOfThought(BasicQA)

# Use it
question = "What is the color of the sky?"
pred = generate_answer(question=question)
print(pred.answer)
```

## Learning Path

Follow the notebooks in sequence:

1. **Start with Basics**: Begin with `01_Intro to DSPy.ipynb` to understand core concepts
2. **Build Custom Solutions**: Move to `02_Custom Modules and Tool Calling.ipynb` for advanced techniques
3. **Deploy to Production**: Complete with `03_Deploying with Databricks.ipynb` for real-world applications

## Use Cases

DSPy is particularly effective for:
- **Information Extraction**: Extract structured data from unstructured text[5]
- **Question Answering**: Build intelligent Q&A systems[4]
- **RAG Pipelines**: Create retrieval-augmented generation systems[5]
- **Agent Development**: Build tool-using AI agents[5]
- **Code Generation**: Generate code from natural language descriptions[5]

## Resources

- **Official Documentation**: [dspy.ai](https://dspy.ai)[2]
- **GitHub Repository**: [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)[1]
- **Community**: Join the DSPy Discord for support and discussions[1]

## Contributing

This is an educational repository. If you find errors or have suggestions for improvements, please open an issue or submit a pull request.

## License

Please refer to the original DSPy project for licensing information.
