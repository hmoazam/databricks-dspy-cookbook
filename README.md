# Databricks DSPy Cookbook

A comprehensive collection of tutorials, examples, and production-ready templates for building DSPy applications on the Databricks platform. This cookbook demonstrates how to leverage [DSPy](https://dspy.ai/) - a framework for programming language models - with Databricks features like Vector Search, MLflow, Mosaic AI Agent Framework, and Foundation Model APIs.

## 📚 Overview

DSPy (Declarative Self-improving Python) is a framework developed at Stanford that enables developers to build modular AI systems by writing compositional Python code instead of brittle prompts. This cookbook shows how to integrate DSPy with Databricks to create production-grade AI applications with built-in optimization, evaluation, and deployment capabilities.

## 🎯 What's Inside

### 1. **DSPy Hackathon** (`dspy_hackathon/`)

Progressive learning materials covering DSPy fundamentals through advanced concepts:

- **`01_Intro to DSPy.py`** - Introduction to DSPy fundamentals
  - Inline and class-based signatures
  - Type safety and structured outputs
  - Core modules (Predict, ChainOfThought)
  - Iterative development patterns
  - Multi-modal capabilities

- **`02_Custom Modules and Tool Calling.py`** - Building custom modules and integrating tools
  - Creating custom DSPy modules
  - Tool integration patterns
  - Advanced signature design

- **`03_Advanced_DSPy_Optimizer_GEPA.py`** - Advanced optimization techniques
  - GEPA (Genetic-Pareto) optimization
  - Prompt optimization strategies

- **`04_Advanced_DSPy_Databricks.py`** - Databricks-specific features
  - Integration with Databricks Foundation Model APIs
  - MLflow integration and logging

- **`05_Hack!.py`** - Hands-on exercises and challenges

**Resources:** Includes sample data (`financial.csv`) and reference materials

### 2. **RFI Agent** (`rfi_agent/`)

A complete end-to-end example of building, optimizing, and deploying a RAG (Retrieval-Augmented Generation) agent using DSPy on Databrick for responding to RFIs based on documentation.

#### Main Notebooks
- **`01_dspy_without_opt_rag_agent.ipynb`** - Basic RAG agent implementation
- **`02_dspy_with_opt_rag_agent.ipynb`** - Optimized RAG agent with DSPy optimizers
- **`03_register_and_deploy.ipynb`** - Model registration and deployment using Databricks Agent Framework

#### Setup Notebooks (`setup/`)
1. `01_deploy_models.ipynb` - Deploy required foundation models
2. `02_create_vector_search.ipynb` - Set up Vector Search endpoints
3. `03_create_volume.ipynb` - Create Unity Catalog volumes
4. `04_prep_rag_data.ipynb` - Prepare and process RAG data
5. `05_setup_vector_search.ipynb` - Configure Vector Search indices

#### Data
- Databricks documentation chunks (AWS, Azure, GCP) in JSONL format
- Sample Q&A datasets for evaluation

#### Configuration
- `config.yaml` - Centralized configuration for catalog, schema, endpoints, and secrets

### 3. **DSPy Genie Multi-Agent** (`dspy_genie/`)

Demonstrates building sophisticated multi-agent systems using DSPy and Databricks Genie:

- **`DSPy Multiagent Genie.ipynb`** - Complete multi-agent implementation
  - Integration with Databricks Genie for natural language SQL queries
  - Multi-agent coordination patterns
  - MLflow ChatAgent wrapping for Databricks compatibility
  - Agent deployment and testing

**Use Case:** Build multi-agent systems where Genie handles structured data queries while other agents handle different specialized tasks.

### 4. **DSPy Spark Integration** (`dspy_spark/`)

Learn how to scale DSPy applications using Apache Spark for batch processing:

- **`dspy-spark-udf.py`** - DSPy with Spark UDFs
  - Two approaches for DSPy caching with Spark
  - Best practices for distributed DSPy execution
  - Serverless compute compatibility

### 5. **Text Classification with GEPA** (`text_classification_with_GEPA_optimization/`)

Production example of prompt optimization for classification tasks:

- **`dspy-classifier-GEPA-optimization.py`**
  - Text classification using PubMed dataset
  - GEPA (Genetic-Pareto) optimizer implementation
  - Multi-objective optimization (accuracy and latency)
  - Best practices for production classification pipelines

## 🚀 Getting Started

### Prerequisites

- Databricks workspace with access to:
  - Unity Catalog
  - Vector Search
  - Foundation Model APIs or Model Serving endpoints
  - MLflow
- Python 3.9+
- Personal Access Token (PAT) for API access

### Installation

```bash
# Install DSPy
pip install dspy

# Install Databricks integrations
pip install "mlflow[databricks]>=3.1.0" databricks-sdk databricks-agents

# For development version
pip install git+https://github.com/stanfordnlp/dspy.git
```

### Quick Start

1. **Set up Databricks secrets:**
   ```python
   # Create a secret scope for your PAT token
   dbutils.secrets.createScope("my-scope")
   ```

2. **Configure your environment:**
   - For RFI Agent: Fill out `rfi_agent/config.yaml`
   - Set up compute: Use general purpose instances (e.g., m5d.large for development)

3. **Start with the Hackathon:**
   - Begin with `dspy_hackathon/01_Intro to DSPy.py`
   - Follow the numbered sequence for progressive learning

4. **Build your first RAG agent:**
   - Run setup notebooks in `rfi_agent/setup/`
   - Follow `01_dspy_without_opt_rag_agent.ipynb`
   - Optimize with `02_dspy_with_opt_rag_agent.ipynb`

## 💡 Key Concepts

### DSPy Core Modules

- **Predict**: Basic predictor for LLM inference
- **ChainOfThought**: Adds reasoning steps before answering
- **ReAct**: Agent pattern with tool use capabilities
- **Retrieve**: Handles retrieval for RAG applications
- **ProgramOfThought**: Generates and executes code for complex reasoning

### DSPy Signatures

Define input/output interfaces for your AI modules with:
- Type safety and validation
- Structured outputs
- Reusable components
- Automatic prompt generation

### DSPy Optimizers

Automatically improve prompts and model performance:
- **BootstrapFewShot**: Learn from examples
- **MIPRO**: Multi-prompt instruction optimization
- **GEPA**: Genetic-Pareto multi-objective optimization

## 🏗️ Architecture Patterns

### RAG Pattern
```python
import dspy

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```

### Multi-Agent Pattern
- Specialized agents for different capabilities
- Coordinator agent for routing and orchestration
- Integration with Databricks Genie for SQL tasks

### Production Deployment
1. Develop and test DSPy modules locally
2. Optimize using DSPy optimizers
3. Log to MLflow with automatic tracing
4. Register in Unity Catalog
5. Deploy using Databricks Agent Framework
6. Evaluate using Mosaic Agent Eval

## 📊 Use Cases

This cookbook demonstrates DSPy for:

- **Question Answering**: RAG systems over enterprise knowledge bases
- **Text Classification**: Optimized classification with GEPA
- **Multi-Agent Systems**: Coordinated agents with specialized capabilities
- **SQL Generation**: Natural language to SQL with Genie integration
- **Large-Scale Processing**: Distributed DSPy with Spark UDFs
- **Agent Deployment**: Production-ready deployment on Databricks

## 🔧 Databricks Integration Features

- **Vector Search**: High-performance similarity search for RAG
- **MLflow Integration**: Automatic tracing and logging with `mlflow.dspy.autolog()`
- **Foundation Model APIs**: Access to state-of-the-art models (Llama, Claude, etc.)
- **Unity Catalog**: Centralized governance and model registry
- **Agent Framework**: Production deployment and serving
- **Mosaic Agent Eval**: Automated agent evaluation and quality metrics

## 📖 Learning Path

**Beginner:**
1. Start with `dspy_hackathon/01_Intro to DSPy.py`
2. Complete the TODO exercises in each hackathon notebook
3. Experiment with different modules and signatures

**Intermediate:**
1. Build the RFI Agent following the setup and main notebooks
2. Explore DSPy optimizers with the classification example
3. Learn Spark integration for scaling

**Advanced:**
1. Build multi-agent systems with Genie
2. Implement custom DSPy modules and optimizers
3. Deploy production systems with full MLOps pipeline

## 🤝 Contributing

This is an educational repository. Contributions are welcome:
- Bug fixes and improvements
- Additional examples and use cases
- Documentation enhancements
- Performance optimizations

## 📚 Additional Resources

- **DSPy Documentation**: [dspy.ai](https://dspy.ai)
- **DSPy GitHub**: [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **Databricks AI**: [Databricks Generative AI](https://www.databricks.com/product/machine-learning/generative-ai)
- **Agent Framework Docs**: 
  - [AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/)
  - [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/)

## 🙏 Acknowledgments

Built with:
- [DSPy](https://dspy.ai) by Stanford NLP
- [Databricks](https://databricks.com) platform and tooling
- Community contributions and feedback

## 📝 License

Please refer to the original DSPy project and Databricks terms for licensing information.

---

**Note**: This cookbook uses Databricks-specific features. Some examples require access to Databricks workspace services like Vector Search, Foundation Model APIs, and Unity Catalog.

