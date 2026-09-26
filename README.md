# Best Deal

> An AI-powered multi-agent system for product price estimation and intelligent deal detection.

Best Deal is an end-to-end AI/ML project designed to answer a simple but challenging question:

**Is this product actually a good deal?**

Instead of looking only at the current product price, Best Deal estimates the product's fair value using multiple AI/ML approaches, combines their predictions, and compares the estimated value with the current price.

The project combines **Machine Learning, Deep Learning, LLMs, RAG, and Multi-Agent Systems** into a single application.

---

## What Problem Does It Solve?

A low price does not necessarily mean a product is a good deal.

A product may appear discounted while still being expensive compared with similar products or its estimated market value.

Best Deal approaches the problem by estimating a product's expected value and measuring the difference between that value and its current price.

```text
Product
   |
   v
Product Information + Current Price
   |
   v
+-------------------------------+
|       Pricing Agents          |
|                               |
|  DNN   RAG   Fine-Tuned Model |
+-------------------------------+
   |
   v
Multiple Price Estimates
   |
   v
Ensemble / Regression
   |
   v
Estimated Fair Value
   |
   v
Deal Evaluation
   |
   v
Genuine Deal Candidate
```

---

## How Best Deal Works

The system combines several pricing approaches rather than relying on a single model.

### 1. Data Pipeline

The project starts with product data that goes through a structured data pipeline:

```text
Raw Data
   |
   v
Data Collection
   |
   v
Data Cleaning
   |
   v
Feature Engineering
   |
   v
Validation
   |
   v
Model-Ready Data
```

The research notebooks contain the exploratory analysis and experiments, while reusable data-processing logic is implemented inside the `src` package.

---

### 2. Machine Learning

Classical regression models were used as baseline approaches.

The ML workflow includes:

- Data preprocessing
- Feature engineering
- Baseline models
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Model comparison

These experiments establish measurable baselines before introducing more advanced models.

---

### 3. Deep Neural Network

A Deep Neural Network is used as one of the pricing agents.

The source code contains the model architecture and inference logic, while the trained checkpoint is kept outside Git because of its size.

```text
Product
   |
   v
DNN Inference
   |
   v
Estimated Price
```

---

### 4. Fine-Tuned Model

A language model was fine-tuned specifically for the product price-estimation task.

The training process was performed remotely.

The application does not retrain the model during normal execution.

Instead, the existing trained model is accessed through an integration layer:

```text
Specialist Agent
      |
      v
Remote Fine-Tuned Model
      |
      v
Estimated Price
```

---

### 5. RAG Pricing Agent

The RAG agent retrieves relevant product information before generating a price estimate.

The pipeline includes:

```text
Product Query
     |
     v
Query Rewriting
     |
     +----------------+
     |                |
     v                v
Original Query   Rewritten Query
     |                |
     +--------+-------+
              |
              v
       Hybrid Retrieval
        /            \
       v              v
  Dense Search      BM25
        \            /
         \          /
          v        v
        RRF Fusion
             |
             v
          Reranker
             |
             v
       Relevant Context
             |
             v
       Structured LLM
             |
             v
       Estimated Price
```

The RAG implementation is divided into independent modules for loading, splitting, embeddings, vector search, BM25, retrieval, reranking, generation, ingestion, and evaluation.

---

## Multi-Agent Architecture

The different pricing approaches are exposed through specialized agents.

```text
                 Agent Framework
                        |
                        v
             Autonomous Planning
                        |
                        v
                  Deal Scanner
                        |
                        v
                 Ensemble Agent
                  /     |      \
                 /      |       \
                v       v        v
              RAG      DNN    Specialist
               |        |         |
               v        v         v
            Estimate Estimate  Estimate
                 \       |       /
                  \      |      /
                   v     v     v
                    Ensemble
                       |
                       v
                 Deal Evaluation
                       |
                       v
                Messaging Agent
                       |
                       v
                    Pushover
```

The agent architecture separates:

- Model-specific inference
- Agent orchestration
- Deal evaluation
- Notification delivery

This allows the individual AI components to evolve independently.

---

## RAG Evaluation

The RAG system has a dedicated evaluation pipeline.

A frozen golden dataset is used to evaluate retrieval and pricing behavior.

```text
Golden Dataset
      |
      v
RAG Evaluation
      |
      +--> Retrieval Metrics
      |
      +--> Price Metrics
      |
      +--> Category Metrics
      |
      v
Aggregated Results
      |
      v
Evaluation Dashboard
```

The evaluation logic is separated from the Gradio dashboard so that it can later be reused by automated evaluation jobs, CI/CD, or an admin backend.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| LLM / RAG | LLM APIs, Chroma, BM25 |
| Fine-Tuning | Remote fine-tuning infrastructure |
| Agent Architecture | Custom Multi-Agent System |
| Interface | Gradio |
| Notifications | Pushover |
| Dataset Management | Hugging Face Datasets |
| Experiment Tracking | Weights & Biases |
| Version Control | Git / GitHub |
| Planned Backend | FastAPI |
| Planned Database | PostgreSQL / Supabase |
| Planned Frontend | Next.js / TypeScript |
| Planned Deployment | Docker / Azure |

---

## Project Structure

The application uses a Python `src` layout:

```text
src/
└── best_deal/
    ├── main.py

    ├── agents/
    │   ├── framework.py
    │   ├── scanner_agent.py
    │   ├── ensemble_agent.py
    │   ├── frontier_agent.py
    │   ├── specialist_agent.py
    │   ├── neural_network_agent.py
    │   └── messaging_agent.py

    ├── data/
    │   ├── models.py
    │   ├── collection.py
    │   ├── preprocessing.py
    │   ├── validation.py
    │   └── llm_batch.py

    ├── deep_learning/
    │   ├── model.py
    │   └── inference.py

    ├── ml/
    │   ├── ensemble/
    │   └── evaluation/

    ├── integrations/
    │   └── modal_service.py

    ├── rag/
    │   ├── loader.py
    │   ├── splitter.py
    │   ├── embeddings.py
    │   ├── vectorstore.py
    │   ├── bm25.py
    │   ├── retriever.py
    │   ├── reranker.py
    │   ├── llm.py
    │   ├── services/
    │   └── evaluation/

    ├── utils/
    │
    └── config.py
```

Research notebooks are kept separately from reusable application code.

---

## Research vs Application Code

The project intentionally separates experimentation from reusable runtime logic.

### Research notebooks

The notebooks contain:

- Exploratory Data Analysis
- Model experiments
- Cross-validation
- Hyperparameter tuning
- Neural network training
- Fine-tuning experiments
- Model comparisons
- Experimental visualizations

### Source package

The `src` package contains reusable logic required by the application:

- Data processing
- Data validation
- Model inference
- RAG
- Agent orchestration
- Ensemble logic
- Deal evaluation
- Notifications
- Evaluation pipelines
- Application startup

This separation prevents the application from depending on notebook execution.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd Best_Deal
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Depending on the components being used, the application may require credentials for:

- LLM providers
- Modal
- Pushover
- RAG services

Secrets are not committed to Git.

### 5. Run the application

```bash
python -m best_deal.main
```

The current interface is built with Gradio.

---

## Model Artifacts

Large model artifacts are intentionally excluded from Git.

For example:

```text
artifacts/
└── models/
    └── deep_neural_network.pth
```

The fine-tuned model is hosted remotely and accessed through its integration layer.

This keeps the repository lightweight and allows model artifacts to be moved later to dedicated cloud storage.

---

## Current Status

### Completed

- Data collection and preprocessing pipeline
- Classical ML experiments
- Deep Neural Network
- Fine-tuned pricing model
- RAG pricing pipeline
- RAG evaluation pipeline
- Multi-agent architecture
- Ensemble pricing
- Deal evaluation
- Pushover notification integration
- Gradio application
- Modular `src` architecture

### Current Development

The project is now moving from an AI/ML research application toward a production-oriented software system.

The next engineering layer includes:

- Backend API
- Persistent database
- User accounts
- User preferences
- Personalized deal selection
- Product watchlists
- Scheduled deal scanning
- Production notification management

---

## Roadmap

```text
AI / ML Research
       |
       v
Modular Source Architecture   <-- Current Stage
       |
       v
FastAPI
       |
       v
PostgreSQL / Supabase
       |
       +---- Users
       +---- Products
       +---- Price History
       +---- Deals
       +---- Watchlists
       +---- Notifications
       |
       v
Scheduled Workers / Cron
       |
       v
Next.js Dashboard
       |
       v
Docker
       |
       v
CI/CD
       |
       v
Azure Deployment
```

The architecture is intentionally being built incrementally so that the existing AI/ML components can be reused by the future backend, scheduled jobs, and user-facing applications.

---

## Documentation

More detailed technical documentation is available in the [`docs/`](docs/) directory.

- [Project Overview](docs/project_overview.md)
- [ML Pipeline](docs/ml_pipeline.md)
- [Experiments](docs/experiments.md)

Additional architecture and deployment documentation will be added as those stages are implemented.

---

## Engineering Focus

Best Deal is being developed as an end-to-end engineering project rather than only a machine-learning experiment.

The long-term workflow is:

```text
Data Engineering
      ↓
Machine Learning
      ↓
Deep Learning
      ↓
LLMs / RAG
      ↓
Multi-Agent Systems
      ↓
Software Architecture
      ↓
Backend Engineering
      ↓
Database Design
      ↓
Testing
      ↓
Docker
      ↓
CI/CD
      ↓
Cloud Deployment
```

The goal is to turn an AI research pipeline into a maintainable, testable, and deployable software product.

---

## Project Status

**Current stage:** AI/ML core → modular application architecture

Best Deal is actively under development.
