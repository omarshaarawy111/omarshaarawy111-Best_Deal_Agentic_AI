# Best Deal

Best Deal is an AI/ML multi-agent deal detection and price-estimation project. The current repository stage converts the reusable parts of the research notebooks into a clean `src` package while keeping expensive research and training artifacts outside the application startup path.

> **Current entry point:** `src/best_deal/main.py`  
> The filename is intentionally kept for continuity with the existing Gradio application, while the project name is **Best Deal**.

## 1. What this migration does

The migration follows one rule:

> **A notebook records experiments and discoveries. `src` contains reusable behavior required by the system.**

That means the project does **not** convert every notebook into a Python file. Reusable pipeline code is extracted, while experiments, EDA, charts, one-off comparisons, and completed training runs remain in notebooks.

### Source layout

```text
src/
└── best_deal/
    ├── __init__.py
    ├── config.py
    ├── main.py              # Gradio application entry point
    │
    ├── agents/
    │   ├── agent.py                   # shared agent behavior and logging
    │   ├── deals.py                   # deal scraping + domain models
    │   ├── preprocessor.py            # lightweight LLM preprocessing
    │   ├── scanner_agent.py            # deal discovery and selection
    │   ├── frontier_agent.py           # thin RAG adapter
    │   ├── specialist_agent.py         # thin Modal fine-tuned-model adapter
    │   ├── neural_network_agent.py     # thin DNN inference adapter
    │   ├── ensemble_agent.py            # specialist orchestration
    │   ├── messaging_agent.py           # Pushover notifications
    │   ├── planning_agent.py            # deterministic planning baseline
    │   ├── autonomous_planning_agent.py # current tool-calling planner
    │   └── framework.py                 # runtime lifecycle + memory
    │
    ├── data/
    │   ├── models.py                   # Item data model
    │   ├── collection.py               # Hugging Face collection
    │   ├── preprocessing.py            # cleaning + feature engineering
    │   ├── validation.py               # reusable data validation
    │   └── llm_batch.py                # explicit OpenAI batch workflow
    │
    ├── deep_learning/
    │   ├── model.py                    # DNN architecture only
    │   └── inference.py                # checkpoint loading + inference
    │
    ├── ml/
    │   ├── ensemble/
    │   │   ├── ensemble.py             # weighted model combination
    │   │   └── business_metrics.py     # deal gap/business calculations
    │   └── evaluation/
    │       └── metrics.py              # reusable pure ML metrics
    │
    ├── integrations/
    │   └── modal_service.py             # remote fine-tuned model adapter
    │
    ├── rag/
    │   ├── config.py
    │   ├── schemas.py
    │   ├── prompts.py
    │   ├── loader.py
    │   ├── splitter.py
    │   ├── embeddings.py
    │   ├── vectorstore.py
    │   ├── bm25.py
    │   ├── retriever.py
    │   ├── reranker.py
    │   ├── llm.py
    │   ├── services/
    │   │   ├── answer.py                # complete answer pipeline
    │   │   └── ingest.py                # explicit RAG index build
    │   └── evaluation/
    │       ├── metrics.py               # retrieval + price metrics
    │       ├── evaluator.py             # everything before the dashboard
    │       └── dashboard.py             # Gradio UI only
    │
    └── utils/
        ├── paths.py
        └── logging.py
```

## 2. Why `data` is split by responsibility

The data layer is **not** one giant `data.py` file and it is also **not** split into one file per notebook cell.

The boundary is the responsibility:

```text
collection.py       -> obtain source data
preprocessing.py    -> deterministic cleaning + feature engineering
validation.py       -> validate data before downstream use
models.py           -> stable product representation
llm_batch.py        -> expensive batch preparation/execution
```

EDA does not become `eda.py` just because an EDA notebook exists. Charts, distributions, correlation checks, and exploratory observations are research artifacts unless they become part of a real production data-quality or monitoring requirement.

## 3. RAG architecture

RAG is intentionally more granular because the pipeline already contains distinct responsibilities:

```text
Question
   |
   v
get_query
   |
   v
query rewriting -----------+
   |                        |
   +---- original ----------+---- rewritten
             |                       |
             v                       v
        HybridRetriever        HybridRetriever
        (dense + BM25)         (dense + BM25)
             \                       /
              +------ merge --------+
                         |
                         v
                     reranker
                         |
                         v
                   top-k context
                         |
                         v
                 structured LLM
                         |
                         v
                    RAGAnswer
```

### RAG module responsibilities

`embeddings.py` handles embeddings only.

`vectorstore.py` handles Chroma persistence and dense search only.

`bm25.py` handles sparse retrieval only.

`retriever.py` combines dense and sparse retrieval with Reciprocal Rank Fusion.

`reranker.py` handles the ranking stage.

`llm.py` handles query rewriting, reranking structured output, and price estimation.

`services/answer.py` is the orchestration layer that calls those components in the correct sequence.

`services/ingest.py` is an explicit build operation. It is **not** run when the Gradio app starts.

This separation is important for future FastAPI, Cron, Docker, and Azure deployment because those components can reuse `RAGAnswerService` without importing Gradio.

## 4. RAG evaluation architecture

The evaluation notebook is intentionally **not** converted into one giant function.

Everything that happens before the dashboard is moved into `RAGEvaluator`:

```text
Load frozen golden dataset
        |
Validate dataset integrity
        |
Evaluate one case
        |
Run RAGAnswerService
        |
Calculate retrieval metrics
        |
Calculate price metrics
        |
Checkpoint every case
        |
Resume after interruption
        |
Aggregate overall/category metrics
        |
Persist CSV
        |
        v
   dashboard.py
        |
        v
      Gradio
```

This gives us two clean entry points later:

```python
from best_deal.rag.evaluation.evaluator import RAGEvaluator

RAGEvaluator().run()
```

and, separately:

```python
from best_deal.rag.evaluation.dashboard import launch_dashboard

launch_dashboard()
```

The second one is UI. The first one can later be called from CI, a scheduled evaluation job, or an admin backend.

## 5. What stays in notebooks

### Stays as research/offline work

- EDA and exploratory plots
- Baseline model training
- Cross-validation experiments
- Hyperparameter tuning experiments
- Model comparison experiments
- Neural-network training
- Fine-tuning training
- Fine-tuning cloud execution
- One-off experiment visualizations
- Embedding visualization such as t-SNE

### Converted into reusable source modules

- Data collection logic
- Deterministic data preprocessing and feature engineering
- Data validation
- LLM batch preparation helpers
- DNN architecture needed for inference
- DNN inference
- Modal remote-model adapter
- Agent implementations
- Ensemble combination
- RAG ingestion
- RAG retrieval
- RAG reranking
- RAG final answer service
- RAG evaluation execution and metrics
- Gradio application orchestration

## 6. Fine-tuned model decision

The fine-tuned model is **not retrained as part of this migration**.

Training is already completed remotely. Rewriting the project from notebook code into `src` does not justify spending the training cost again.

The source stage therefore contains only:

```text
agents/specialist_agent.py
        |
        v
integrations/modal_service.py
        |
        v
existing remote Modal Pricer
```

The training notebook remains the historical record of how the model was produced.

## 7. DNN artifact decision

The DNN inference architecture is source code, but the checkpoint is not committed:

```text
src/best_deal/deep_learning/model.py
src/best_deal/deep_learning/inference.py

external artifact:
artifacts/models/deep_neural_network.pth
```

The `.pth` file is ignored by Git because it is a large model artifact.

The path can be supplied with:

```text
BEST_DEAL_DNN_WEIGHTS=artifacts/models/deep_neural_network.pth
```

This also makes the project suitable for later Docker/Azure artifact storage instead of baking a large checkpoint into Git history.

## 8. No machine-specific paths

Paths from notebooks such as local Windows paths, notebook-relative path hacks, or `/kaggle/...` paths are not used in source.

The project root is resolved from the package location, and environment variables can override artifact locations.

Use:

```python
from best_deal.config import PROJECT_ROOT
```

instead of hardcoding a developer's local directory.

## 9. Absolute imports only

The source package intentionally uses imports like:

```python
from best_deal.rag.services.answer import RAGAnswerService
from best_deal.agents.framework import DealAgentFramework
```

It does not use:

```python
sys.path.insert(...)
from answer import ...
from agents... import ...
```

That is deliberate. Absolute package imports are the cleaner base for future Cron jobs, workers, Docker images, FastAPI processes, and Azure deployment.

Install the package in editable mode during development:

```bash
pip install -e .
```

Run the application as a module:

```bash
python -m best_deal.main
```

## 10. Runtime behavior

The Gradio entry point is still the main demo/application entry point:

```text
main.py
      |
      v
DealAgentFramework
      |
      v
AutonomousPlanningAgent
      |
      +--> ScannerAgent
      |
      +--> EnsembleAgent
      |       +--> Preprocessor
      |       +--> FrontierAgent -> RAGAnswerService
      |       +--> SpecialistAgent -> Modal
      |       +--> NeuralNetworkAgent -> .pth artifact
      |
      +--> MessagingAgent -> Pushover
```

The current five-minute Gradio timer is preserved.

RAG ingestion, RAG evaluation, model training, and fine-tuning are **not** accidentally triggered by application startup.

## 11. Existing logic that was corrected during migration

A source migration is not just a copy/paste operation. The following issues were cleaned while preserving the intended workflow:

- `NeuralNetworkAgent` no longer imports a nonexistent `agents.deep_neural_network` module. It uses the actual DNN inference module.
- `FrontierAgent` no longer duplicates the complete RAG implementation. It delegates to `RAGAnswerService`.
- RAG duplicate notebook definitions are consolidated into single source implementations.
- RAG dense and sparse retrieval now use the same stable product identifier so RRF fusion can actually match results from both stores.
- Autonomous planner messages are initialized per run instead of relying on accidental instance state.
- Logging handlers used by the Gradio UI are removed after a run to avoid accumulating duplicate handlers.
- Machine-specific filesystem paths are removed.

## 12. Git policy

The repository does not commit:

- `.env` and secrets
- Local SQLite/database files
- runtime memory
- large `.pth`, `.pt`, `.ckpt`, `.bin`, `.safetensors`, and `.onnx` model artifacts
- pickled caches
- Chroma/BM25 generated indexes
- W&B experiment directories
- temporary batch outputs
- notebook checkpoints

`.env.example` contains placeholders only.

## 13. Stage 1 migration map

| Original notebook/file | Source destination | Decision |
| --- | --- | --- |
| `01_data_collection.ipynb` | `data/collection.py` | Reusable collection logic extracted |
| `02_data_preprocessing.ipynb` | `data/preprocessing.py` | Cleaning/features extracted, EDA stays notebook |
| `03_llm_batch_preparation.ipynb` | `data/llm_batch.py` | Explicit batch helpers extracted |
| `04_fine_tuning_preprossing.ipynb` | notebook | Historical training preparation, no runtime startup use |
| ML baseline notebooks | notebooks | Training/experiments stay offline |
| NN/DNN notebooks | `deep_learning/model.py` + `inference.py` | Inference architecture extracted, checkpoint external |
| fine-tuning notebooks | `integrations/modal_service.py` + `specialist_agent.py` | Remote runtime adapter only |
| `01_RAG_answer.ipynb` | `rag/*` + `rag/services/answer.py` | Full reusable RAG flow extracted |
| `05_RAG_ingestion.ipynb` | `rag/services/ingest.py` + supporting modules | Explicit build workflow extracted, EDA stays notebook |
| `02_RAG_evaluation.ipynb` | `rag/evaluation/evaluator.py` + `dashboard.py` | Pre-dashboard execution separated from UI |
| existing agent source | `agents/*` | Converted to package modules with absolute imports |
| existing `main.py` | `best_deal/main.py` | Remains application entry point |

## 14. Validation strategy for this stage

This stage is validated in layers:

1. Python syntax compilation across the whole `src` tree.
2. Absolute-import and machine-path scan.
3. Module/function docstring checks.
4. Pure-function smoke tests for ensemble and metrics.
5. Full runtime verification in the project's real environment, where OpenAI, Chroma, LiteLLM, BM25, Modal, the DNN checkpoint, and secrets are available.

Because the source stage deliberately excludes the large model checkpoint and generated RAG indexes, a completely fresh environment cannot reproduce the full live pipeline until those external artifacts and credentials are supplied.

## 15. Current scope versus later stages

### Stage 1, this migration

`src` package + clean modules + Gradio entry point + reusable RAG + source-level evaluation + configuration + packaging.

### Later stages

FastAPI, PostgreSQL/Supabase, authentication and RBAC, user preferences, watchlists, scheduled jobs, production notification routing, Next.js/TypeScript, admin dashboard, Docker, CI/CD, Azure deployment, tests, observability, and database-backed repositories.

The architecture in this stage is intentionally prepared for those additions without putting those technologies into the codebase prematurely.
