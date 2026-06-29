---
name: lumen-workspace-orchestrator
description: Guidelines for managing the Lumen-1 multimodal foundation model pipeline, including launching training (SFT/DPO), configuring speculative decoding, querying the Chroma vector database, and resolving virtual environment or dashboard runtime errors.
---

# Lumen-1 Workspace Orchestration & Management Guide

This skill provides workspace-specific context and commands for operating, debugging, and expanding the Lumen-1 model dashboard and backend infrastructure.

---

## 1. Dashboard Management & Running
The control panel is a FastAPI application situated in `dashboard/app.py`.
- **Command to Run**:
  ```powershell
  uvicorn dashboard.app:app --host 127.0.0.1 --port 8000 --reload
  ```
- **Port**: `8000` (Static files served from `dashboard/static/`).
- **Dependencies**: Uses `websockets` for streaming telemetry, `GPUtil` for NVIDIA card analytics, and standard web dependencies.

---

## 2. Virtual Environment & Dependency Installation
If you encounter unresolved imports (e.g., `fastapi`, `torch`, `transformers`, `GPUtil`), verify that you are active in the local virtual environment `.venv`:
1. **Activation (PowerShell)**:
   ```powershell
   .\.venv\Scripts\activate
   ```
2. **Installation**:
   ```powershell
   pip install -e .\lumen-1[train,eval,dev,inference,data]
   pip install GPUtil websockets langchain langchain-community chromadb pypdf SpeechRecognition sentence-transformers psutil
   ```

---

## 3. Training & Dataset Pipeline
Lumen-1 supports SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).
- **SFT Script**: `scripts/train_mistral_fast.py` (Outputs logs/checkpoints to `checkpoints/lumen_mistral_finetuned`).
- **DPO Script**: `scripts/align.py` (Outputs logs/checkpoints to `checkpoints/lumen_mistral_dpo`).
- **RAG Vector Storage**: chroma database is located at `lumen-1/chroma_db` and uses HuggingFace `all-MiniLM-L6-v2` embeddings. To set up, run `scripts/rag_setup.py`.

---

## 4. Inference & Speculative Decoding
- **Speculative decoding**: Speeds up target model generation (e.g., 7B model) by generating candidate tokens with a draft model (e.g., 1B model).
- The draft model config is initialized in `lumen/inference/speculative.py`.
- **Agent Mode**: Utilizes an execution loop in `lumen/inference/agent.py` supporting `calculator`, `search` (Wikipedia), and `local_travel_db` (Chroma RAG search) tools.
