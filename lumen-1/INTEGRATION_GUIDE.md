# Integration Guide: AI Modes Sandbox & Critical Bug Fixes

This document details the modifications applied to the Lumen-1 travel assistant project to integrate the **6 AI Sandbox Modes** (Default, Analyst, Optimizer, Refactor, Database, Security) and apply critical bug fixes.

---

## 🔧 1. Integrated AI Sandbox Modes (Playground)

The dashboard playground has been enhanced with a robust multi-mode orchestration layer:

### 🎮 6 Conversation Modes
* **Default** (`/default`): Travel guide assistant.
* **Analyst** (`/analyst`): Analyzes telemetry loss graphs and SFT/DPO data profiles.
* **Optimizer** (`/optimizer`): Proposes hyperparameter schedules (LoRA Rank, Learning Rate).
* **Refactor** (`/refactor`): Conducts code review and resolves programming bugs.
* **Database** (`/database`): Handles JSON database queries and formats schema metadata.
* **Security** (`/security`): Safeguards against system prompt injection jailbreaks.

### 🌟 UI/UX Components added:
1. **Mode Switcher Bar**: Glassmorphic pills positioned at the top of the chat area to switch active modes instantly.
2. **Left Config Sidebar**: Contains a **System Prompt Preview** text box and **Quick Prompts** buttons that dynamically adjust based on the selected mode.
3. **Slash Commands Autocomplete**: Typing `/` inside the prompt textarea triggers a dropdown menu displaying all available modes. Users can navigate using **Arrow Keys / Enter / Escape** or click to select, autocompleting and switching modes.
4. **Ctrl+Enter Shortcut**: Enables instant sandbox generation submission.
5. **Mode Badge Tags**: Prepended to assistant responses in the chat thread, matching the corresponding mode theme colors (Grey, Blue, Teal, Gold, Purple, Red).

---

## 🛠️ 2. Applied Code Modifications

### 🐍 Backend Changes
* **`scripts/test_tripme_ai.py`**: Added mode parameter parsing. Reads `sys.argv[3]` for selected mode and maps it to specific system instructions templates:
  * `/default`: General Sri Lankan travel advisory.
  * `/analyst`: Telemetry validation.
  * `/optimizer`: Hyperparameter optimizer.
  * `/refactor`: Code review & bug refactoring.
  * `/database`: GeoJSON queries.
  * `/security`: Safety profiling.
* **`dashboard/app.py`**: Updated `TestRequest` schema and FastAPI endpoints (`/api/test-model`, `/api/test-audio-model`) to support `mode` argument. The backend calls the Python execution subprocess with `[sys.executable, "test_tripme_ai.py", prompt_arg, "0.7", mode]`.

### 🌐 Frontend Changes
* **`dashboard/static/index.html`**: Structured updated layout containing `.mode-switcher-bar`, `.playground-sidebar`, `.textarea-relative-wrapper`, `.slash-dropdown`, and `#current-mode-badge`.
* **`dashboard/static/style.css`**: Styles for the 4-column playground layout, active state buttons, neon colors, and mode badge tags.
* **`dashboard/static/main.js`**: JavaScript logic for mode states (`activeMode`), autocomplete filtering, input caret tracking, arrow key navigation, Ctrl+Enter listeners, and chat bubble tags.

---

## 🛡️ 3. Critical Training Bug Fixes

### 🐞 SFT dataset_text_field Mismatch Fix
* **Problem**: `SFTTrainer` crashed during dataset loading because the default configuration parsed for a `"text"` column, whereas `sft.jsonl` contains conversational format (`"messages"` list).
* **Fix**: Set `dataset_text_field: "messages"` in `configs/sft.yaml` and updated `scripts/train_mistral_fast.py` to fetch `dataset_text_field` dynamically from the SFT configuration (defaulting to `"messages"`).

### 💀 DPO Alignment Skeleton Generator
* **Fix**: Added `scripts/data_gathering/create_dpo_skeleton.py` to quickly compile the initial chosen/rejected preference pairs (including "Nuwara Wewa" Anuradhapura disambiguation samples) to activate the DPO training route.

---

## ⚡ Next Steps for Execution

Run these scripts on your local system terminal to initialize database training files:

1. **Initialize DPO Preferences Skeleton**:
   ```powershell
   python scripts/data_gathering/create_dpo_skeleton.py
   ```
2. **(Optional) Run full DPO compiler**:
   ```powershell
   python scripts/data_gathering/generate_dpo_data.py
   ```
3. **Start Dashboard Server**:
   ```powershell
   python dashboard/app.py
   ```
   Open `http://127.0.0.1:8000` in your web browser and click on the **Playground** tab to test the modes!
4. **Test standalone Sandbox demo**:
   Double click the `playground.html` file in the project folder to open and try it out directly.
