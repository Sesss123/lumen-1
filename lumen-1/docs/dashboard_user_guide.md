# Lumen-1 Admin Control Panel User & Testing Guide

Welcome to the **Lumen-1 Admin Control Panel**. This guide outlines all features built into the interactive dashboard, how they work behind the scenes, and step-by-step test instructions for each.

---

## 💻 1. Real-time Telemetry & Diagnostics (Overview & System Health)
- **What it does**: Streams active hardware statistics (CPU, RAM, GPU utilization, temperature, VRAM) and server metrics using a persistent WebSocket connection.
- **How to test**:
  1. Open the dashboard.
  2. Navigate to the **System Health** tab.
  3. Observe the live charts and bars updating every 1.5 seconds without reload.
  4. Perform some local activity (like opening another program or running an inference) to watch the resource levels dynamically change.

---

## ⚙️ 2. Hyperparameter Configuration Editor (Configuration)
- **What it does**: Provides a visual settings panel to modify hyperparameters (like learning rate, batch size, epochs, LoRA rank/alpha) inside `configs/sft.yaml` and `configs/dpo.yaml`.
- **How to test**:
  1. Go to the **Configuration** tab.
  2. Switch between **SFT Tuning Config** and **DPO Alignment Config** tabs.
  3. Modify a hyperparameter value (e.g., change `learning_rate` or toggle `use_lora`).
  4. Click the **Save Config** button.
  5. Open `lumen-1/configs/sft.yaml` or `dpo.yaml` to confirm the value was successfully written.

---

## 📁 3. CSV Dataset Studio & Validator (CSV Dataset Studio)
- **What it does**: Allows drag-and-drop CSV dataset uploads, maps columns (Instruction, Input, Response), and runs quality validation checks (Sinhala text checking, average word count, safety violations).
- **How to test**:
  1. Navigate to **CSV Dataset Studio**.
  2. Drag and drop `lumen-1/data/test_dataset.csv` or select it manually.
  3. Map the column fields (e.g., map Instruction, Input, and Response headers).
  4. Click **Validate Dataset**.
  5. Check the parsed stats (Passed rows, safety warnings, and visual word count distribution graphs).
  6. Click **Apply to SFT Dataset** to merge it.

---

## 🧪 4. Interactive Playground Sandbox (Playground)
- **What it does**: The central chat interface supporting voice recordings, image attachments, speculative decoding, and custom conversational modes.
- **How to test**:
  - **Custom Modes**: Switch between Default, Analyst, Optimizer, Database, and Security modes. Note how the system prompt updates.
  - **Voice Transcription (Speech-to-Text)**: Click the microphone icon, speak into it, and see your recording transcribed into Sinhala/English automatically.
  - **Speculative Decoding Telemetry**: 
    1. Toggle **Use Speculative Decoding** at the bottom.
    2. Submit a query (with or without video/audio attachments).
    3. Look at the sidebar telemetry box to verify acceptance rate, generation speeds (Tokens/sec), and relative speedup multiplier.
  - **Markdown Export**: Have a quick conversation, then click the **Export Chat** button in the top-right of the chat box to download a formatted Markdown log.

---

## 🤖 5. Agent Mode & Autonomous Tools (Playground)
- **What it does**: Leverages the agentic loop (`lumen/inference/agent.py`) using tools (Calculator, Wikipedia search, Local ChromaDB Travel Database, Weather forecast, Travel distance route calculator, and Currency converter) to resolve queries.
- **How to test**:
  1. Select the **Agent** mode on the playground switcher.
  2. **Test Travel Database**: Ask: *"ශ්‍රී ලංකාවේ Camping කරන්න හොඳම තැන් මොනවාද?"* (queries local travel database).
  3. **Test Calculator**: Ask: *"calculate 450 * 3.14"*.
  4. **Test Live Weather**: Ask: *"Colombo weather"* or *"Sigiriya current weather"* (queries Open-Meteo API).
  5. **Test Distance & Route**: Ask: *"distance from Colombo to Ella"* or *"travel duration between Kandy and Sigiriya"* (queries routing matrix).
  6. **Test Currency Converter**: Ask: *"convert 150 USD to LKR"* or *"what is 50 EUR in Sri Lankan rupees?"* (queries currency converter tool).
  7. Verify that intermediate nested boxes appear detailing the agent's step-by-step thoughts, tool executions, and the tools' returned data before printing the final answer.

---

## 📈 6. Live Training Mock Convergence (Overview Header)
- **What it does**: Simulates active SFT/DPO training convergence metrics so you can verify the dashboard chart behavior without waiting for a multi-hour model training run.
- **How to test**:
  1. Go to the **Overview** tab.
  2. Click the yellow **"Mock Metric"** button in the header repeatedly.
  3. Observe the **Training Loss**, **Learning Rate**, and **Epoch Progress** cards update with decaying values.
  4. Watch the **Live Training Convergence** line chart draw and animate step-by-step loss reduction in real-time.

---

## 📊 7. Diagnostics Exporter (Overview Header)
- **What it does**: Downloads a full system snapshot JSON detailing CPU specifications, GPU models, metrics history, and model configuration metadata.
- **How to test**:
  1. Click the green **"Export Telemetry"** button in the header.
  2. Confirm that a file named `lumen_telemetry_export.json` downloads successfully to your local machine.
