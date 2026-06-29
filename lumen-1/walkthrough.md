# Walkthrough: Dashboard Enhancements & Real-time Upgrades

We have successfully implemented the persistent loader, WebSockets, Agent mode, speculative decoding speed comparisons, persistent custom modes, and the new dashboard enhancements.

Below is a walkthrough of the latest advanced pipeline controls added.

---

## Latest Pipeline Controls Implemented

### 1. QLoRA weight Adapter Exporter Tab & Quantization Formats (GGUF / AWQ)
- **Tab Layout & Configuration**: Created a dedicated **Model Exporter** navigation tab in the control panel to configure Model weights merging parameter variables (Base model name/path, LoRA adapter path, merged model target output directory, and export format).
- **Offline Quantization Formats**: Added options to export models as **AWQ 4-bit Quantized (.pt)** and **GGUF 4-bit (.gguf)** files, making merged models ready for mobile/edge offline deployment.
- **Dynamic Conversion Script**: Added `scripts/convert_gguf.py` which merges LoRA checkpoints, automatically downloads the standard `convert_hf_to_gguf.py` conversion script from the llama.cpp repository, installs `gguf` python package dependencies, converts HF weights, and saves the final `.gguf` quantized file.
- **Console Log Streamer**: Features real-time log capturing and terminal style log rendering with color coding (errors in red, success in green, warnings in gold) inside the exporter dashboard block.

### 2. Interactive Database Spot Density Heatmap
- **Sri Lankan District Overlay Mapping**: Coordinates for all 25 districts of Sri Lanka mapped into Leaflet.
- **Concentration Circles**: Renders beautiful translucent glowing circle overlays on the playground interactive map, scaled in radius (`5000` to `35000` meters) and color-coded based on count density (Teal for low, Gold/Orange for moderate, Red for high concentration).
- **Tooltips**: Features custom Leaflet sticky tooltip hover boxes describing exact spot concentration statistics.

### 3. DPO Duplicate Prevention Warning & Diagnostics Chart
- **Similarity Comparison**: Configured an endpoint `/api/dpo/check-similarity` on the FastAPI server that loads preference pairs from `dpo_data.jsonl` and computes similarity via Python's `difflib.SequenceMatcher`.
- **Dynamic Warning Banner**: Displays an animated warning alert box above the DPO prompt compiler text area if the user types a prompt with similarity exceeding 65%, highlighting the exact matching percentage and the text of the existing record.
- **DPO Bias Chart & Metrics**: Added a horizontal dual-bar chart in the Dataset Compiler tab that compares Chosen vs Rejected response length averages to warn the user if a verbosity bias (> 1.5 ratio) is present.

### 4. Dynamic Prompt Template Previewer
- **Multi-Template Support**: Toggle between Alpaca (Instruction/Input/Response), ChatML (standard chat im_start/im_end tags), and ShareGPT JSON turns directly inside the Dataset Explorer tab.
- **Estimated Token Metrics**: Evaluates approximate token count dynamically (`Math.ceil(chars / 4)`) and renders an interactive sequence space occupancy bar based on a standard 2048 sequence limit.

### 5. Multi-Turn Interactive Chat-to-SFT Dataset Exporter
- **Chat Appender Button**: Added a secondary "Add Chat to SFT" button under the inference playground chat interface.
- **ChatML Serializer**: Formats the current sandbox conversation turns (including system prompts) into standard ChatML JSONL formats and appends them to `data/sft.jsonl` via `POST /api/dataset/append-chat`.
- **Dynamic Statistics Update**: Instantly triggers a dataset statistic reload in the Dataset Explorer tab after adding a chat, updating the total SFT records counter without needing to refresh the page.

### 6. Sri Lanka "Safe Travel" Smart Route Simulator
- **District Connectivity Graph**: Programmed a 25-node topological adjacency list representation of Sri Lanka's districts on the client side.
- **BFS Shortest Path Finder**: Computes the shortest path of connected districts between any start and end pins selected on the map.
- **Glowing Safe Route Polyline**: Draws a glowing dashed line on the map connecting the districts. The line is dynamically color-coded: **Green** for safe routes, **Gold** if warnings exist, and **Red** if landslide/heavy rain warnings are currently active in those districts.
- **Safety Directions Panel**: Populates a floating overlay containing the exact path turns and safety bulletin logs for the user's travel route.

### 7. SFT/DPO Side-by-Side Model Arena & RLHF Feedback Tab
- **Arena Layout & Dual Settings**: Added a new **Model Arena** navigation tab with a side-by-side comparison screen. Users can configure independent settings (AI Mode, system prompts, temperature sliders) for Model A (Left) and Model B (Right).
- **Concurrent Inference**: Submits prompts to both models concurrently, tracking rendering latency and outputs in split columns.
- **Direct DPO Feedback Logger**: Clickable preference voting buttons ("Choose Model A" / "Choose Model B") under the responses write the preferred/non-preferred generation pairs directly to `dpo_data.jsonl` via `POST /api/dpo/add`.

### 8. RAG-Based Voice Tour Guide Simulator
- **Leaflet Popup Voice Trigger**: Integrated a `🔊 Play Voice Tour` button directly inside the Leaflet map spot pins.
- **SpeechSynthesis Integration**: Uses Web Speech API TTS to read aloud complete RAG database details (safety, popularity, signal strength, best time to visit, and description).
- **Voice Overlay Controller**: Features a floating bottom-right audio dashboard to play, pause, stop, adjust speed rate (`0.5x` to `2.0x`), and visualize the voice guide using a CSS-animated bouncing sound wave block.

### 9. RAG-Based Synthetic Training Data Generator
- **Auto-Synthesizer**: Added a panel in the Dataset Compiler tab that auto-generates conversational datasets from the travel database using the active Mistral model.
- **Robust Template Fallback**: If GPU limits are hit or the model fails, the system automatically falls back to populating rich Sinhala/English template structures with actual spot attributes to guarantee 100% database compilations.
- **Auto-Appending SFT/DPO**: Generated synthetic SFT data is formatted into ChatML and written to `sft.jsonl`, while DPO data is written to `dpo_data.jsonl`.

### 10. Speculative Decoding Latency Profiler & Acceptance Rate Histogram
- **Acceptance Rate Histogram**: Visualizes token verification distribution (the count of accepted tokens per draft verify step from 0 to 6 tokens) in System Health.
- **Latency Line Chart**: Compares standard target model generation times vs speculative draft decoding times for recent runs.
- **Historical Telemetries**: Tracks cumulative statistics (average speedups and verify acceptance percentages) across server runs.

---

## Verification and Running Instructions

> [!IMPORTANT]
> Since the agent is run under **User-Driven Command System rules**, you must start the dashboard app yourself from your terminal.

Please run the following commands in your workspace:

1. **Activate Python Virtual Environment**:
   ```powershell
   .\.venv\Scripts\activate
   ```
   *(or `.\.venv\Scripts\Activate.ps1` in PowerShell)*

2. **Run the Dashboard server**:
   ```powershell
   uvicorn dashboard.app:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Open browser**:
   Navigate to `http://127.0.0.1:8000` to interact with the updated dashboard!
   - Test **Model Arena**: Go to the **Model Arena** tab. Write a prompt and click **Send to Arena**. Rate latencies and click preferred choices to save to the DPO compiler list.
   - Test **Voice Tour Guide**: Click a tourist pin on the map, click **Play Voice Tour**, and check if the audio synthesis starts and the wave visualizer animates.
   - Test **Synthetic Generator**: Go to the **Dataset Compiler** tab, configure the **Synthetic Training Data Generator** card, and click **Synthesize Dataset Blocks** to auto-compile training sets.
   - Test **Speculative Profiler**: Execute a speculative run, click the **System Health** tab, and view the diagnostics latency lines and acceptance histograms.
