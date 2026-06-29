// Main JS logic for Lumen-1 Training Dashboard
document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let activeTab = 'overview';
    
    // Inject custom CSS styling block for voice tour visualizer waves
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes wave-bounce {
            0%, 100% { height: 5px; }
            50% { height: 35px; }
        }
        .audio-wave-bars.playing .wave-bar { animation: wave-bounce 0.8s infinite ease-in-out; }
        .audio-wave-bars.playing #wave-bar-1 { animation-delay: 0.1s; }
        .audio-wave-bars.playing #wave-bar-2 { animation-delay: 0.3s; }
        .audio-wave-bars.playing #wave-bar-3 { animation-delay: 0.5s; }
        .audio-wave-bars.playing #wave-bar-4 { animation-delay: 0.2s; }
        .audio-wave-bars.playing #wave-bar-5 { animation-delay: 0.4s; }
    `;
    document.head.appendChild(style);
    let demoWarningShown = false;
    let logHistory = '';
    let lossChart = null;
    let overviewDbCategoryChart = null;
    let autoscroll = true;
    let statusInterval = null;
    let systemInterval = null;
    let trainingRunning = false;
    let attachedImageFile = null;
    let playgroundMap = null;
    let mapMarkers = [];

    // DOM Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const currentTabTitle = document.getElementById('current-tab-title');
    const currentTabSubtitle = document.getElementById('current-tab-subtitle');
    
    // Status Elements
    const globalStatusDot = document.getElementById('global-status-dot');
    const globalStatusText = document.getElementById('global-status-text');
    
    // Action Buttons
    const btnStartSft = document.getElementById('btn-start-sft');
    const btnStartDpo = document.getElementById('btn-start-dpo');
    
    // Stats Elements
    const valLoss = document.getElementById('val-loss');
    const valLr = document.getElementById('val-lr');
    const valEpoch = document.getElementById('val-epoch');
    const valStep = document.getElementById('val-step');
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const chartTypeBadge = document.getElementById('chart-type-badge');

    // System Telemetry Elements
    const sysCpuVal = document.getElementById('sys-cpu-val');
    const sysCpuBar = document.getElementById('sys-cpu-bar');
    const sysRamVal = document.getElementById('sys-ram-val');
    const sysRamBar = document.getElementById('sys-ram-bar');
    const gpuContainer = document.getElementById('gpu-container');

    // Config Elements
    const btnSaveSftConfig = document.getElementById('btn-save-sft-config');
    const btnSaveDpoConfig = document.getElementById('btn-save-dpo-config');
    let fullConfigCache = {};

    // Dataset Elements
    const dataSftCount = document.getElementById('data-sft-count');
    const dataDpoCount = document.getElementById('data-dpo-count');
    const datasetSamplesBody = document.getElementById('dataset-samples-body');

    // Playground Elements
    const playgroundPrompt = document.getElementById('playground-prompt');
    const btnRunInference = document.getElementById('btn-run-inference');
    const playgroundResponse = document.getElementById('playground-response');
    const btnUploadImage = document.getElementById('btn-upload-image');
    const playgroundImageInput = document.getElementById('playground-image-input');
    const playgroundImagePreviewName = document.getElementById('playground-image-preview-name');
    const btnClearImage = document.getElementById('btn-clear-image');

    // AI Modes & Autocomplete Elements
    const currentModeBadge = document.getElementById('current-mode-badge');
    const systemPromptPreview = document.getElementById('playground-system-prompt-preview');
    const quickPromptsContainer = document.getElementById('playground-quick-prompts');
    const modeButtons = document.querySelectorAll('.mode-btn');
    const slashDropdown = document.getElementById('slash-autocomplete-dropdown');

    let activeMode = 'default';

    const BASE_MODE_PROMPTS = {
        default: "You are Lumen-1, an advanced AI travel assistant for Sri Lanka. When answering questions, think deeply and structure your reasoning step-by-step to provide the best, most accurate, and safest output.",
        analyst: "You are the Analyst mode of Lumen-1. Your purpose is to analyze training loss, evaluate data quality, and parse performance metrics. Think step-by-step and provide detailed analytical reports.",
        optimizer: "You are the Optimizer mode of Lumen-1. Your purpose is to recommend hyperparameters like LoRA rank, learning rate, batch size, and optimization techniques. Explain the trade-offs of your recommendations.",
        refactor: "You are the Refactor mode of Lumen-1. Your purpose is to review Python code, identify bugs, and suggest robust refactoring solutions. Provide clean code snippets.",
        database: "You are the Database mode of Lumen-1. Your purpose is to query the TripMe JSON database, search and filter records, and write query scripts. Focus on exact matches and JSON validation.",
        security: "You are the Security mode of Lumen-1. Your purpose is to detect prompt injections, monitor safety violations, and classify safety risks. Provide clear safety status reports.",
        agent: "You are the Agent mode of Lumen-1. Your purpose is to use Wikipedia search, mathematical calculation, and local travel database lookup tools to answer user queries accurately."
    };

    const BASE_MODE_QUICK_PROMPTS = {
        default: [
            { label: "⛺ Camping spots", prompt: "⛺ කඳවුරු බැඳීමට (Camping) ශ්‍රී ලංකාවේ සුදුසුම ස්ථාන මොනවාද?" },
            { label: "🌊 Quiet beaches", prompt: "🌊 ශ්‍රී ලංකාවේ සංචාරකයන් අඩු, නිස්කලංක සහ ආරක්ෂිත වෙරළ තීරයන් මොනවාද?" },
            { label: "🏞️ Safe waterfalls", prompt: "🏞️ නෑමට සහ විනෝද වීමට ආරක්ෂිත, දියකඩවල් ලඟ ඇති දියඇලි මොනවාද?" },
            { label: "⛰️ Hiking routes", prompt: "⛰️ ශ්‍රී ලංකාවේ සුන්දරම දර්ශන නැරඹිය හැකි කඳු තරණය කිරීමේ (Hiking) මාර්ග මොනවාද?" }
        ],
        analyst: [
            { label: "📊 Analyze Loss Curves", prompt: "How should I analyze a training loss curve that oscillates heavily or stays flat?" },
            { label: "📈 SFT Data Quality Check", prompt: "What are the key metrics to evaluate SFT conversational dataset quality?" },
            { label: "📉 Detect Overfitting", prompt: "Explain step-by-step how to identify and prevent overfitting in QLoRA fine-tuning." }
        ],
        optimizer: [
            { label: "⚙️ LoRA Rank/Alpha", prompt: "What are the recommended values for LoRA rank (r) and alpha when fine-tuning a 7B model?" },
            { label: "🚀 LR Recommendations", prompt: "Provide a learning rate and scheduler recommendation for SFT vs DPO training steps." },
            { label: "📦 Batch Size Scaling", prompt: "How does batch size and gradient accumulation steps affect model convergence?" }
        ],
        refactor: [
            { label: "🛠️ Fix Context Loss", prompt: "Review my Python code for context loss and state handling in custom decoder loops." },
            { label: "🧩 Optimize Tokenizer", prompt: "Show me a robust regex-based python script to pad multimodal tag placeholders." },
            { label: "🐞 Debug Multimodal Forward", prompt: "How do I fix a dimension mismatch during video frame encoding projection?" }
        ],
        database: [
            { label: "🔍 Search Nuwara Wewa", prompt: "Write a JSON query or Python script to retrieve 'Nuwara Wewa' details from augmented DB." },
            { label: "📁 Verify Schema", prompt: "Explain the standard JSON fields required for indexing Sri Lankan locations in ChromaDB." },
            { label: "🧮 Count Category Stats", prompt: "Write an API endpoint to count and return location counts grouped by travel category." }
        ],
        security: [
            { label: "🛡️ Prompt Injection", prompt: "Analyze this system prompt structure to prevent system instructions jailbreak attacks." },
            { label: "🚫 Toxicity Filtering", prompt: "What parameters should I set in safety.yaml to block unsafe, toxic travel recommendations?" },
            { label: "🔒 Safety Pipeline", prompt: "Explain how to hook a pre-inference and post-inference validation stack to FastAPI." }
        ],
        agent: [
            { label: "🔍 Search Colombo", prompt: "Colombo Sri Lanka" },
            { label: "🧮 Calculator test", prompt: "calculate 345 * 24" }
        ]
    };

    const BASE_SLASH_COMMANDS = [
        { cmd: "/default", mode: "default", purpose: "Sri Lanka travel assistant", color: "grey" },
        { cmd: "/analyst", mode: "analyst", purpose: "Training loss, data quality analysis", color: "blue" },
        { cmd: "/optimizer", mode: "optimizer", purpose: "LoRA/LR/batch size recommendations", color: "teal" },
        { cmd: "/refactor", mode: "refactor", purpose: "Python code review & bug fixes", color: "gold" },
        { cmd: "/database", mode: "database", purpose: "TripMe JSON database queries", color: "purple" },
        { cmd: "/security", mode: "security", purpose: "Prompt injection & safety detection", color: "red" },
        { cmd: "/agent", mode: "agent", purpose: "Agentic tool usage mode", color: "purple" }
    ];

    let activeModePrompts = { ...BASE_MODE_PROMPTS };
    let activeModeQuickPrompts = { ...BASE_MODE_QUICK_PROMPTS };
    let activeSlashCommands = [ ...BASE_SLASH_COMMANDS ];


    function setPlaygroundMode(mode) {
        activeMode = mode;
        
        const allButtons = document.querySelectorAll('.mode-btn');
        allButtons.forEach(btn => {
            if (btn.getAttribute('data-mode') === mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        if (currentModeBadge) {
            currentModeBadge.innerText = `Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
            currentModeBadge.className = 'badge';
            const badgeColors = {
                default: 'badge-grey',
                analyst: 'badge-blue',
                optimizer: 'badge-teal',
                refactor: 'badge-gold',
                database: 'badge-purple',
                security: 'badge-red',
                green: 'badge-green',
                orange: 'badge-orange',
                pink: 'badge-pink'
            };
            
            let badgeClass = badgeColors[mode];
            if (!badgeClass) {
                const customModes = JSON.parse(localStorage.getItem('lumen_custom_modes') || '[]');
                const found = customModes.find(m => m.mode === mode);
                if (found) badgeClass = `badge-${found.color}`;
            }
            currentModeBadge.classList.add(badgeClass || 'badge-purple');
        }

        if (systemPromptPreview) {
            systemPromptPreview.value = activeModePrompts[mode] || '';
        }

        if (quickPromptsContainer) {
            quickPromptsContainer.innerHTML = '';
            const prompts = activeModeQuickPrompts[mode] || [];
            prompts.forEach(p => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'quick-prompt-btn';
                btn.setAttribute('data-prompt', p.prompt);
                btn.innerText = p.label;
                btn.addEventListener('click', () => {
                    playgroundPrompt.value = p.prompt;
                    playgroundPrompt.focus();
                });
                quickPromptsContainer.appendChild(btn);
            });
        }
    }

    // Console Elements
    const consoleOutput = document.getElementById('console-output');
    const btnClearConsole = document.getElementById('btn-clear-console');
    const btnToggleAutoscroll = document.getElementById('btn-toggle-autoscroll');

    // Initialize Lucide Icons
    lucide.createIcons();

    // 1. Navigation & Tab Switching
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
            
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    function switchTab(tabId) {
        activeTab = tabId;
        tabPanes.forEach(pane => {
            pane.classList.remove('active');
            if (pane.id === `tab-${tabId}`) {
                pane.classList.add('active');
                
                // Auto-select the first sub-tab if present
                const firstSubTabBtn = pane.querySelector('.sub-tab-btn');
                if (firstSubTabBtn) {
                    firstSubTabBtn.click();
                }
            }
        });

        // Set Tab Titles & Subtitles
        switch (tabId) {
            case 'overview':
                currentTabTitle.innerText = "Overview";
                currentTabSubtitle.innerText = "Real-time model training metrics and diagnostics";
                break;
            case 'training':
                currentTabTitle.innerText = "Training Pipeline";
                currentTabSubtitle.innerText = "Fine-tune and evaluate models via SFT, DPO, and hardware accelerators";
                break;
            case 'inference':
                currentTabTitle.innerText = "Inference Arena";
                currentTabSubtitle.innerText = "Interact with model variants, test audio, and compare configurations side-by-side";
                break;
            case 'data':
                currentTabTitle.innerText = "Data Studio";
                currentTabSubtitle.innerText = "Explore trip data, query RAG stores, and compile alignment datasets";
                break;
            case 'security':
                currentTabTitle.innerText = "Security Controls";
                currentTabSubtitle.innerText = "Monitor 12-layer security middleware, configure rules, and verify log chain integrity";
                break;
        }
    }

    // 2. Setup Chart.js
    function initChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        // Neon Purple Gradient
        const purpleGlow = ctx.createLinearGradient(0, 0, 0, 300);
        purpleGlow.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
        purpleGlow.addColorStop(1, 'rgba(139, 92, 246, 0.01)');

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#8b5cf6',
                    borderWidth: 2,
                    backgroundColor: purpleGlow,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 2,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#8b5cf6',
                    pointBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        title: { display: true, text: 'Steps', color: '#8e9bb3' },
                        ticks: { color: '#8e9bb3' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        title: { display: true, text: 'Loss', color: '#8e9bb3' },
                        ticks: { color: '#8e9bb3' }
                    }
                }
            }
        });
    }
    initChart();

    // 3. Status Poll / Log fetch
    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            updateStatusUI(data.status, data.type, data.error, data.demo_mode);
        } catch (e) {
            console.error("Error fetching status:", e);
            globalStatusText.innerText = "Status: Server Offline";
            globalStatusDot.className = "status-dot status-idle";
        }
    }

    function updateStatusUI(status, type, error, demoMode = false) {
        globalStatusDot.className = `status-dot status-${status}`;
        
        const demoModeBadge = document.getElementById('demo-mode-badge');
        if (demoModeBadge) {
            if (demoMode) {
                demoModeBadge.classList.remove('hidden');
                
                // Show glassmorphic warning once on page load
                const overlay = document.getElementById('demo-mode-warning-overlay');
                if (overlay && !demoWarningShown) {
                    overlay.classList.add('show');
                    demoWarningShown = true;
                }
            } else {
                demoModeBadge.classList.add('hidden');
            }
        }
        
        const stopButtons = document.querySelectorAll('.btn-stop-training');
        
        if (status === "running") {
            trainingRunning = true;
            globalStatusText.innerText = `Status: Training (${type.toUpperCase()})`;
            btnStartSft.disabled = true;
            btnStartDpo.disabled = true;
            stopButtons.forEach(btn => btn.disabled = false);
            chartTypeBadge.innerText = `${type.toUpperCase()} RUNNING`;
            chartTypeBadge.className = "badge badge-purple";
        } else {
            trainingRunning = false;
            btnStartSft.disabled = false;
            btnStartDpo.disabled = false;
            stopButtons.forEach(btn => btn.disabled = true);
            
            if (status === "completed") {
                globalStatusText.innerText = "Status: Completed!";
                chartTypeBadge.innerText = "COMPLETED";
                chartTypeBadge.className = "badge";
            } else if (status === "error") {
                globalStatusText.innerText = `Status: Error (${error})`;
                chartTypeBadge.innerText = "ERROR";
                chartTypeBadge.className = "badge";
            } else {
                globalStatusText.innerText = "Status: Idle";
            }
        }
    }

    async function fetchLogs() {
        if (!trainingRunning) return;
        try {
            const res = await fetch('/api/logs');
            const data = await res.json();
            
            if (data.logs && data.logs !== logHistory) {
                logHistory = data.logs;
                renderConsoleLogs(data.logs);
            }
        } catch (e) {
            console.error("Error fetching logs:", e);
        }
    }

    function renderConsoleLogs(logsText) {
        consoleOutput.innerHTML = '';
        const lines = logsText.split('\n');
        
        lines.forEach(line => {
            if (!line.trim()) return;
            
            const div = document.createElement('div');
            div.className = 'log-line';
            
            if (line.includes('ERROR') || line.includes('❌') || line.includes('Fail')) {
                div.classList.add('log-error');
            } else if (line.includes('WARNING') || line.includes('⚠️')) {
                div.classList.add('log-warn');
            } else if (line.includes('Complete') || line.includes('✅') || line.includes('Saving')) {
                div.classList.add('log-success');
            } else if (line.includes('🚀') || line.includes('Starting')) {
                div.classList.add('log-sys');
            } else {
                div.classList.add('log-info');
            }
            
            div.textContent = line;
            consoleOutput.appendChild(div);
        });

        if (autoscroll) {
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
    }

    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            
            if (data.history && data.history.length > 0) {
                const latest = data.history[data.history.length - 1];
                
                // Update stats cards
                valLoss.innerText = latest.loss ? latest.loss.toFixed(4) : (latest.train_loss ? latest.train_loss.toFixed(4) : "N/A");
                valLr.innerText = latest.learning_rate ? latest.learning_rate.toExponential(2) : "N/A";
                valEpoch.innerText = latest.epoch ? latest.epoch.toFixed(2) : "N/A";
                valStep.innerText = `${latest.step} / ${latest.max_steps || 'N/A'}`;

                // Update Progress bar
                if (latest.max_steps) {
                    const percent = Math.min(Math.round((latest.step / latest.max_steps) * 100), 100);
                    progressBar.style.width = `${percent}%`;
                    progressPercentage.innerText = `${percent}%`;
                }

                // Update Chart
                const steps = data.history.map(h => h.step);
                const losses = data.history.map(h => h.loss || h.train_loss || 0);
                
                lossChart.data.labels = steps;
                lossChart.data.datasets[0].data = losses;
                lossChart.update();
            }
        } catch (e) {
            console.error("Error fetching metrics:", e);
        }
    }

    // 4. System Telemetry Poll
    async function fetchSystemStats() {
        try {
            const res = await fetch('/api/system');
            const data = await res.json();

            // CPU
            sysCpuVal.innerText = `${data.cpu}%`;
            sysCpuBar.style.width = `${data.cpu}%`;

            // RAM
            const ramShort = document.getElementById('sys-ram-val-short');
            if (ramShort) ramShort.innerText = `${data.ram.percent}%`;
            sysRamVal.innerText = `${data.ram.used} / ${data.ram.total} GB (${data.ram.percent}%)`;
            sysRamBar.style.width = `${data.ram.percent}%`;

            // GPUs
            if (data.gpus && data.gpus.length > 0) {
                gpuContainer.innerHTML = '';
                data.gpus.forEach(gpu => {
                    const gpuCard = document.createElement('div');
                    gpuCard.className = 'gpu-card';
                    gpuCard.innerHTML = `
                        <div class="gpu-header">
                            <span>${gpu.name}</span>
                            <span class="badge badge-purple">CUDA</span>
                        </div>
                        <div class="gpu-grid">
                            <div class="gpu-stat-item">
                                <span>Utilization</span>
                                <p>${gpu.utilization}%</p>
                            </div>
                            <div class="gpu-stat-item">
                                <span>Temp</span>
                                <p>${gpu.temperature}°C</p>
                            </div>
                            <div class="gpu-stat-item">
                                <span>VRAM (Used/Total)</span>
                                <p>${(gpu.memory_used / 1024).toFixed(2)} / ${(gpu.memory_total / 1024).toFixed(2)} GB</p>
                            </div>
                        </div>
                        <div class="metric-meter-group" style="margin-top: 10px;">
                            <div class="meter-bar-bg">
                                <div class="meter-bar-fill meter-teal" style="width: ${gpu.memory_percent}%;"></div>
                            </div>
                        </div>
                    `;
                    gpuContainer.appendChild(gpuCard);
                });
            } else {
                gpuContainer.innerHTML = `
                    <div class="empty-state">
                        <i data-lucide="cpu"></i>
                        <p>No GPUs identified. Verify CUDA drivers or GPUtil package installation.</p>
                    </div>
                `;
                lucide.createIcons();
            }
        } catch (e) {
            console.error("Error fetching system stats:", e);
        }
    }

    // 5. Configs Layer
    async function loadConfigs() {
        try {
            const res = await fetch('/api/configs');
            const data = await res.json();
            fullConfigCache = data;
            renderConfigFields('sft');
            renderConfigFields('dpo');
        } catch (e) {
            console.error("Error fetching configs:", e);
        }
    }

    function validateConfigSafety(type) {
        const banner = document.getElementById(`${type}-config-validator-banner`);
        const messagesDiv = document.getElementById(`${type}-config-validator-messages`);
        if (!banner || !messagesDiv) return;
        
        const warnings = [];

        const batchSizeEl = document.getElementById(`cfg-${type}-per_device_train_batch_size`);
        const rankEl = document.getElementById(`cfg-${type}-lora_r`);
        const alphaEl = document.getElementById(`cfg-${type}-lora_alpha`);
        const optimEl = document.getElementById(`cfg-${type}-optim`);

        if (batchSizeEl) {
            const bs = parseInt(batchSizeEl.value);
            if (bs >= 8) {
                warnings.push(`⚠️ <strong>VRAM Warning:</strong> Train batch size is set to ${bs}. Batch sizes >= 8 often trigger Out Of Memory (OOM) errors on systems with less than 16GB VRAM. Consider reducing to 2 or 4.`);
            }
        }

        if (rankEl && alphaEl) {
            const r = parseInt(rankEl.value);
            const a = parseInt(alphaEl.value);
            if (a !== 2 * r) {
                warnings.push(`⚠️ <strong>LoRA Alpha Ratio rule:</strong> Recommended LoRA practice is <code>lora_alpha = 2 * lora_r</code> (currently alpha is ${a} and rank is ${r}). This maintains training stability.`);
            }
        }

        if (optimEl) {
            const opt = optimEl.value;
            if (opt && !opt.includes('paged')) {
                warnings.push(`⚠️ <strong>Memory safety warning:</strong> Optimizer is set to <code>${opt}</code>. Non-paged optimizers do not utilize memory paging and are prone to VRAM spikes. Recommended: <code>paged_adamw_32bit</code>.`);
            }
        }

        if (warnings.length > 0) {
            messagesDiv.innerHTML = `<ul style="margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 4px;">` + 
                warnings.map(w => `<li>${w}</li>`).join('') + `</ul>`;
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }
    }

    function renderConfigFields(type) {
        const conf = fullConfigCache[type];
        if (!conf) return;

        const container = document.getElementById(`${type}-config-fields`);
        if (!container) return;
        
        container.innerHTML = '';
        const availableModules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

        if (type === 'sft') {
            const modulesGroup = document.createElement('div');
            modulesGroup.className = 'form-group';
            modulesGroup.style.gridColumn = 'span 2';
            
            const label = document.createElement('label');
            label.innerText = 'PEFT TARGET MODULES';
            modulesGroup.appendChild(label);
            
            const checkboxesDiv = document.createElement('div');
            checkboxesDiv.style.display = 'flex';
            checkboxesDiv.style.flexWrap = 'wrap';
            checkboxesDiv.style.gap = '15px';
            checkboxesDiv.style.marginTop = '8px';
            checkboxesDiv.style.background = 'rgba(0,0,0,0.15)';
            checkboxesDiv.style.padding = '12px';
            checkboxesDiv.style.borderRadius = '8px';
            checkboxesDiv.style.border = '1px solid rgba(255,255,255,0.05)';
            
            const selectedModules = conf.target_modules || ["q_proj", "k_proj", "v_proj", "o_proj"];
            
            availableModules.forEach(mod => {
                const lbl = document.createElement('label');
                lbl.style.display = 'flex';
                lbl.style.alignItems = 'center';
                lbl.style.gap = '6px';
                lbl.style.fontSize = '13px';
                lbl.style.cursor = 'pointer';
                lbl.style.textTransform = 'none';
                
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = `cfg-${type}-target-module-checkbox`;
                cb.value = mod;
                cb.checked = selectedModules.includes(mod);
                cb.style.accentColor = 'var(--purple)';
                cb.addEventListener('change', () => validateConfigSafety(type));
                
                lbl.appendChild(cb);
                lbl.appendChild(document.createTextNode(mod));
                checkboxesDiv.appendChild(lbl);
            });
            
            modulesGroup.appendChild(checkboxesDiv);
            container.appendChild(modulesGroup);
        }

        for (const [key, value] of Object.entries(conf)) {
            if (key === 'target_modules') continue;
            if (typeof value === 'object' && value !== null) continue;

            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';
            
            const label = document.createElement('label');
            label.innerText = key.replace(/_/g, ' ').toUpperCase();
            formGroup.appendChild(label);
            
            let input;
            
            if (key === 'num_train_epochs') {
                const sliderWrapper = document.createElement('div');
                sliderWrapper.style.display = 'flex';
                sliderWrapper.style.alignItems = 'center';
                sliderWrapper.style.gap = '12px';
                
                input = document.createElement('input');
                input.type = 'range';
                input.min = '1';
                input.max = '10';
                input.step = '1';
                input.value = value;
                input.style.flexGrow = '1';
                
                const valSpan = document.createElement('span');
                valSpan.style.color = '#c084fc';
                valSpan.style.fontWeight = '600';
                valSpan.style.minWidth = '20px';
                valSpan.innerText = value;
                
                input.addEventListener('input', () => {
                    valSpan.innerText = input.value;
                    validateConfigSafety(type);
                });
                
                sliderWrapper.appendChild(input);
                sliderWrapper.appendChild(valSpan);
                formGroup.appendChild(sliderWrapper);
            } else if (key === 'lora_r') {
                const select = document.createElement('select');
                [4, 8, 16, 32, 64, 128, 256].forEach(r => {
                    select.innerHTML += `<option value="${r}" ${value == r ? 'selected' : ''}>${r}</option>`;
                });
                select.addEventListener('change', () => validateConfigSafety(type));
                input = select;
                formGroup.appendChild(input);
            } else if (key === 'lora_alpha') {
                const select = document.createElement('select');
                [8, 16, 32, 64, 128, 256, 512].forEach(a => {
                    select.innerHTML += `<option value="${a}" ${value == a ? 'selected' : ''}>${a}</option>`;
                });
                select.addEventListener('change', () => validateConfigSafety(type));
                input = select;
                formGroup.appendChild(input);
            } else if (key === 'per_device_train_batch_size') {
                const select = document.createElement('select');
                [1, 2, 4, 8, 16, 32, 64].forEach(b => {
                    select.innerHTML += `<option value="${b}" ${value == b ? 'selected' : ''}>${b}</option>`;
                });
                select.addEventListener('change', () => validateConfigSafety(type));
                input = select;
                formGroup.appendChild(input);
            } else if (key === 'learning_rate') {
                const sliderWrapper = document.createElement('div');
                sliderWrapper.style.display = 'flex';
                sliderWrapper.style.alignItems = 'center';
                sliderWrapper.style.gap = '12px';
                
                input = document.createElement('input');
                input.type = 'text';
                input.value = value;
                input.style.width = '100px';
                
                const presetSelect = document.createElement('select');
                presetSelect.style.flexGrow = '1';
                const presets = [
                    { name: 'Custom', val: value },
                    { name: '1e-6 (Fine adjustment)', val: '1e-6' },
                    { name: '5e-6 (Standard DPO)', val: '5e-6' },
                    { name: '1e-5 (Moderate Tuning)', val: '1e-5' },
                    { name: '5e-5 (Standard SFT)', val: '5e-5' },
                    { name: '1e-4 (Fast SFT)', val: '1e-4' },
                    { name: '2e-4 (Aggressive)', val: '2e-4' }
                ];
                presets.forEach(p => {
                    presetSelect.innerHTML += `<option value="${p.val}" ${value == p.val ? 'selected' : ''}>${p.name} (${p.val})</option>`;
                });
                
                presetSelect.addEventListener('change', () => {
                    input.value = presetSelect.value;
                    validateConfigSafety(type);
                });
                input.addEventListener('input', () => validateConfigSafety(type));
                
                sliderWrapper.appendChild(input);
                sliderWrapper.appendChild(presetSelect);
                formGroup.appendChild(sliderWrapper);
            } else if (key === 'optim') {
                const select = document.createElement('select');
                const optims = ['paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch', 'adamw_hf'];
                optims.forEach(o => {
                    select.innerHTML += `<option value="${o}" ${value == o ? 'selected' : ''}>${o}</option>`;
                });
                select.addEventListener('change', () => validateConfigSafety(type));
                input = select;
                formGroup.appendChild(input);
            } else if (typeof value === 'boolean') {
                input = document.createElement('select');
                input.innerHTML = `
                    <option value="true" ${value ? 'selected' : ''}>TRUE</option>
                    <option value="false" ${!value ? 'selected' : ''}>FALSE</option>
                `;
                input.addEventListener('change', () => validateConfigSafety(type));
                formGroup.appendChild(input);
            } else {
                input = document.createElement('input');
                input.value = value;
                input.type = typeof value === 'number' ? 'number' : 'text';
                input.addEventListener('input', () => validateConfigSafety(type));
                formGroup.appendChild(input);
            }
            
            input.id = `cfg-${type}-${key}`;
            input.setAttribute('data-key', key);
            container.appendChild(formGroup);
        }
        
        validateConfigSafety(type);
    }

    async function saveConfig(type) {
        const conf = fullConfigCache[type];
        if (!conf) return;

        const updatedConfig = { ...conf };
        const container = document.getElementById(`${type}-config-fields`);
        if (!container) return;
        const inputs = container.querySelectorAll('input, select');
        
        inputs.forEach(el => {
            const key = el.getAttribute('data-key');
            if (!key) return;
            let val = el.value;
            
            if (el.tagName === 'SELECT') {
                if (val === 'true' || val === 'false') {
                    val = val === 'true';
                } else if (!isNaN(val) && val.trim() !== '') {
                    val = Number(val);
                }
            } else if (el.type === 'number' || el.type === 'range') {
                val = Number(val);
            }
            
            updatedConfig[key] = val;
        });

        const checkboxes = container.querySelectorAll(`.cfg-${type}-target-module-checkbox`);
        if (checkboxes.length > 0) {
            const selectedModules = [];
            checkboxes.forEach(cb => {
                if (cb.checked) selectedModules.push(cb.value);
            });
            updatedConfig['target_modules'] = selectedModules;
        }

        try {
            const res = await fetch('/api/configs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: type, config: updatedConfig })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                alert(`${type.toUpperCase()} config saved successfully!`);
                fullConfigCache[type] = updatedConfig;
            } else {
                alert(`Error saving config: ${data.detail}`);
            }
        } catch (e) {
            alert(`Error saving config: ${e.message}`);
        }
    }

    if (btnSaveSftConfig) {
        btnSaveSftConfig.addEventListener('click', () => saveConfig('sft'));
    }
    if (btnSaveDpoConfig) {
        btnSaveDpoConfig.addEventListener('click', () => saveConfig('dpo'));
    }

    // 6. Dataset Telemetry
    async function loadDatasetInfo() {
        try {
            const res = await fetch('/api/dataset');
            const data = await res.json();

            dataSftCount.innerText = data.sft.count;
            dataDpoCount.innerText = data.dpo.count;

            // Render samples
            if (data.sft.samples && data.sft.samples.length > 0) {
                datasetSamplesBody.innerHTML = '';
                data.sft.samples.forEach((sample, i) => {
                    const tr = document.createElement('tr');
                    // Sample format might contain 'text' directly
                    const promptText = sample.text || sample.prompt || JSON.stringify(sample);
                    tr.innerHTML = `
                        <td>${i + 1}</td>
                        <td style="font-family: var(--font-mono); font-size: 13px; max-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${promptText}
                        </td>
                    `;
                    datasetSamplesBody.appendChild(tr);
                });
            } else {
                datasetSamplesBody.innerHTML = `
                    <tr>
                        <td colspan="2" class="text-center">No samples found in data/sft.jsonl</td>
                    </tr>
                `;
            }
        } catch (e) {
            console.error("Error fetching dataset stats:", e);
        }
    }

    // 7. Inference Sandbox Playground & Photo Attachment
    btnUploadImage.addEventListener('click', () => playgroundImageInput.click());

    playgroundImageInput.addEventListener('change', () => {
        if (playgroundImageInput.files.length > 0) {
            attachedImageFile = playgroundImageInput.files[0];
            playgroundImagePreviewName.innerText = attachedImageFile.name;
            btnClearImage.classList.remove('hidden');
            
            // Populate and show the floating preview card
            const filename = attachedImageFile.name;
            const type = attachedImageFile.type || 'unknown type';
            const size = (attachedImageFile.size / (1024 * 1024)).toFixed(2) + ' MB';
            
            document.getElementById('multimodal-preview-filename').innerText = filename;
            document.getElementById('multimodal-preview-type').innerText = `${type} (${size})`;
            
            const thumb = document.getElementById('multimodal-preview-thumbnail');
            if (type.startsWith('image/')) {
                thumb.innerText = '🖼️';
                const reader = new FileReader();
                reader.onload = (e) => {
                    thumb.innerHTML = `<img src="${e.target.result}" style="width:100%; height:100%; object-fit:cover;">`;
                };
                reader.readAsDataURL(attachedImageFile);
            } else if (type.startsWith('video/')) {
                thumb.innerText = '🎬';
            } else if (type.startsWith('audio/')) {
                thumb.innerText = '🎵';
            } else {
                thumb.innerText = '📁';
            }
            
            document.getElementById('multimodal-preview-card').classList.remove('hidden');
        }
    });

    const clearMultimodalBtn = document.getElementById('btn-clear-multimodal');
    if (clearMultimodalBtn) {
        clearMultimodalBtn.addEventListener('click', () => {
            attachedImageFile = null;
            playgroundImageInput.value = '';
            playgroundImagePreviewName.innerText = '';
            btnClearImage.classList.add('hidden');
            document.getElementById('multimodal-preview-card').classList.add('hidden');
        });
    }

    btnClearImage.addEventListener('click', () => {
        attachedImageFile = null;
        playgroundImageInput.value = '';
        playgroundImagePreviewName.innerText = '';
        btnClearImage.classList.add('hidden');
        document.getElementById('multimodal-preview-card').classList.add('hidden');
    });

    function addSpeakButtonToBubble(bubble, textContent) {
        const existingBtn = bubble.querySelector('.btn-speak-reply');
        if (existingBtn) existingBtn.remove();

        const speakBtn = document.createElement('button');
        speakBtn.className = 'btn-icon btn-speak-reply';
        speakBtn.style.alignSelf = 'flex-start';
        speakBtn.style.marginTop = '8px';
        speakBtn.style.width = '24px';
        speakBtn.style.height = '24px';
        speakBtn.style.borderRadius = '50%';
        speakBtn.style.padding = '0';
        speakBtn.style.display = 'inline-flex';
        speakBtn.style.alignItems = 'center';
        speakBtn.style.justifyContent = 'center';
        speakBtn.title = 'Speak Reply';
        speakBtn.innerHTML = `<i data-lucide="volume-2" style="width: 12px; height: 12px;"></i>`;
        
        speakBtn.addEventListener('click', () => {
            if (window.speechSynthesis.speaking) {
                window.speechSynthesis.cancel();
                speakBtn.innerHTML = `<i data-lucide="volume-2" style="width: 12px; height: 12px;"></i>`;
                lucide.createIcons();
            } else {
                const cleanText = textContent.replace(/<[^>]*>/g, '').trim();
                const utterance = new SpeechSynthesisUtterance(cleanText);
                const hasSinhala = /[\u0d80-\u0dff]/.test(cleanText);
                if (hasSinhala) {
                    utterance.lang = 'si-LK';
                } else {
                    utterance.lang = 'en-US';
                }
                speakBtn.innerHTML = `<i data-lucide="volume-x" style="width: 12px; height: 12px; color: var(--danger);"></i>`;
                lucide.createIcons();
                utterance.onend = () => {
                    speakBtn.innerHTML = `<i data-lucide="volume-2" style="width: 12px; height: 12px;"></i>`;
                    lucide.createIcons();
                };
                utterance.onerror = () => {
                    speakBtn.innerHTML = `<i data-lucide="volume-2" style="width: 12px; height: 12px;"></i>`;
                    lucide.createIcons();
                };
                window.speechSynthesis.speak(utterance);
            }
        });
        
        const metaDiv = bubble.querySelector('.chat-bubble-meta');
        if (metaDiv) {
            bubble.insertBefore(speakBtn, metaDiv);
        } else {
            bubble.appendChild(speakBtn);
        }
        lucide.createIcons();
        return speakBtn;
    }

    function appendChatBubble(sender, message, mediaName = null, isVoice = false) {
        const emptyState = document.getElementById('playground-empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble chat-bubble-${sender}`;
        
        if (sender === 'assistant' && message !== '...') {
            const tag = document.createElement('div');
            tag.className = `chat-bubble-mode-tag tag-${activeMode}`;
            tag.innerText = `${activeMode.toUpperCase()} MODE`;
            bubble.appendChild(tag);
        }

        const textNode = document.createElement('div');
        textNode.style.whiteSpace = 'pre-wrap';
        textNode.textContent = message;
        bubble.appendChild(textNode);

        if (sender === 'user') {
            if (mediaName) {
                const mediaDiv = document.createElement('div');
                mediaDiv.className = 'chat-bubble-media';
                mediaDiv.innerHTML = `<i data-lucide="clapperboard" style="width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px;"></i><span>Attached Media: ${mediaName}</span>`;
                bubble.appendChild(mediaDiv);
            } else if (isVoice) {
                const mediaDiv = document.createElement('div');
                mediaDiv.className = 'chat-bubble-media';
                mediaDiv.innerHTML = `<i data-lucide="mic" style="width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px;"></i><span>Voice Recording</span>`;
                bubble.appendChild(mediaDiv);
            }
        }

        const metaDiv = document.createElement('div');
        metaDiv.className = 'chat-bubble-meta';
        const role = sender === 'user' ? 'You' : 'Lumen-1';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        metaDiv.innerText = `${role} • ${time}`;
        bubble.appendChild(metaDiv);

        if (sender === 'assistant' && message !== '...') {
            addSpeakButtonToBubble(bubble, message);
        }

        playgroundResponse.appendChild(bubble);
        playgroundResponse.scrollTop = playgroundResponse.scrollHeight;
        
        lucide.createIcons();
        return bubble;
    }

    btnRunInference.addEventListener('click', async () => {
        const fileToUpload = attachedImageFile;
        const prompt = playgroundPrompt.value.trim();
        if (!prompt && !fileToUpload) return;

        const useRag = document.getElementById('playground-rag-toggle').checked;
        const useSpeculative = document.getElementById('playground-speculative-toggle').checked;
        const attachedName = fileToUpload ? fileToUpload.name : null;

        // 1. Append User Bubble
        appendChatBubble('user', prompt || "Analyzed attached media...", attachedName, false);
        
        // Clear elements
        playgroundPrompt.value = '';
        attachedImageFile = null;
        playgroundImageInput.value = '';
        playgroundImagePreviewName.innerText = '';
        btnClearImage.classList.add('hidden');
        document.getElementById('multimodal-preview-card').classList.add('hidden');

        // 2. Append Assistant Loading Bubble
        const loadingBubble = appendChatBubble('assistant', "...");
        loadingBubble.firstChild.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; font-style: italic; color: var(--text-secondary);">
                <div class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-bottom: 0;"></div>
                <span>Lumen-1 is thinking...</span>
            </div>
        `;

        btnRunInference.disabled = true;

        try {
            let res;
            if (fileToUpload) {
                // Upload file to /api/multimodal/upload first
                const uploadFormData = new FormData();
                uploadFormData.append('file', fileToUpload);
                const uploadRes = await fetch('/api/multimodal/upload', {
                    method: 'POST',
                    body: uploadFormData
                });
                const uploadData = await uploadRes.json();
                
                if (uploadRes.status !== 200 || uploadData.status !== 'success') {
                    throw new Error(uploadData.detail || "Failed to upload file to backend.");
                }

                // Call inference with filepath parameter
                res = await fetch(`/api/test-multimodal-model?prompt=${encodeURIComponent(prompt)}&use_rag=${useRag}&use_speculative=${useSpeculative}&filepath=${encodeURIComponent(uploadData.filepath)}`, {
                    method: 'POST'
                });
            } else {
                // Pure text query with active mode and custom prompt editing support
                const customSystemPrompt = systemPromptPreview ? systemPromptPreview.value.trim() : '';
                res = await fetch('/api/test-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        prompt, 
                        use_rag: useRag, 
                        mode: activeMode,
                        system_prompt: customSystemPrompt 
                    })
                });
            }
            const data = await res.json();
            
            if (res.status === 200 && data.status === 'success') {
                loadingBubble.firstChild.innerHTML = '';
                
                // Show speculative stats if available
                const speculativePanel = document.getElementById('speculative-telemetry-panel');
                if (data.speculative_stats && data.speculative_stats.enabled) {
                    speculativePanel.classList.remove('hidden');
                    document.getElementById('spec-stat-speedup').innerText = `${data.speculative_stats.speedup.toFixed(2)}x`;
                    document.getElementById('spec-stat-acceptance').innerText = `${data.speculative_stats.acceptance_rate.toFixed(1)}%`;
                    document.getElementById('spec-stat-speed').innerText = `${data.speculative_stats.tokens_per_sec.toFixed(1)} tok/s`;
                    document.getElementById('spec-stat-std-speed').innerText = `${data.speculative_stats.std_tokens_per_sec.toFixed(1)} tok/s`;
                } else {
                    speculativePanel.classList.add('hidden');
                }

                // Show agent thoughts if available
                if (data.thoughts && data.thoughts.length > 0) {
                    const thoughtsDiv = document.createElement('div');
                    thoughtsDiv.className = 'agent-thoughts';
                    thoughtsDiv.style.background = 'rgba(255,255,255,0.03)';
                    thoughtsDiv.style.border = '1px solid rgba(255,255,255,0.05)';
                    thoughtsDiv.style.borderRadius = '8px';
                    thoughtsDiv.style.padding = '8px 12px';
                    thoughtsDiv.style.marginBottom = '10px';
                    thoughtsDiv.style.fontSize = '12px';
                    
                    let thoughtsHtml = `<div style="font-weight:700; color:var(--purple); margin-bottom:5px; display:flex; align-items:center; gap:5px;"><i data-lucide="bot" style="width:14px; height:14px;"></i> Agent Thought Process:</div>`;
                    data.thoughts.forEach(thought => {
                        thoughtsHtml += `<div style="margin-left:10px; border-left:2px solid rgba(139,92,246,0.3); padding-left:8px; margin-bottom:4px; color:#a3a6b8;">${thought}</div>`;
                    });
                    thoughtsDiv.innerHTML = thoughtsHtml;
                    loadingBubble.firstChild.appendChild(thoughtsDiv);
                }

                const responseTextNode = document.createElement('div');
                responseTextNode.style.whiteSpace = 'pre-wrap';
                responseTextNode.textContent = data.response;
                loadingBubble.firstChild.appendChild(responseTextNode);

                // Render keyframes grid if available (Multimodal Sandbox vision support)
                if (data.keyframes && data.keyframes.length > 0) {
                    const grid = document.createElement('div');
                    grid.style.display = 'grid';
                    grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(60px, 1fr))';
                    grid.style.gap = '8px';
                    grid.style.marginTop = '12px';
                    grid.style.marginBottom = '8px';
                    grid.style.padding = '8px';
                    grid.style.background = 'rgba(0,0,0,0.2)';
                    grid.style.borderRadius = '8px';
                    grid.style.border = '1px solid rgba(255,255,255,0.05)';
                    
                    data.keyframes.forEach((src, idx) => {
                        const img = document.createElement('img');
                        img.src = src;
                        img.style.width = '100%';
                        img.style.height = '45px';
                        img.style.objectFit = 'cover';
                        img.style.borderRadius = '4px';
                        img.style.border = '1px solid rgba(255,255,255,0.1)';
                        img.title = `Keyframe ${idx + 1}`;
                        grid.appendChild(img);
                    });
                    loadingBubble.firstChild.appendChild(grid);
                }

                // Render animated frequency spectrograph if available (Multimodal Sandbox audio support)
                if (data.audio_amplitudes && data.audio_amplitudes.length > 0) {
                    const waveContainer = document.createElement('div');
                    waveContainer.style.marginTop = '12px';
                    waveContainer.style.marginBottom = '8px';
                    waveContainer.style.padding = '10px 15px';
                    waveContainer.style.background = 'rgba(139, 92, 246, 0.05)';
                    waveContainer.style.borderRadius = '10px';
                    waveContainer.style.border = '1px solid rgba(139, 92, 246, 0.15)';
                    waveContainer.style.display = 'flex';
                    waveContainer.style.flexDirection = 'column';
                    waveContainer.style.gap = '6px';
                    
                    const label = document.createElement('span');
                    label.style.fontSize = '9px';
                    label.style.color = '#a78bfa';
                    label.style.fontWeight = '700';
                    label.style.textTransform = 'uppercase';
                    label.style.letterSpacing = '1px';
                    label.innerText = '🔊 Speculative Audio Spectrogram';
                    waveContainer.appendChild(label);

                    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                    svg.setAttribute('width', '100%');
                    svg.setAttribute('height', '36');
                    svg.style.overflow = 'visible';
                    
                    const barSpacing = 4;
                    const barWidth = 3;
                    
                    data.audio_amplitudes.forEach((amp, i) => {
                        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                        const x = i * (barWidth + barSpacing);
                        rect.setAttribute('x', x.toString());
                        rect.setAttribute('width', barWidth.toString());
                        const h = Math.max(2, amp * 30);
                        const y = 30 - h;
                        rect.setAttribute('y', y.toString());
                        rect.setAttribute('height', h.toString());
                        rect.setAttribute('fill', 'url(#spectrogram-gradient)');
                        rect.style.rx = '1.5px';
                        
                        const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
                        animate.setAttribute('attributeName', 'height');
                        animate.setAttribute('values', `${h};${Math.max(2, h * 0.4)};${h}`);
                        animate.setAttribute('dur', `${0.6 + (i % 5) * 0.15}s`);
                        animate.setAttribute('repeatCount', 'indefinite');
                        rect.appendChild(animate);
                        
                        const animateY = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
                        animateY.setAttribute('attributeName', 'y');
                        animateY.setAttribute('values', `${y};${30 - Math.max(2, h * 0.4)};${y}`);
                        animateY.setAttribute('dur', `${0.6 + (i % 5) * 0.15}s`);
                        animateY.setAttribute('repeatCount', 'indefinite');
                        rect.appendChild(animateY);
                        
                        svg.appendChild(rect);
                    });
                    
                    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                    gradient.setAttribute('id', 'spectrogram-gradient');
                    gradient.setAttribute('x1', '0%');
                    gradient.setAttribute('y1', '100%');
                    gradient.setAttribute('x2', '0%');
                    gradient.setAttribute('y2', '0%');
                    
                    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                    stop1.setAttribute('offset', '0%');
                    stop1.setAttribute('stop-color', '#8b5cf6');
                    
                    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                    stop2.setAttribute('offset', '100%');
                    stop2.setAttribute('stop-color', '#ec4899');
                    
                    gradient.appendChild(stop1);
                    gradient.appendChild(stop2);
                    defs.appendChild(gradient);
                    svg.appendChild(defs);
                    waveContainer.appendChild(svg);
                    
                    loadingBubble.firstChild.appendChild(waveContainer);
                }
                
                parseAndPlotLocations(data.response);
                
                // Add speak button and check Auto-TTS toggle
                const speakBtn = addSpeakButtonToBubble(loadingBubble, data.response);
                const ttsToggle = document.getElementById('playground-tts-toggle');
                if (ttsToggle && ttsToggle.checked) {
                    speakBtn.click();
                }
                
                lucide.createIcons();
            } else {
                loadingBubble.firstChild.innerHTML = '';
                loadingBubble.firstChild.textContent = `Error: ${data.detail || 'Inference call failed'}`;
                loadingBubble.style.borderColor = 'var(--danger)';
            }
        } catch (e) {
            loadingBubble.firstChild.innerHTML = '';
            loadingBubble.firstChild.textContent = `Network Error: ${e.message}`;
            loadingBubble.style.borderColor = 'var(--danger)';
        } finally {
            btnRunInference.disabled = false;
            playgroundResponse.scrollTop = playgroundResponse.scrollHeight;
        }
    });

    // 8. Process Control Handlers (Start/Stop)
    async function runTraining(type) {
        try {
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type })
            });
            const data = await res.json();
            
            if (res.status === 200 && data.status === 'success') {
                trainingRunning = true;
                logHistory = '';
                consoleOutput.innerHTML = '<div class="log-line log-sys">Process initialization started. Launching pipeline...</div>';
                updateStatusUI('running', type, null);
            } else {
                alert(`Error: ${data.detail || 'Could not start process'}`);
            }
        } catch (e) {
            alert(`Error connecting to server: ${e.message}`);
        }
    }

    btnStartSft.addEventListener('click', () => runTraining('sft'));
    btnStartDpo.addEventListener('click', () => runTraining('dpo'));

    document.querySelectorAll('.btn-stop-training').forEach(btn => {
        btn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to stop/kill the active training process? Any unsaved checkpoint step will be lost.")) return;
            try {
                const res = await fetch('/api/stop', { method: 'POST' });
                const data = await res.json();
                if (res.status === 200) {
                    trainingRunning = false;
                    updateStatusUI('idle', null, null);
                    const div = document.createElement('div');
                    div.className = 'log-line log-error';
                    div.textContent = '⚠️ Training process manually terminated by User.';
                    consoleOutput.appendChild(div);
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                }
            } catch (e) {
                alert(`Error stopping process: ${e.message}`);
            }
        });
    });

    // 9. Console actions
    btnClearConsole.addEventListener('click', () => {
        consoleOutput.innerHTML = '';
        logHistory = '';
    });

    btnToggleAutoscroll.addEventListener('click', () => {
        autoscroll = !autoscroll;
        btnToggleAutoscroll.classList.toggle('active', autoscroll);
    });

    // 10. Database Editor JS Integration
    const dbSearch = document.getElementById('db-search');
    const dbCategoryFilter = document.getElementById('db-category-filter');
    const dbBtnAdd = document.getElementById('db-btn-add');
    const dbFileBadge = document.getElementById('db-file-badge');
    const dbRecordsBody = document.getElementById('db-records-body');
    
    const dbModal = document.getElementById('db-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalClose = document.getElementById('modal-close');
    const dbForm = document.getElementById('db-form');
    const btnCancelPlace = document.getElementById('btn-cancel-place');
    
    const formId = document.getElementById('form-id');
    const formName = document.getElementById('form-name');
    const formCategory = document.getElementById('form-category');
    const formDistrict = document.getElementById('form-district');
    const formProvince = document.getElementById('form-province');
    const formLat = document.getElementById('form-lat');
    const formLng = document.getElementById('form-lng');
    const formSafety = document.getElementById('form-safety');
    const formWildlife = document.getElementById('form-wildlife');
    const formPopularity = document.getElementById('form-popularity');
    const formBudget = document.getElementById('form-budget');
    const formOpening = document.getElementById('form-opening');
    const formMobile = document.getElementById('form-mobile');
    const formRoad = document.getElementById('form-road');
    const formActivities = document.getElementById('form-activities');
    const formTicket = document.getElementById('form-ticket');
    const formBestTime = document.getElementById('form-best-time');
    const formHeight = document.getElementById('form-height');
    const formLength = document.getElementById('form-length');
    const formSurfing = document.getElementById('form-surfing');
    const formFamily = document.getElementById('form-family');
    const formParking = document.getElementById('form-parking');
    const formToilets = document.getElementById('form-toilets');
    const formFood = document.getElementById('form-food');
    const formWheelchair = document.getElementById('form-wheelchair');
    const formCamping = document.getElementById('form-camping');
    const formGuide = document.getElementById('form-guide');
    const formRain = document.getElementById('form-rain');
    const formMonsoon = document.getElementById('form-monsoon');
    const formDescription = document.getElementById('form-description');

    let cachedPlaces = [];

    async function loadDatabaseRecords() {
        try {
            const res = await fetch('/api/database');
            const data = await res.json();
            if (data.status === 'success') {
                cachedPlaces = data.data;
                dbFileBadge.innerText = data.file;
                renderDatabaseTable();
                loadDatabaseStats();
            } else {
                dbRecordsBody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Error: ${data.detail || 'Could not load records'}</td></tr>`;
            }
        } catch (e) {
            dbRecordsBody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Network Error: ${e.message}</td></tr>`;
        }
    }

    function renderDatabaseTable() {
        const query = dbSearch.value.toLowerCase().trim();
        const category = dbCategoryFilter.value;

        const filtered = cachedPlaces.filter(place => {
            const matchesQuery = place.name.toLowerCase().includes(query) || place.district_id.toLowerCase().includes(query);
            const matchesCategory = category === 'All' || place.category_id.toLowerCase() === category.toLowerCase();
            return matchesQuery && matchesCategory;
        });

        dbRecordsBody.innerHTML = '';
        if (filtered.length === 0) {
            dbRecordsBody.innerHTML = `<tr><td colspan="6" class="text-center">No matching records found.</td></tr>`;
            return;
        }

        filtered.forEach(place => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary);">${place.id}</td>
                <td><strong>${place.name}</strong></td>
                <td><span class="badge badge-purple">${place.category_id}</span></td>
                <td>${place.district_id}</td>
                <td>
                    <span class="badge ${place.safety_level === 'Safe' ? 'badge-purple' : (place.safety_level === 'Moderate' ? 'badge-gold' : 'btn-icon-danger')}">
                        ${place.safety_level}
                    </span>
                </td>
                <td style="text-align: center; display: flex; gap: 8px; justify-content: center;">
                    <button class="btn-icon btn-edit" data-id="${place.id}" title="Edit location details">
                        <i data-lucide="edit-2" style="width: 14px; height: 14px;"></i>
                    </button>
                    <button class="btn-icon btn-icon-danger btn-delete" data-id="${place.id}" title="Delete record">
                        <i data-lucide="trash" style="width: 14px; height: 14px; color: var(--danger);"></i>
                    </button>
                </td>
            `;
            dbRecordsBody.appendChild(tr);
        });

        lucide.createIcons();

        // Bind Edit buttons
        dbRecordsBody.querySelectorAll('.btn-edit').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.getAttribute('data-id');
                const place = cachedPlaces.find(p => p.id === id);
                if (place) openModal(place);
            });
        });

        // Bind Delete buttons
        dbRecordsBody.querySelectorAll('.btn-delete').forEach(btn => {
            btn.addEventListener('click', async () => {
                const id = btn.getAttribute('data-id');
                const place = cachedPlaces.find(p => p.id === id);
                if (!place) return;
                if (confirm(`Are you sure you want to permanently delete "${place.name}" from the database?`)) {
                    try {
                        const res = await fetch(`/api/database/delete/${id}`, { method: 'DELETE' });
                        const result = await res.json();
                        if (res.status === 200) {
                            alert(result.message);
                            loadDatabaseRecords();
                        } else {
                            alert(`Error deleting record: ${result.detail}`);
                        }
                    } catch (e) {
                        alert(`Network Error: ${e.message}`);
                    }
                }
            });
        });
    }

    function openModal(place = null) {
        dbForm.reset();
        if (place) {
            modalTitle.innerText = "Edit Place Details";
            formId.value = place.id;
            formName.value = place.name;
            formCategory.value = place.category_id;
            formDistrict.value = place.district_id;
            formProvince.value = place.province_id;
            formLat.value = place.lat;
            formLng.value = place.lng;
            formSafety.value = place.safety_level || 'Safe';
            formWildlife.value = place.wildlife_hazard || '';
            formPopularity.value = place.tourist_popularity || 'High';
            formBudget.value = place.budget_category || 'Free';
            formOpening.value = place.opening_hours || '';
            formMobile.value = place.mobile_signal || '';
            formRoad.value = place.road_condition || '';
            formActivities.value = place.activities || '';
            formTicket.value = place.ticket_price || '';
            formBestTime.value = place.best_time_to_visit || '';
            formHeight.value = place.Height_m || '0';
            formLength.value = place.Length_km || '0';
            formSurfing.value = place.Surfing || 'no';
            formFamily.value = place.family_friendly || 'yes';
            formParking.value = place.parking_avail || 'yes';
            formToilets.value = place.toilets || 'yes';
            formFood.value = place.food_nearby || 'yes';
            formWheelchair.value = place.wheelchair_access || 'no';
            formCamping.value = place.camping_allowed || 'no';
            formGuide.value = place.guide_required || 'no';
            formRain.value = place.rain_sensitivity || 'Safe';
            formMonsoon.value = place.monsoon_note || 'None';
            formDescription.value = place.description || '';
        } else {
            modalTitle.innerText = "Add New Travel Destination";
            formId.value = '';
            formSafety.value = 'Safe';
            formPopularity.value = 'High';
            formBudget.value = 'Free';
            formHeight.value = '0';
            formLength.value = '0';
            formSurfing.value = 'no';
            formFamily.value = 'yes';
            formParking.value = 'yes';
            formToilets.value = 'yes';
            formFood.value = 'yes';
            formWheelchair.value = 'no';
            formCamping.value = 'no';
            formGuide.value = 'no';
            formRain.value = 'Safe';
            formMonsoon.value = 'None';
        }
        dbModal.classList.remove('hidden');
    }

    function closeModal() {
        dbModal.classList.add('hidden');
    }

    dbSearch.addEventListener('input', renderDatabaseTable);
    dbCategoryFilter.addEventListener('change', renderDatabaseTable);
    dbBtnAdd.addEventListener('click', () => openModal(null));
    modalClose.addEventListener('click', closeModal);
    btnCancelPlace.addEventListener('click', closeModal);

    dbForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const id = formId.value;
        const placeData = {
            name: formName.value,
            description: formDescription.value,
            district_id: formDistrict.value,
            province_id: formProvince.value,
            category_id: formCategory.value,
            lat: parseFloat(formLat.value),
            lng: parseFloat(formLng.value),
            opening_hours: formOpening.value,
            mobile_signal: formMobile.value,
            road_condition: formRoad.value,
            activities: formActivities.value,
            tourist_popularity: formPopularity.value,
            family_friendly: formFamily.value,
            budget_category: formBudget.value,
            ticket_price: formTicket.value,
            parking_avail: formParking.value,
            toilets: formToilets.value,
            food_nearby: formFood.value,
            wheelchair_access: formWheelchair.value,
            camping_allowed: formCamping.value,
            safety_level: formSafety.value,
            wildlife_hazard: formWildlife.value,
            guide_required: formGuide.value,
            rain_sensitivity: formRain.value,
            monsoon_note: formMonsoon.value,
            best_time_to_visit: formBestTime.value,
            Height_m: formHeight.value,
            Length_km: formLength.value,
            Surfing: formSurfing.value
        };

        const isEdit = id !== '';
        const url = isEdit ? '/api/database/update' : '/api/database/add';
        
        if (isEdit) {
            placeData.id = id;
        }

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(placeData)
            });
            const result = await res.json();
            if (res.status === 200 && result.status === 'success') {
                alert(result.message);
                closeModal();
                loadDatabaseRecords();
            } else {
                alert(`Error: ${result.detail || 'Save failed'}`);
            }
        } catch (e) {
            alert(`Network Error: ${e.message}`);
        }
    });

    // 12. RAG Document Uploader
    const ragDropZone = document.getElementById('rag-drop-zone');
    const ragFileInput = document.getElementById('rag-file-input');
    const ragUploadProgressWrapper = document.getElementById('rag-upload-progress-wrapper');
    const ragFileName = document.getElementById('rag-file-name');
    const ragProgressText = document.getElementById('rag-progress-text');
    const ragProgressBar = document.getElementById('rag-progress-bar');
    const ragDocsBadge = document.getElementById('rag-docs-badge');
    const ragDocumentsBody = document.getElementById('rag-documents-body');

    ragDropZone.addEventListener('click', () => ragFileInput.click());

    ragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        ragDropZone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
        ragDropZone.addEventListener(type, () => {
            ragDropZone.classList.remove('dragover');
        });
    });

    ragDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        ragDropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleRagUpload(e.dataTransfer.files[0]);
        }
    });

    ragFileInput.addEventListener('change', () => {
        if (ragFileInput.files.length > 0) {
            handleRagUpload(ragFileInput.files[0]);
        }
    });

    function handleRagUpload(file) {
        ragFileName.innerText = file.name;
        ragProgressText.innerText = '0%';
        ragProgressBar.style.width = '0%';
        ragUploadProgressWrapper.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/rag/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                ragProgressText.innerText = `${percent}%`;
                ragProgressBar.style.width = `${percent}%`;
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const res = JSON.parse(xhr.responseText);
                alert(res.message);
                loadRagDocuments();
            } else {
                alert('Upload failed. Ensure PDF/TXT parsing dependencies are installed.');
            }
            setTimeout(() => {
                ragUploadProgressWrapper.classList.add('hidden');
            }, 3000);
        };

        xhr.onerror = () => {
            alert('Network error occurred during document upload.');
            ragUploadProgressWrapper.classList.add('hidden');
        };

        xhr.send(formData);
    }

    // 13. Database stats charts integration
    let dbCategoryChart = null;
    let dbSafetyChart = null;

    async function loadDatabaseStats() {
        try {
            const res = await fetch('/api/database/stats');
            const data = await res.json();
            if (data.status === 'success') {
                renderStatsCharts(data.categories, data.safety_levels);
            }
        } catch (e) {
            console.error("Error loading database stats:", e);
        }
    }

    function renderStatsCharts(categories, safety) {
        if (dbCategoryChart) dbCategoryChart.destroy();
        if (dbSafetyChart) dbSafetyChart.destroy();

        const catCtx = document.getElementById('chart-db-categories').getContext('2d');
        const catLabels = Object.keys(categories);
        const catValues = Object.values(categories);

        dbCategoryChart = new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels: catLabels,
                datasets: [{
                    data: catValues,
                    backgroundColor: [
                        'rgba(139, 92, 246, 0.6)', 
                        'rgba(59, 130, 246, 0.6)',  
                        'rgba(20, 184, 166, 0.6)',  
                        'rgba(245, 158, 11, 0.6)',  
                        'rgba(239, 68, 68, 0.6)'    
                    ],
                    borderColor: 'rgba(255, 255, 255, 0.05)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#8e9bb3', font: { size: 10 } }
                    }
                }
            }
        });

        const safeCtx = document.getElementById('chart-db-safety').getContext('2d');
        const safeLabels = Object.keys(safety);
        const safeValues = Object.values(safety);

        dbSafetyChart = new Chart(safeCtx, {
            type: 'bar',
            data: {
                labels: safeLabels,
                datasets: [{
                    label: 'Locations',
                    data: safeValues,
                    backgroundColor: [
                        'rgba(20, 184, 166, 0.6)',  
                        'rgba(245, 158, 11, 0.6)',  
                        'rgba(239, 68, 68, 0.6)'    
                    ],
                    borderColor: 'rgba(255, 255, 255, 0.05)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#8e9bb3' }, grid: { display: false } },
                    y: { ticks: { color: '#8e9bb3' }, grid: { color: 'rgba(255,255,255,0.03)' } }
                }
            }
        });

        // Overview tab database category distribution chart
        const overviewCatCtx = document.getElementById('dbCategoryChart');
        if (overviewCatCtx) {
            if (overviewDbCategoryChart) overviewDbCategoryChart.destroy();
            
            const totalSpots = Object.values(categories).reduce((a, b) => a + b, 0);
            const badgeEl = document.getElementById('overview-db-total-badge');
            if (badgeEl) badgeEl.innerText = `${totalSpots} Spots`;
            
            overviewDbCategoryChart = new Chart(overviewCatCtx.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: catLabels,
                    datasets: [{
                        data: catValues,
                        backgroundColor: [
                            'rgba(139, 92, 246, 0.6)', 
                            'rgba(59, 130, 246, 0.6)',  
                            'rgba(20, 184, 166, 0.6)',  
                            'rgba(245, 158, 11, 0.6)',  
                            'rgba(239, 68, 68, 0.6)'    
                         ],
                         borderColor: 'rgba(255, 255, 255, 0.05)',
                         borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: { color: '#8e9bb3', font: { size: 10 } }
                        }
                    }
                }
            });
        }
    }

    // 14. Playground Microphone Audio Recording
    const btnRecordAudio = document.getElementById('btn-record-audio');
    const micIcon = document.getElementById('mic-icon');
    const recordText = document.getElementById('record-text');
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    if (btnRecordAudio) {
        btnRecordAudio.addEventListener('click', async () => {
            if (!isRecording) {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert("Recording is not supported in this browser or over insecure connections.");
                    return;
                }
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        sendAudioForInference(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    btnRecordAudio.classList.add('recording-active');
                    if (recordText) recordText.innerText = 'Recording...';
                } catch (err) {
                    console.error("Failed to access microphone:", err);
                    alert("Microphone access denied or failed.");
                }
            } else {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                isRecording = false;
                btnRecordAudio.classList.remove('recording-active');
                if (recordText) recordText.innerText = 'Voice Input';
            }
        });
    }

    async function sendAudioForInference(audioBlob) {
        // 1. Append User Chat Bubble with Voice label
        appendChatBubble('user', "Voice Query", null, true);

        // 2. Append Assistant Loading Bubble
        const loadingBubble = appendChatBubble('assistant', "...");
        loadingBubble.firstChild.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; font-style: italic; color: var(--text-secondary);">
                <div class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-bottom: 0;"></div>
                <span>Processing voice audio & transcribing...</span>
            </div>
        `;
        
        if (btnRecordAudio) btnRecordAudio.disabled = true;

        const useRag = document.getElementById('playground-rag-toggle').checked;
        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.wav');

        try {
            const res = await fetch(`/api/test-audio-model?use_rag=${useRag}&mode=${activeMode}`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.status === 200 && data.status === 'success') {
                loadingBubble.firstChild.innerHTML = '';
                
                let htmlContent = `<div><strong>Transcription:</strong> "${data.transcription}"</div>`;
                
                // Show thoughts if available
                if (data.thoughts && data.thoughts.length > 0) {
                    htmlContent += `
                        <div class="agent-thoughts" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 8px 12px; margin-top: 10px; margin-bottom: 10px; font-size: 12px;">
                            <div style="font-weight:700; color:var(--purple); margin-bottom:5px; display:flex; align-items:center; gap:5px;"><i data-lucide="bot" style="width:14px; height:14px;"></i> Agent Thought Process:</div>
                    `;
                    data.thoughts.forEach(thought => {
                        htmlContent += `<div style="margin-left:10px; border-left:2px solid rgba(139,92,246,0.3); padding-left:8px; margin-bottom:4px; color:#a3a6b8;">${thought}</div>`;
                    });
                    htmlContent += `</div>`;
                }
                
                htmlContent += `<div style="margin-top: 8px; white-space: pre-wrap;">${data.response}</div>`;
                loadingBubble.firstChild.innerHTML = htmlContent;
                parseAndPlotLocations(data.response);
                
                // Add speak button and check Auto-TTS toggle
                const speakBtn = addSpeakButtonToBubble(loadingBubble, data.response);
                const ttsToggle = document.getElementById('playground-tts-toggle');
                if (ttsToggle && ttsToggle.checked) {
                    speakBtn.click();
                }
                
                lucide.createIcons();
            } else {
                loadingBubble.firstChild.innerHTML = '';
                loadingBubble.firstChild.textContent = `Error: ${data.detail || 'Voice inference failed'}`;
                loadingBubble.style.borderColor = 'var(--danger)';
            }
        } catch (e) {
            loadingBubble.firstChild.innerHTML = '';
            loadingBubble.firstChild.textContent = `Network Error: ${e.message}`;
            loadingBubble.style.borderColor = 'var(--danger)';
        } finally {
            if (btnRecordAudio) btnRecordAudio.disabled = false;
            playgroundResponse.scrollTop = playgroundResponse.scrollHeight;
        }
    }

    // 14B. RAG vector documents list & delete logic
    async function loadRagDocuments() {
        try {
            const res = await fetch('/api/rag/documents');
            const data = await res.json();
            if (data.status === 'success') {
                if (ragDocsBadge) ragDocsBadge.innerText = `${data.data.length} Files`;
                renderRagDocumentsTable(data.data);
            } else {
                if (ragDocumentsBody) ragDocumentsBody.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Error: ${data.detail || 'Could not load inventory'}</td></tr>`;
            }
        } catch (e) {
            if (ragDocumentsBody) ragDocumentsBody.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Network Error: ${e.message}</td></tr>`;
        }
    }

    function renderRagDocumentsTable(docs) {
        if (!ragDocumentsBody) return;
        ragDocumentsBody.innerHTML = '';
        if (docs.length === 0) {
            ragDocumentsBody.innerHTML = `<tr><td colspan="4" class="text-center">No indexed guides found in RAG vector database.</td></tr>`;
            return;
        }

        docs.forEach(doc => {
            const tr = document.createElement('tr');
            const sizeKB = (doc.size_bytes / 1024).toFixed(1);
            tr.innerHTML = `
                <td><strong>${doc.filename}</strong></td>
                <td>${sizeKB} KB</td>
                <td><span class="badge badge-purple">${doc.chunks} chunks</span></td>
                <td style="text-align: center;">
                    <button class="btn-icon btn-icon-danger btn-delete-rag" data-name="${doc.filename}" title="Remove index from collection">
                        <i data-lucide="trash" style="width: 14px; height: 14px; color: var(--danger);"></i>
                    </button>
                </td>
            `;
            ragDocumentsBody.appendChild(tr);
        });

        lucide.createIcons();

        // Bind delete
        ragDocumentsBody.querySelectorAll('.btn-delete-rag').forEach(btn => {
            btn.addEventListener('click', async () => {
                const filename = btn.getAttribute('data-name');
                if (confirm(`Are you sure you want to permanently delete "${filename}" and its vector indices from the RAG database?`)) {
                    try {
                        const res = await fetch(`/api/rag/documents/${filename}`, { method: 'DELETE' });
                        const result = await res.json();
                        if (res.status === 200) {
                            alert(result.message);
                            loadRagDocuments();
                        } else {
                            alert(`Error: ${result.detail || 'Deletion failed'}`);
                        }
                    } catch (e) {
                        alert(`Network Error: ${e.message}`);
                    }
                }
            });
        });
    }

    // 14C. SFT to DPO Preference Dataset Compiler Logic
    const compilerSftSelect = document.getElementById('compiler-sft-select');
    const btnCompilerLoadRandom = document.getElementById('btn-compiler-load-random');
    const compilerPrompt = document.getElementById('compiler-prompt');
    const btnCompilerGenerate = document.getElementById('btn-compiler-generate');
    const compilerCandidatesWrapper = document.getElementById('compiler-candidates-wrapper');
    const compilerCandidateA = document.getElementById('compiler-candidate-a');
    const compilerCandidateB = document.getElementById('compiler-candidate-b');
    const prefSelectA = document.getElementById('pref-select-a');
    const prefSelectB = document.getElementById('pref-select-b');
    const compilerActionsBox = document.getElementById('compiler-actions-box');
    const btnCompilerSubmit = document.getElementById('btn-compiler-submit');
    const btnCompilerReset = document.getElementById('btn-compiler-reset');

    let sftPromptsCache = [];

    async function loadCompilerSFTList() {
        if (sftPromptsCache.length > 0) return;
        try {
            const res = await fetch('/api/dataset');
            const data = await res.json();
            if (data.sft && data.sft.samples) {
                sftPromptsCache = data.sft.samples.map(sample => {
                    return sample.text || sample.prompt || JSON.stringify(sample);
                });
                
                if (compilerSftSelect) {
                    compilerSftSelect.innerHTML = '<option value="">-- Load a prompt from training data or type custom below --</option>';
                    sftPromptsCache.forEach((promptText, i) => {
                        const opt = document.createElement('option');
                        opt.value = promptText;
                        opt.textContent = promptText.length > 80 ? `${promptText.substring(0, 80)}...` : promptText;
                        compilerSftSelect.appendChild(opt);
                    });
                }
            }
        } catch (e) {
            console.error("Error loading compiler prompt samples:", e);
        }
    }

    if (compilerSftSelect) {
        compilerSftSelect.addEventListener('change', () => {
            if (compilerSftSelect.value) {
                compilerPrompt.value = compilerSftSelect.value;
            }
        });
    }

    if (btnCompilerLoadRandom) {
        btnCompilerLoadRandom.addEventListener('click', () => {
            if (sftPromptsCache.length > 0) {
                const rand = sftPromptsCache[Math.floor(Math.random() * sftPromptsCache.length)];
                compilerPrompt.value = rand;
                compilerSftSelect.value = rand;
            } else {
                alert("No SFT prompts loaded. Try typing in a custom prompt.");
            }
        });
    }

    if (btnCompilerGenerate) {
        btnCompilerGenerate.addEventListener('click', async () => {
            const prompt = compilerPrompt.value.trim();
            if (!prompt) {
                alert("Please enter or select a prompt context first.");
                return;
            }

            btnCompilerGenerate.disabled = true;
            btnCompilerGenerate.innerHTML = `<div class="spinner" style="width: 14px; height: 14px; border-width: 2px; display: inline-block; vertical-align: middle; margin-right: 6px; margin-bottom: 0;"></div><span>Generating Candidates...</span>`;

            try {
                const res = await fetch('/api/compiler/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await res.json();
                
                if (res.status === 200 && data.status === 'success') {
                    compilerCandidateA.value = data.response_a;
                    compilerCandidateB.value = data.response_b;
                    compilerCandidatesWrapper.style.display = 'grid';
                    compilerActionsBox.style.display = 'flex';
                } else {
                    alert(`Error generating candidates: ${data.detail || 'Inference call failed'}`);
                }
            } catch (e) {
                alert(`Error: ${e.message}`);
            } finally {
                btnCompilerGenerate.disabled = false;
                btnCompilerGenerate.innerHTML = `<i data-lucide="sparkles"></i><span>Generate A/B Candidates</span>`;
                lucide.createIcons();
            }
        });
    }

    function resetCompiler() {
        if (compilerCandidatesWrapper) compilerCandidatesWrapper.style.display = 'none';
        if (compilerActionsBox) compilerActionsBox.style.display = 'none';
        if (compilerCandidateA) compilerCandidateA.value = '';
        if (compilerCandidateB) compilerCandidateB.value = '';
        if (compilerPrompt) compilerPrompt.value = '';
        if (compilerSftSelect) compilerSftSelect.value = '';
    }

    if (btnCompilerReset) {
        btnCompilerReset.addEventListener('click', resetCompiler);
    }

    if (btnCompilerSubmit) {
        btnCompilerSubmit.addEventListener('click', async () => {
            const prompt = compilerPrompt.value.trim();
            const responseA = compilerCandidateA.value.trim();
            const responseB = compilerCandidateB.value.trim();
            
            if (!prompt || !responseA || !responseB) {
                alert("Prompt and both candidate responses are required.");
                return;
            }

            const isAChozen = prefSelectA.checked;
            const chosen = isAChozen ? responseA : responseB;
            const rejected = isAChozen ? responseB : responseA;

            btnCompilerSubmit.disabled = true;
            
            try {
                const res = await fetch('/api/dpo/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, chosen, rejected })
                });
                const result = await res.json();
                
                if (res.status === 200 && result.status === 'success') {
                    alert(result.message);
                    resetCompiler();
                    loadDatasetInfo();
                    loadCompilerDpoList();
                    loadDpoDiagnostics();
                } else {
                    alert(`Error: ${result.detail || 'Save failed'}`);
                }
            } catch (e) {
                alert(`Network Error: ${e.message}`);
            } finally {
                btnCompilerSubmit.disabled = false;
            }
        });
    }

    async function loadCompilerDpoList() {
        const body = document.getElementById('compiler-dpo-body');
        const badge = document.getElementById('compiler-dpo-badge');
        if (!body) return;
        
        try {
            const res = await fetch('/api/dpo');
            const data = await res.json();
            
            if (data.status === 'success') {
                const pairs = data.data;
                if (badge) badge.innerText = `${pairs.length} Pairs`;
                
                body.innerHTML = '';
                if (pairs.length === 0) {
                    body.innerHTML = '<tr><td colspan="4" class="text-center">No compiled preference pairs found. Use the compiler above to create some!</td></tr>';
                    return;
                }
                
                pairs.forEach(pair => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-family: var(--font-mono); font-size: 12px; white-space: pre-wrap; max-width: 200px; word-break: break-all;">${pair.prompt}</td>
                        <td style="white-space: pre-wrap; max-width: 250px; word-break: break-all; color: #34d399;">${pair.chosen}</td>
                        <td style="white-space: pre-wrap; max-width: 250px; word-break: break-all; color: #f87171;">${pair.rejected}</td>
                        <td style="text-align: center;">
                            <button class="btn-icon btn-icon-danger btn-delete-dpo" data-index="${pair.index}" title="Delete DPO pair">
                                <i data-lucide="trash" style="width: 14px; height: 14px; color: var(--danger);"></i>
                            </button>
                        </td>
                    `;
                    body.appendChild(tr);
                });
                
                lucide.createIcons();
                
                body.querySelectorAll('.btn-delete-dpo').forEach(btn => {
                    btn.addEventListener('click', async () => {
                        const idx = btn.getAttribute('data-index');
                        if (confirm(`Are you sure you want to delete this DPO preference pair?`)) {
                            try {
                                const deleteRes = await fetch(`/api/dpo/${idx}`, { method: 'DELETE' });
                                const deleteData = await deleteRes.json();
                                if (deleteRes.status === 200 && deleteData.status === 'success') {
                                    alert(deleteData.message);
                                    loadCompilerDpoList();
                                    loadDatasetInfo();
                                    loadDpoDiagnostics();
                                } else {
                                    alert(`Error: ${deleteData.detail || 'Delete failed'}`);
                                }
                            } catch (e) {
                                alert(`Network Error: ${e.message}`);
                            }
                        }
                    });
                });
            } else {
                body.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Error: ${data.detail || 'Failed to load DPO pairs'}</td></tr>`;
            }
        } catch (e) {
            body.innerHTML = `<tr><td colspan="4" class="text-center text-danger">Network Error: ${e.message}</td></tr>`;
        }
    }

    // 14D. Playground Map Integration (Leaflet.js) & Quick Prompts
    function initPlaygroundMap() {
        try {
            // Sri Lanka coordinates: lat 7.8731, lng 80.7718. Zoom level 7-8 is perfect.
            playgroundMap = L.map('playground-map', {
                zoomControl: false
            }).setView([7.8731, 80.7718], 7.5);
            
            L.control.zoom({ position: 'bottomright' }).addTo(playgroundMap);

            // Use CartoDB Dark Matter tile layer for a beautiful glowing dark theme
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 20
            }).addTo(playgroundMap);
            
            playgroundMap.on('popupopen', () => {
                lucide.createIcons();
            });
            
            // Invalidate size on tab switch to avoid Leaflet hidden rendering glitches
            const playgroundNavBtn = document.querySelector('.nav-btn[data-tab="playground"]');
            if (playgroundNavBtn) {
                playgroundNavBtn.addEventListener('click', () => {
                    setTimeout(() => {
                        if (playgroundMap) {
                            playgroundMap.invalidateSize();
                        }
                    }, 200);
                });
            }
        } catch (error) {
            console.error("Leaflet map initialization failed:", error);
        }
    }

    function parseAndPlotLocations(responseText) {
        if (!playgroundMap || !cachedPlaces || cachedPlaces.length === 0) return;

        // Clear existing markers
        mapMarkers.forEach(m => playgroundMap.removeLayer(m));
        mapMarkers = [];

        const matchedPlaces = [];
        const textToSearch = responseText.toLowerCase();

        cachedPlaces.forEach(place => {
            const nameLower = place.name.toLowerCase();
            if (textToSearch.includes(nameLower)) {
                matchedPlaces.push(place);
            }
        });

        if (matchedPlaces.length === 0) return;

        const bounds = [];
        matchedPlaces.forEach(place => {
            const lat = parseFloat(place.lat);
            const lng = parseFloat(place.lng);
            if (isNaN(lat) || isNaN(lng)) return;

            let markerColor = '#8b5cf6'; // Default purple
            if (place.safety_level === 'Moderate') markerColor = '#f59e0b'; // Gold
            if (place.safety_level === 'Dangerous') markerColor = '#ef4444'; // Red

            const customIcon = L.divIcon({
                className: 'custom-map-pin',
                html: `<div class="pin-pulse" style="background-color: ${markerColor}; box-shadow: 0 0 10px ${markerColor};"></div>`,
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            });

            const popupContent = `
                <div class="map-popup-card">
                    <h4>${place.name}</h4>
                    <span class="badge ${place.safety_level === 'Safe' ? 'badge-purple' : (place.safety_level === 'Moderate' ? 'badge-gold' : 'btn-icon-danger')}" style="font-size:9px; padding:2px 6px;">
                        ${place.safety_level}
                    </span>
                    <p style="margin: 5px 0 0 0; font-size: 11px; color: #a3a6b8; line-height: 1.4;">${place.description.substring(0, 100)}...</p>
                    <div style="margin-top: 6px; font-size: 10px; color: var(--text-secondary); font-weight: 500; margin-bottom: 8px;">
                        District: ${place.district_id} | Category: ${place.category_id}
                    </div>
                    <button type="button" class="btn btn-secondary btn-voice-tour" style="padding:4px 8px; font-size:11px; width:100%; height:auto; display:flex; align-items:center; justify-content:center; gap:4px; border-color: rgba(139, 92, 246, 0.25); color: #c084fc;" data-place-id="${place.id}">
                        <i data-lucide="volume-2" style="width:12px; height:12px;"></i>
                        <span>Play Voice Tour</span>
                    </button>
                </div>
            `;

            const marker = L.marker([lat, lng], { icon: customIcon })
                .addTo(playgroundMap)
                .bindPopup(popupContent);
                
            mapMarkers.push(marker);
            bounds.push([lat, lng]);
        });

        if (bounds.length > 0) {
            playgroundMap.fitBounds(bounds, {
                padding: [50, 50],
                maxZoom: 13
            });
        }
    }

    // Mode Switcher Buttons click handler
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.getAttribute('data-mode');
            setPlaygroundMode(mode);
        });
    });

    // Autocomplete slash command logic
    let autocompleteVisible = false;
    let filteredCommands = [];
    let focusedIndex = -1;

    if (playgroundPrompt) {
        playgroundPrompt.addEventListener('input', (e) => {
            const val = e.target.value;
            const caretPos = e.target.selectionStart;
            
            const textUpToCaret = val.substring(0, caretPos);
            const words = textUpToCaret.split(/\s+/);
            const lastWord = words[words.length - 1];

            if (lastWord.startsWith('/')) {
                const query = lastWord.toLowerCase();
                filteredCommands = activeSlashCommands.filter(cmd => cmd.cmd.startsWith(query));
                
                if (filteredCommands.length > 0) {
                    showAutocompleteDropdown(filteredCommands);
                } else {
                    hideAutocompleteDropdown();
                }
            } else {
                hideAutocompleteDropdown();
            }
        });

        playgroundPrompt.addEventListener('keydown', (e) => {
            if (autocompleteVisible) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    focusedIndex = (focusedIndex + 1) % filteredCommands.length;
                    highlightDropdownItem(focusedIndex);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    focusedIndex = (focusedIndex - 1 + filteredCommands.length) % filteredCommands.length;
                    highlightDropdownItem(focusedIndex);
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    if (focusedIndex >= 0 && focusedIndex < filteredCommands.length) {
                        selectAutocompleteItem(filteredCommands[focusedIndex]);
                    }
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    hideAutocompleteDropdown();
                }
            } else {
                if (e.key === 'Enter' && e.ctrlKey) {
                    e.preventDefault();
                    btnRunInference.click();
                }
            }
        });
    }

    function showAutocompleteDropdown(commands) {
        if (!slashDropdown) return;
        slashDropdown.innerHTML = '';
        focusedIndex = 0;
        autocompleteVisible = true;
        slashDropdown.classList.remove('hidden');

        commands.forEach((cmd, idx) => {
            const item = document.createElement('div');
            item.className = 'slash-dropdown-item';
            if (idx === 0) item.classList.add('focused');
            item.innerHTML = `
                <div>
                    <span class="slash-dropdown-command" style="color: white; font-weight:700;">${cmd.cmd}</span>
                    <span style="margin: 0 8px; opacity: 0.3;">|</span>
                    <span class="slash-dropdown-purpose">${cmd.purpose}</span>
                </div>
                <span class="slash-dropdown-badge tag-${cmd.color}">${cmd.mode.toUpperCase()}</span>
            `;
            item.addEventListener('click', () => selectAutocompleteItem(cmd));
            slashDropdown.appendChild(item);
        });
    }

    function hideAutocompleteDropdown() {
        if (!slashDropdown) return;
        slashDropdown.classList.add('hidden');
        autocompleteVisible = false;
        focusedIndex = -1;
        filteredCommands = [];
    }

    function highlightDropdownItem(index) {
        if (!slashDropdown) return;
        const items = slashDropdown.querySelectorAll('.slash-dropdown-item');
        items.forEach((item, idx) => {
            if (idx === index) {
                item.classList.add('focused');
                item.scrollIntoView({ block: 'nearest' });
            } else {
                item.classList.remove('focused');
            }
        });
    }

    function selectAutocompleteItem(cmd) {
        if (!playgroundPrompt) return;
        const val = playgroundPrompt.value;
        const caretPos = playgroundPrompt.selectionStart;
        const textBeforeCaret = val.substring(0, caretPos);
        const textAfterCaret = val.substring(caretPos);
        
        const lastSlashIdx = textBeforeCaret.lastIndexOf('/');
        const newTextBeforeCaret = textBeforeCaret.substring(0, lastSlashIdx) + cmd.cmd + " ";
        
        playgroundPrompt.value = newTextBeforeCaret + textAfterCaret;
        playgroundPrompt.selectionStart = playgroundPrompt.selectionEnd = newTextBeforeCaret.length;
        playgroundPrompt.focus();

        setPlaygroundMode(cmd.mode);
        hideAutocompleteDropdown();
    }

    // 14E. Dynamic Custom Mode Management (localStorage & modal events)
    const customModeModal = document.getElementById('custom-mode-modal');
    const customModeModalClose = document.getElementById('custom-mode-modal-close');
    const btnCancelCustomMode = document.getElementById('btn-cancel-custom-mode');
    const btnAddCustomMode = document.getElementById('btn-add-custom-mode');
    const customModeForm = document.getElementById('custom-mode-form');
    
    const customModeCmd = document.getElementById('custom-mode-cmd');
    const customModeName = document.getElementById('custom-mode-name');
    const customModePurpose = document.getElementById('custom-mode-purpose');
    const customModeColor = document.getElementById('custom-mode-color');
    const customModeSystem = document.getElementById('custom-mode-system');
    const customModeQuickPrompts = document.getElementById('custom-mode-quick-prompts');

    async function loadCustomModesFromBackend() {
        try {
            const res = await fetch('/api/custom-modes');
            const data = await res.json();
            return data;
        } catch (e) {
            console.error("Error loading custom modes:", e);
            return [];
        }
    }
    
    async function saveCustomModeToBackend(modeObj) {
        try {
            const res = await fetch('/api/custom-modes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modeObj)
            });
            return await res.json();
        } catch (e) {
            console.error("Error saving custom mode:", e);
            return { status: 'error', detail: e.message };
        }
    }

    async function renderCustomModesUI() {
        const spacer = document.getElementById('custom-modes-spacer');
        if (!spacer) return;
        
        // Clear previous buttons
        document.querySelectorAll('.custom-mode-btn').forEach(btn => btn.remove());
        
        const customModes = await loadCustomModesFromBackend();
        
        // Reset dynamic lists
        activeModePrompts = { ...BASE_MODE_PROMPTS };
        activeModeQuickPrompts = { ...BASE_MODE_QUICK_PROMPTS };
        activeSlashCommands = [ ...BASE_SLASH_COMMANDS ];
        
        customModes.forEach(m => {
            activeModePrompts[m.mode] = m.system;
            activeModeQuickPrompts[m.mode] = m.quickPrompts.map(qp => ({ label: qp, prompt: qp }));
            activeSlashCommands.push({
                cmd: m.cmd,
                mode: m.mode,
                purpose: m.purpose,
                color: m.color
            });
            
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `mode-btn custom-mode-btn mode-btn-${m.color}`;
            btn.setAttribute('data-mode', m.mode);
            btn.title = `${m.name} Mode`;
            btn.innerHTML = `<i data-lucide="sparkles" style="width: 14px; height: 14px;"></i> <span>${m.name}</span>`;
            
            btn.addEventListener('click', () => {
                setPlaygroundMode(m.mode);
            });
            
            spacer.appendChild(btn);
        });
        
        lucide.createIcons();

        // Re-bind click event for default buttons so everything behaves unified
        const defaultModeButtons = document.querySelectorAll('.mode-btn');
        defaultModeButtons.forEach(btn => {
            if (btn.id === 'btn-add-custom-mode') return;
            btn.replaceWith(btn.cloneNode(true));
        });
        
        const freshModeBtns = document.querySelectorAll('.mode-btn');
        freshModeBtns.forEach(btn => {
            if (btn.id === 'btn-add-custom-mode') return;
            btn.addEventListener('click', () => {
                const mode = btn.getAttribute('data-mode');
                setPlaygroundMode(mode);
            });
        });
    }

    if (btnAddCustomMode) {
        btnAddCustomMode.addEventListener('click', () => {
            customModeForm.reset();
            customModeModal.classList.remove('hidden');
        });
    }
    
    if (customModeModalClose) {
        customModeModalClose.addEventListener('click', () => customModeModal.classList.add('hidden'));
    }
    
    if (btnCancelCustomMode) {
        btnCancelCustomMode.addEventListener('click', () => customModeModal.classList.add('hidden'));
    }
    
    if (customModeForm) {
        customModeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            let cmdVal = customModeCmd.value.trim().toLowerCase();
            if (!cmdVal.startsWith('/')) {
                cmdVal = '/' + cmdVal;
            }
            const modeVal = cmdVal.substring(1).replace(/[^a-z0-9]/g, '');
            
            if (['default', 'analyst', 'optimizer', 'refactor', 'database', 'security', 'agent'].includes(modeVal)) {
                alert("Cannot overwrite a system built-in mode.");
                return;
            }
            
            const qpRaw = customModeQuickPrompts.value.split(',').map(s => s.trim()).filter(s => s.length > 0);
            
            const newMode = {
                cmd: cmdVal,
                mode: modeVal,
                name: customModeName.value.trim(),
                purpose: customModePurpose.value.trim(),
                color: customModeColor.value,
                system: customModeSystem.value.trim(),
                quickPrompts: qpRaw
            };
            
            const res = await saveCustomModeToBackend(newMode);
            if (res.status === 'success') {
                customModeModal.classList.add('hidden');
                await renderCustomModesUI();
                setPlaygroundMode(modeVal);
            } else {
                alert("Error saving custom mode: " + (res.detail || "Unknown error"));
            }
        });
    }

    // 14F. Session Chat Export to Markdown
    const btnExportChat = document.getElementById('btn-export-chat');
    if (btnExportChat) {
        btnExportChat.addEventListener('click', () => {
            const bubbles = playgroundResponse.querySelectorAll('.chat-bubble');
            if (bubbles.length === 0 || (bubbles.length === 1 && bubbles[0].classList.contains('empty-state'))) {
                alert("No chat conversation to export.");
                return;
            }
            
            let mdContent = `# Lumen-1 Chat Export - ${new Date().toLocaleDateString()}\n\n`;
            
            bubbles.forEach(b => {
                if (b.classList.contains('empty-state')) return;
                
                const isUser = b.classList.contains('user');
                const senderName = isUser ? "User" : "Lumen-1";
                
                let badgeText = "";
                if (!isUser) {
                    const tagEl = b.querySelector('.chat-bubble-mode-tag');
                    if (tagEl) {
                        badgeText = ` [${tagEl.innerText.trim()}]`;
                    }
                }
                
                let text = '';
                const divs = b.querySelectorAll('div');
                const textParts = [];
                divs.forEach(d => {
                    if (!d.classList.contains('chat-bubble-mode-tag') && 
                        !d.classList.contains('chat-bubble-meta') && 
                        !d.classList.contains('chat-bubble-media') &&
                        !d.classList.contains('spinner') &&
                        d.style.display !== 'none') {
                        if (d.style.whiteSpace === 'pre-wrap' || d.textContent.trim().length > 0) {
                            textParts.push(d.innerText || d.textContent);
                        }
                    }
                });
                text = textParts.join('\n').trim();
                if (!text) {
                    text = b.innerText;
                }
                
                mdContent += `### **${senderName}**${badgeText}\n${text}\n\n---\n\n`;
            });
            
            const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `lumen1_chat_${new Date().toISOString().slice(0,10)}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // Mock metrics button handler
    const btnMockMetrics = document.getElementById('btn-mock-metrics');
    if (btnMockMetrics) {
        btnMockMetrics.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/metrics/mock', { method: 'POST' });
                const data = await response.json();
                if (data.status === 'success') {
                    alert(`Mock metric step ${data.step} generated successfully!\nLoss: ${data.loss}\nLearning Rate: ${data.learning_rate}`);
                } else {
                    alert('Failed to generate mock metrics.');
                }
            } catch (err) {
                alert(`Error generating mock metrics: ${err.message}`);
            }
        });
    }

    // Export telemetry button handler
    const btnExportTelemetry = document.getElementById('btn-export-telemetry');
    if (btnExportTelemetry) {
        btnExportTelemetry.addEventListener('click', () => {
            window.location.href = '/api/telemetry/export';
        });
    }

    // --- CSV Dataset Studio & Validator Logic ---
    let uploadedCsvPath = null;
    let csvChartLanguageInstance = null;
    let csvPreviewSamples = [];

    const csvDropZone = document.getElementById('csv-drop-zone');
    const csvFileInput = document.getElementById('csv-file-input');
    const csvFileBanner = document.getElementById('csv-file-banner');
    const csvBannerName = document.getElementById('csv-banner-name');
    const csvBannerMeta = document.getElementById('csv-banner-meta');
    const btnCsvRemove = document.getElementById('btn-csv-remove');
    const csvMappingWorkshop = document.getElementById('csv-mapping-workshop');
    const csvMapInstruction = document.getElementById('csv-map-instruction');
    const csvMapResponse = document.getElementById('csv-map-response');
    const csvMapInput = document.getElementById('csv-map-input');
    const csvMapSystem = document.getElementById('csv-map-system');
    const btnCsvValidate = document.getElementById('btn-csv-validate');
    
    const csvReportPanel = document.getElementById('csv-report-panel');
    const csvStatTotal = document.getElementById('csv-stat-total');
    const csvStatPassed = document.getElementById('csv-stat-passed');
    const csvStatStatus = document.getElementById('csv-stat-status');
    const csvStatTokens = document.getElementById('csv-stat-tokens');
    
    const csvWordsAvg = document.getElementById('csv-words-avg');
    const csvWordsAvgBar = document.getElementById('csv-words-avg-bar');
    const csvWordsMin = document.getElementById('csv-words-min');
    const csvWordsMax = document.getElementById('csv-words-max');
    
    const csvWarningsBadge = document.getElementById('csv-warnings-badge');
    const csvWarningsLog = document.getElementById('csv-warnings-log');
    
    const csvPreviewChatBody = document.getElementById('csv-preview-chat-body');
    const csvPreviewToggles = document.getElementById('csv-preview-toggles');
    
    const btnCsvApply = document.getElementById('btn-csv-apply');
    const btnCsvStartTrain = document.getElementById('btn-csv-start-train');

    if (csvDropZone) {
        csvDropZone.addEventListener('click', () => {
            csvFileInput.click();
        });

        csvDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            csvDropZone.style.borderColor = 'var(--purple)';
            csvDropZone.style.background = 'rgba(139, 92, 246, 0.05)';
        });

        csvDropZone.addEventListener('dragleave', () => {
            csvDropZone.style.borderColor = 'rgba(139, 92, 246, 0.3)';
            csvDropZone.style.background = 'rgba(255,255,255,0.01)';
        });

        csvDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            csvDropZone.style.borderColor = 'rgba(139, 92, 246, 0.3)';
            csvDropZone.style.background = 'rgba(255,255,255,0.01)';
            
            if (e.dataTransfer.files.length > 0) {
                handleCsvFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (csvFileInput) {
        csvFileInput.addEventListener('change', () => {
            if (csvFileInput.files.length > 0) {
                handleCsvFile(csvFileInput.files[0]);
            }
        });
    }

    async function handleCsvFile(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a valid CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            csvDropZone.innerHTML = `
                <div class="spinner" style="width: 32px; height: 32px; display: inline-block;"></div>
                <p style="margin-top: 10px; color: var(--text-secondary);">Uploading and analyzing CSV headers...</p>
            `;
            
            const res = await fetch('/api/dataset/upload-csv', {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            
            if (res.status === 200 && data.status === 'success') {
                uploadedCsvPath = data.file_path;
                csvBannerName.innerText = data.filename;
                csvBannerMeta.innerText = `${(file.size / 1024).toFixed(1)} KB • ${data.total_rows} Rows`;
                csvFileBanner.classList.remove('hidden');
                csvDropZone.classList.add('hidden');
                csvMapSystem.value = "You are Lumen-1, an advanced AI travel assistant for Sri Lanka. When answering questions, think deeply and structure your reasoning step-by-step to provide the best, most accurate, and safest output.";
                populateDropdowns(data.headers);
                csvMappingWorkshop.classList.remove('hidden');
            } else {
                alert(`Upload failed: ${data.detail || 'Unknown error'}`);
                resetCsvUploader();
            }
        } catch (e) {
            alert(`Error uploading file: ${e.message}`);
            resetCsvUploader();
        }
    }

    function populateDropdowns(headers) {
        csvMapInstruction.innerHTML = '';
        csvMapResponse.innerHTML = '';
        csvMapInput.innerHTML = '<option value="">-- None --</option>';
        
        headers.forEach(h => {
            const opt1 = document.createElement('option');
            opt1.value = h;
            opt1.innerText = h;
            csvMapInstruction.appendChild(opt1);
            
            const opt2 = document.createElement('option');
            opt2.value = h;
            opt2.innerText = h;
            csvMapResponse.appendChild(opt2);
            
            const opt3 = document.createElement('option');
            opt3.value = h;
            opt3.innerText = h;
            csvMapInput.appendChild(opt3);
        });

        const instMatch = headers.find(h => /instruction|prompt|question|query|q|text/i.test(h));
        if (instMatch) csvMapInstruction.value = instMatch;
        
        const respMatch = headers.find(h => /response|answer|output|res|reply|a/i.test(h));
        if (respMatch) csvMapResponse.value = respMatch;
        
        const inputMatch = headers.find(h => /input|context/i.test(h));
        if (inputMatch) csvMapInput.value = inputMatch;
    }

    if (btnCsvRemove) {
        btnCsvRemove.addEventListener('click', () => {
            resetCsvUploader();
        });
    }

    function resetCsvUploader() {
        uploadedCsvPath = null;
        csvFileInput.value = '';
        csvFileBanner.classList.add('hidden');
        csvMappingWorkshop.classList.add('hidden');
        csvReportPanel.classList.add('hidden');
        btnCsvStartTrain.disabled = true;
        
        csvDropZone.classList.remove('hidden');
        csvDropZone.innerHTML = `
            <i data-lucide="upload-cloud" style="width: 48px; height: 48px; color: var(--purple); margin-bottom: 12px; opacity: 0.8; display: inline-block;"></i>
            <h4 style="margin-bottom: 8px; color: white;">Drag & Drop your training CSV file here</h4>
            <p style="margin: 0; font-size: 13px; color: var(--text-secondary);">or <span style="color: var(--purple); text-decoration: underline; font-weight: 600;">browse files</span> from your computer</p>
        `;
        lucide.createIcons();
    }

    if (btnCsvValidate) {
        btnCsvValidate.addEventListener('click', async () => {
            if (!uploadedCsvPath) return;

            const instruction_col = csvMapInstruction.value;
            const response_col = csvMapResponse.value;
            const input_col = csvMapInput.value;
            const system_prompt = csvMapSystem.value;

            if (instruction_col === response_col) {
                alert('Instruction and Response columns cannot be the same!');
                return;
            }

            try {
                btnCsvValidate.disabled = true;
                btnCsvValidate.innerHTML = `<div class="spinner" style="width: 14px; height: 14px; display: inline-block; margin-right: 5px;"></div> Validating...`;

                const res = await fetch('/api/dataset/validate-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_path: uploadedCsvPath,
                        instruction_col,
                        response_col,
                        input_col,
                        system_prompt
                    })
                });

                const data = await res.json();

                if (res.status === 200) {
                    renderValidationReport(data);
                } else {
                    alert(`Validation failed: ${data.detail || 'Server encountered an error.'}`);
                }
            } catch (e) {
                alert(`Network error during validation: ${e.message}`);
            } finally {
                btnCsvValidate.disabled = false;
                btnCsvValidate.innerHTML = `<i data-lucide="shield-check"></i> <span>Verify & Run Quality Check</span>`;
                lucide.createIcons();
            }
        });
    }

    function renderValidationReport(data) {
        csvReportPanel.classList.remove('hidden');
        csvStatTotal.innerText = data.total_rows;
        csvStatPassed.innerText = data.passed_rows;
        csvStatTokens.innerText = data.stats.est_tokens.toLocaleString();
        
        const statusEl = csvStatStatus;
        statusEl.innerText = data.status.toUpperCase();
        
        const iconContainer = document.getElementById('csv-icon-status');
        
        if (data.status === 'valid') {
            statusEl.className = 'stat-value text-glowing-teal';
            iconContainer.className = 'stat-icon stat-icon-teal';
            iconContainer.innerHTML = '<i data-lucide="check-circle"></i>';
        } else if (data.status === 'warnings') {
            statusEl.className = 'stat-value text-glowing-gold';
            iconContainer.className = 'stat-icon stat-icon-gold';
            iconContainer.innerHTML = '<i data-lucide="alert-triangle"></i>';
        } else {
            statusEl.className = 'stat-value text-glowing-red';
            iconContainer.className = 'stat-icon';
            iconContainer.style.background = 'rgba(239, 68, 68, 0.12)';
            iconContainer.style.color = 'var(--danger)';
            iconContainer.style.border = '1px solid rgba(239, 68, 68, 0.2)';
            iconContainer.innerHTML = '<i data-lucide="x-circle"></i>';
        }

        csvWordsAvg.innerText = data.stats.avg_words;
        csvWordsMin.innerText = data.stats.min_words;
        csvWordsMax.innerText = data.stats.max_words;
        
        const percentAvg = data.stats.max_words > 0 ? (data.stats.avg_words / data.stats.max_words * 100) : 0;
        csvWordsAvgBar.style.width = `${percentAvg}%`;

        csvWarningsBadge.innerText = `${data.warnings.length} Warnings`;
        if (data.warnings.length > 0) {
            csvWarningsBadge.style.background = 'rgba(239, 68, 68, 0.1)';
            csvWarningsBadge.style.color = 'var(--danger)';
            
            csvWarningsLog.innerHTML = '';
            data.warnings.forEach(warn => {
                const item = document.createElement('div');
                item.className = warn.includes('CRITICAL') ? 'diagnostic-alert' : 'diagnostic-warning';
                item.innerText = warn;
                csvWarningsLog.appendChild(item);
            });
        } else {
            csvWarningsBadge.style.background = 'rgba(16, 185, 129, 0.1)';
            csvWarningsBadge.style.color = '#10b981';
            csvWarningsLog.innerHTML = '<div class="text-center" style="color: var(--text-secondary); width: 100%;">No issues found. Your dataset is completely clean!</div>';
        }

        csvPreviewSamples = data.preview;
        renderPreviewBubble(0);
        renderLanguageChart(data.stats.languages.sinhala, data.stats.languages.english);
        csvReportPanel.scrollIntoView({ behavior: 'smooth' });
        lucide.createIcons();
    }

    function renderPreviewBubble(idx) {
        const sample = csvPreviewSamples[idx];
        csvPreviewChatBody.innerHTML = '';

        if (!sample) {
            csvPreviewChatBody.innerHTML = '<div class="text-center" style="color: var(--text-secondary); padding: 20px;">No preview available.</div>';
            return;
        }

        sample.messages.forEach(msg => {
            const bubble = document.createElement('div');
            bubble.className = `chat-bubble chat-bubble-${msg.role === 'user' ? 'user' : (msg.role === 'system' ? 'system' : 'assistant')}`;
            
            if (msg.role === 'system') {
                bubble.style.background = 'rgba(255,255,255,0.04)';
                bubble.style.borderColor = 'rgba(255,255,255,0.05)';
                bubble.style.alignSelf = 'center';
                bubble.style.maxWidth = '90%';
            }

            const header = document.createElement('div');
            header.style.fontSize = '10px';
            header.style.color = 'var(--text-secondary)';
            header.style.textTransform = 'uppercase';
            header.style.marginBottom = '4px';
            header.innerText = msg.role.toUpperCase();
            bubble.appendChild(header);

            const content = document.createElement('div');
            content.style.whiteSpace = 'pre-wrap';
            content.textContent = msg.content;
            bubble.appendChild(content);

            csvPreviewChatBody.appendChild(bubble);
        });

        const toggles = csvPreviewToggles.querySelectorAll('button');
        toggles.forEach((t, i) => {
            if (i === idx) t.classList.add('active');
            else t.classList.remove('active');
        });
    }

    if (csvPreviewToggles) {
        csvPreviewToggles.addEventListener('click', (e) => {
            const btn = e.target.closest('button');
            if (btn) {
                const idx = parseInt(btn.getAttribute('data-sample-idx'));
                renderPreviewBubble(idx);
            }
        });
    }

    function renderLanguageChart(sinhala, english) {
        if (csvChartLanguageInstance) {
            csvChartLanguageInstance.destroy();
        }

        const canvasEl = document.getElementById('csvChartLanguage');
        if (!canvasEl) return;
        
        const ctx = canvasEl.getContext('2d');
        csvChartLanguageInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Sinhala', 'English'],
                datasets: [{
                    data: [sinhala, english],
                    backgroundColor: ['#8b5cf6', '#14b8a6'],
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.05)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#8e9bb3', font: { family: 'Outfit', size: 12 } }
                    }
                },
                cutout: '70%'
            }
        });
    }

    if (btnCsvApply) {
        btnCsvApply.addEventListener('click', async () => {
            if (!uploadedCsvPath) return;

            const instruction_col = csvMapInstruction.value;
            const response_col = csvMapResponse.value;
            const input_col = csvMapInput.value;
            const system_prompt = csvMapSystem.value;

            try {
                btnCsvApply.disabled = true;
                btnCsvApply.innerHTML = `<div class="spinner" style="width: 14px; height: 14px; display: inline-block; margin-right: 5px;"></div> Applying...`;

                const res = await fetch('/api/dataset/apply-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_path: uploadedCsvPath,
                        instruction_col,
                        response_col,
                        input_col,
                        system_prompt
                    })
                });

                const data = await res.json();

                if (res.status === 200) {
                    alert(`Success! Dataset applied.\n${data.message}`);
                    btnCsvStartTrain.disabled = false;
                    loadDatasetInfo();
                    loadConfigs();
                } else {
                    alert(`Error applying dataset: ${data.detail || 'Unknown error'}`);
                }
            } catch (e) {
                alert(`Network error applying dataset: ${e.message}`);
            } finally {
                btnCsvApply.disabled = false;
                btnCsvApply.innerHTML = `<i data-lucide="check-square"></i> <span>Apply to SFT Pipeline</span>`;
                lucide.createIcons();
            }
        });
    }

    if (btnCsvStartTrain) {
        btnCsvStartTrain.addEventListener('click', () => {
            if (trainingRunning) {
                alert('Training is already running. Please stop the current run first.');
                return;
            }
            
            btnStartSft.click();
            
            const overviewBtn = Array.from(navButtons).find(btn => btn.getAttribute('data-tab') === 'overview');
            if (overviewBtn) {
                overviewBtn.click();
            }
        });
    }

    // WebSocket status connection
    let socket = null;
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/ws/status`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log("🔌 Connected to real-time status WebSocket");
            globalStatusText.innerText = "Status: Connected";
            globalStatusDot.className = "status-dot status-idle";
        };
        
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'security_alert') {
                    showToast(data.message, 'danger', `Threat Blocked: ${data.threat_level}`);
                    loadSecurityLogs();
                    return;
                }
                
                // 1. Update Status
                if (data.status) {
                    updateStatusUI(data.status.status, data.status.type, data.status.error);
                }
                
                // 2. Update System stats
                if (data.system) {
                    sysCpuVal.innerText = `${data.system.cpu}%`;
                    sysCpuBar.style.width = `${data.system.cpu}%`;

                    const ramShort = document.getElementById('sys-ram-val-short');
                    if (ramShort) ramShort.innerText = `${data.system.ram.percent}%`;
                    sysRamVal.innerText = `${data.system.ram.used} / ${data.system.ram.total} GB (${data.system.ram.percent}%)`;
                    sysRamBar.style.width = `${data.system.ram.percent}%`;

                    if (data.system.gpus && data.system.gpus.length > 0) {
                        gpuContainer.innerHTML = '';
                        data.system.gpus.forEach(gpu => {
                            const gpuCard = document.createElement('div');
                            gpuCard.className = 'gpu-card';
                            gpuCard.innerHTML = `
                                <div class="gpu-header">
                                    <span>${gpu.name}</span>
                                    <span class="badge badge-purple">CUDA</span>
                                </div>
                                <div class="gpu-grid">
                                    <div class="gpu-stat-item">
                                        <span>Utilization</span>
                                        <p>${gpu.utilization}%</p>
                                    </div>
                                    <div class="gpu-stat-item">
                                        <span>Temp</span>
                                        <p>${gpu.temperature}°C</p>
                                    </div>
                                    <div class="gpu-stat-item">
                                        <span>VRAM (Used/Total)</span>
                                        <p>${(gpu.memory_used / 1024).toFixed(2)} / ${(gpu.memory_total / 1024).toFixed(2)} GB</p>
                                    </div>
                                </div>
                                <div class="metric-meter-group" style="margin-top: 10px;">
                                    <div class="meter-bar-bg">
                                        <div class="meter-bar-fill meter-teal" style="width: ${gpu.memory_percent}%;"></div>
                                    </div>
                                </div>
                            `;
                            gpuContainer.appendChild(gpuCard);
                        });
                    } else {
                        gpuContainer.innerHTML = `
                            <div class="empty-state">
                                <i data-lucide="cpu"></i>
                                <p>No GPUs identified. Verify CUDA drivers or GPUtil package installation.</p>
                            </div>
                        `;
                        lucide.createIcons();
                    }
                }
                
                // 3. Update Logs
                if (data.logs && data.logs !== logHistory) {
                    logHistory = data.logs;
                    renderConsoleLogs(data.logs);
                }
                
                // 4. Update Metrics
                if (data.metrics && data.metrics.history && data.metrics.history.length > 0) {
                    const latest = data.metrics.history[data.metrics.history.length - 1];
                    
                    valLoss.innerText = latest.loss ? latest.loss.toFixed(4) : (latest.train_loss ? latest.train_loss.toFixed(4) : "N/A");
                    valLr.innerText = latest.learning_rate ? latest.learning_rate.toExponential(2) : "N/A";
                    valEpoch.innerText = latest.epoch ? latest.epoch.toFixed(2) : "N/A";
                    valStep.innerText = `${latest.step} / ${latest.max_steps || 'N/A'}`;

                    if (latest.max_steps) {
                        const percent = Math.min(Math.round((latest.step / latest.max_steps) * 100), 100);
                        progressBar.style.width = `${percent}%`;
                        progressPercentage.innerText = `${percent}%`;
                    }

                    const steps = data.metrics.history.map(h => h.step);
                    const losses = data.metrics.history.map(h => h.loss || h.train_loss || 0);
                    
                    lossChart.data.labels = steps;
                    lossChart.data.datasets[0].data = losses;
                    lossChart.update();
                }
            } catch (e) {
                console.error("Error parsing WebSocket message:", e);
            }
        };
        
        socket.onclose = () => {
            console.log("🔌 WebSocket status disconnected. Reconnecting in 5 seconds...");
            globalStatusText.innerText = "Status: Reconnecting...";
            globalStatusDot.className = "status-dot status-idle";
            setTimeout(connectWebSocket, 5000);
        };
        
        socket.onerror = (e) => {
            console.error("🔌 WebSocket error:", e);
        };
    }

    // --- 14G. Model Exporter Controls & Live Logging ---
    const btnStartExport = document.getElementById('btn-start-export');
    const btnStopExport = document.getElementById('btn-stop-export');
    const exporterForm = document.getElementById('exporter-form');
    const exportBaseModel = document.getElementById('export-base-model');
    const exportAdapter = document.getElementById('export-adapter');
    const exportOutput = document.getElementById('export-output');
    const exportFormat = document.getElementById('export-format');
    const exporterConsole = document.getElementById('exporter-console');
    const exporterStatusBadge = document.getElementById('exporter-status-badge');
    const exporterLogBadge = document.getElementById('exporter-log-badge');

    let exportRunning = false;
    let exporterStatusInterval = null;
    let exporterLogInterval = null;
    let exportLogHistory = '';

    async function checkExporterStatus() {
        try {
            const res = await fetch('/api/exporter/status');
            const data = await res.json();
            updateExporterStatusUI(data.status, data.error);
        } catch (e) {
            console.error("Error fetching exporter status:", e);
        }
    }

    function updateExporterStatusUI(status, error) {
        if (status === 'running') {
            exportRunning = true;
            if (btnStartExport) btnStartExport.disabled = true;
            if (btnStopExport) btnStopExport.disabled = false;
            if (exporterStatusBadge) {
                exporterStatusBadge.innerText = 'Merging weights...';
                exporterStatusBadge.className = 'badge badge-purple';
            }
            startExporterIntervals();
        } else {
            exportRunning = false;
            if (btnStartExport) btnStartExport.disabled = false;
            if (btnStopExport) btnStopExport.disabled = true;
            
            if (status === 'completed') {
                if (exporterStatusBadge) {
                    exporterStatusBadge.innerText = 'Completed successfully! ✅';
                    exporterStatusBadge.className = 'badge badge-green';
                }
                stopExporterIntervals();
                fetchExporterLogs();
            } else if (status === 'error') {
                if (exporterStatusBadge) {
                    exporterStatusBadge.innerText = 'Merge failed ❌';
                    exporterStatusBadge.className = 'badge badge-red';
                }
                stopExporterIntervals();
                fetchExporterLogs();
            } else {
                if (exporterStatusBadge) {
                    exporterStatusBadge.innerText = 'Exporter Idle';
                    exporterStatusBadge.className = 'badge';
                }
                stopExporterIntervals();
            }
        }
    }

    async function fetchExporterLogs() {
        try {
            const res = await fetch('/api/exporter/logs');
            const data = await res.json();
            if (data.logs && data.logs !== exportLogHistory) {
                exportLogHistory = data.logs;
                renderExporterLogs(data.logs);
            }
        } catch (e) {
            console.error("Error fetching exporter logs:", e);
        }
    }

    function renderExporterLogs(logsText) {
        if (!exporterConsole) return;
        exporterConsole.innerHTML = '';
        const lines = logsText.split('\n');
        
        lines.forEach(line => {
            if (!line.trim()) return;
            const div = document.createElement('div');
            div.className = 'log-line';
            if (line.includes('ERROR') || line.includes('❌') || line.includes('Fail')) {
                div.style.color = '#f87171';
            } else if (line.includes('WARNING') || line.includes('⚠️')) {
                div.style.color = '#fbbf24';
            } else if (line.includes('successfully') || line.includes('✅') || line.includes('Saved')) {
                div.style.color = '#34d399';
            } else {
                div.style.color = '#c4c8d4';
            }
            div.textContent = line;
            exporterConsole.appendChild(div);
        });
        exporterConsole.scrollTop = exporterConsole.scrollHeight;
    }

    function startExporterIntervals() {
        if (!exporterStatusInterval) {
            exporterStatusInterval = setInterval(checkExporterStatus, 2000);
        }
        if (!exporterLogInterval) {
            exporterLogInterval = setInterval(fetchExporterLogs, 2000);
        }
    }

    function stopExporterIntervals() {
        if (exporterStatusInterval) {
            clearInterval(exporterStatusInterval);
            exporterStatusInterval = null;
        }
        if (exporterLogInterval) {
            clearInterval(exporterLogInterval);
            exporterLogInterval = null;
        }
    }

    if (btnStartExport) {
        btnStartExport.addEventListener('click', async () => {
            if (exportRunning) return;
            const base_model = exportBaseModel.value.trim();
            const adapter = exportAdapter.value.trim();
            const output = exportOutput.value.trim();
            const format = exportFormat ? exportFormat.value.trim() : 'safetensors';
            
            if (!base_model || !adapter || !output) {
                alert("Please fill in all exporter configurations.");
                return;
            }

            exporterConsole.innerHTML = `<div class="log-line" style="color: #60a5fa;">Initiating weights merging/quantization (${format.toUpperCase()}) process...</div>`;
            
            try {
                const res = await fetch('/api/exporter/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base_model, adapter, output, format })
                });
                const data = await res.json();
                if (res.status === 200 && data.status === 'success') {
                    exportRunning = true;
                    updateExporterStatusUI('running', null);
                } else {
                    alert(`Error starting merge: ${data.detail || 'Unknown error'}`);
                }
            } catch (e) {
                alert(`Network error starting merge: ${e.message}`);
            }
        });
    }

    if (btnStopExport) {
        btnStopExport.addEventListener('click', async () => {
            if (confirm("Are you sure you want to terminate the active merging process?")) {
                try {
                    const res = await fetch('/api/exporter/stop', { method: 'POST' });
                    if (res.status === 200) {
                        exportRunning = false;
                        updateExporterStatusUI('idle', null);
                        const div = document.createElement('div');
                        div.style.color = '#ef4444';
                        div.textContent = '⚠️ Merging process manually terminated.';
                        exporterConsole.appendChild(div);
                        exporterConsole.scrollTop = exporterConsole.scrollHeight;
                    }
                } catch (e) {
                    alert(`Error terminating process: ${e.message}`);
                }
            }
        });
    }

    // --- 14H. Interactive Sri Lankan District Heatmap ---
    const DISTRICT_COORDINATES = {
        "colombo": [6.9271, 79.8612],
        "gampaha": [7.0873, 79.9924],
        "kalutara": [6.5854, 79.9607],
        "kandy": [7.2906, 80.6337],
        "matale": [7.4675, 80.6234],
        "nuwara eliya": [6.9497, 80.7891],
        "galle": [6.0535, 80.2210],
        "matara": [5.9549, 80.5550],
        "hambantota": [6.1246, 81.1185],
        "jaffna": [9.6615, 80.0255],
        "kilinochchi": [9.3803, 80.3992],
        "mannar": [8.9819, 79.9044],
        "vavuniya": [8.7542, 80.4982],
        "mullaitivu": [9.2671, 80.8142],
        "batticaloa": [7.7170, 81.7010],
        "ampara": [7.3018, 81.6747],
        "trincomalee": [8.5874, 81.2152],
        "kurunegala": [7.4863, 80.3647],
        "puttalam": [8.0362, 79.8283],
        "anuradhapura": [8.3114, 80.4037],
        "polonnaruwa": [7.9397, 81.0004],
        "badulla": [6.9934, 81.0550],
        "moneragala": [6.8724, 81.3507],
        "ratnapura": [6.6828, 80.3992],
        "kegalle": [7.2513, 80.3464]
    };

    const btnToggleHeatmap = document.getElementById('btn-toggle-heatmap');
    let heatmapCircles = [];
    let heatmapMode = false;

    function drawHeatmap() {
        clearHeatmap();
        if (!cachedPlaces || cachedPlaces.length === 0) return;
        
        const counts = {};
        cachedPlaces.forEach(p => {
            const d = (p.district_id || "").toLowerCase().trim();
            if (d) {
                counts[d] = (counts[d] || 0) + 1;
            }
        });

        for (const d in DISTRICT_COORDINATES) {
            const count = counts[d] || 0;
            if (count > 0) {
                const coords = DISTRICT_COORDINATES[d];
                let circleColor = '#14b8a6'; // Teal
                if (count > 3) circleColor = '#f59e0b'; // Gold
                if (count > 8) circleColor = '#ef4444'; // Red

                // Draw translucent glowing circle
                const radius = Math.min(35000, Math.max(7000, count * 5000));
                const circle = L.circle(coords, {
                    color: circleColor,
                    fillColor: circleColor,
                    fillOpacity: 0.22,
                    radius: radius,
                    weight: 1.5
                }).addTo(playgroundMap);

                circle.bindTooltip(`<strong>${d.toUpperCase()} District</strong><br>${count} spots registered`, {
                    sticky: true,
                    className: 'map-tooltip'
                });
                
                heatmapCircles.push(circle);
            }
        }
    }

    function clearHeatmap() {
        heatmapCircles.forEach(c => playgroundMap.removeLayer(c));
        heatmapCircles = [];
    }

    if (btnToggleHeatmap) {
        btnToggleHeatmap.addEventListener('click', () => {
            heatmapMode = !heatmapMode;
            if (heatmapMode) {
                drawHeatmap();
                btnToggleHeatmap.classList.add('active');
                btnToggleHeatmap.innerHTML = `<i data-lucide="layers" style="width: 12px; height: 12px;"></i> <span>Hide Coverage Map</span>`;
            } else {
                clearHeatmap();
                btnToggleHeatmap.classList.remove('active');
                btnToggleHeatmap.innerHTML = `<i data-lucide="layers" style="width: 12px; height: 12px;"></i> <span>Show Coverage Map</span>`;
            }
            lucide.createIcons();
        });
    }

    // --- 14I. DPO Similarity Checker (SequenceMatcher API) ---
    function debounce(func, timeout = 400) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => { func.apply(this, args); }, timeout);
        };
    }

    const dpoSimilarityWarning = document.getElementById('dpo-similarity-warning');
    const dpoSimPercentage = document.getElementById('dpo-sim-percentage');
    const dpoSimMatchedText = document.getElementById('dpo-sim-matched-text');

    const checkDpoSimilarity = debounce(async (promptVal) => {
        if (!promptVal || !dpoSimilarityWarning) return;
        try {
            const res = await fetch('/api/dpo/check-similarity', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: promptVal })
            });
            const data = await res.json();
            if (data.status === 'success' && data.duplicate) {
                dpoSimilarityWarning.classList.remove('hidden');
                if (dpoSimPercentage) dpoSimPercentage.innerText = `${Math.round(data.max_similarity * 100)}%`;
                if (dpoSimMatchedText) dpoSimMatchedText.innerText = `"${data.match}"`;
            } else {
                if (dpoSimilarityWarning) dpoSimilarityWarning.classList.add('hidden');
            }
        } catch (e) {
            console.error("Error checking DPO similarity:", e);
        }
    }, 400);

    if (compilerPrompt) {
        compilerPrompt.addEventListener('input', (e) => {
            const promptVal = e.target.value.trim();
            if (promptVal) {
                checkDpoSimilarity(promptVal);
            } else {
                if (dpoSimilarityWarning) dpoSimilarityWarning.classList.add('hidden');
            }
        });
    }

    // --- 14J. Dynamic Prompt Template Previewer ---
    const previewerFormat = document.getElementById('previewer-format');
    const previewerSystem = document.getElementById('previewer-system');
    const previewerUser = document.getElementById('previewer-user');
    const previewerAssistant = document.getElementById('previewer-assistant');
    const previewerCode = document.getElementById('previewer-code');
    const previewerTokensBadge = document.getElementById('previewer-tokens-badge');
    const previewerPctText = document.getElementById('previewer-pct-text');
    const previewerPctBar = document.getElementById('previewer-pct-bar');

    function updatePreviewer() {
        if (!previewerCode) return;
        const format = previewerFormat ? previewerFormat.value : 'chatml';
        const sysVal = previewerSystem ? previewerSystem.value : '';
        const userVal = previewerUser ? previewerUser.value : '';
        const assistantVal = previewerAssistant ? previewerAssistant.value : '';
        
        let formatted = '';
        
        if (format === 'chatml') {
            formatted = `<|im_start|>system\n${sysVal}<|im_end|>\n<|im_start|>user\n${userVal}<|im_end|>\n<|im_start|>assistant\n${assistantVal}<|im_end|>`;
        } else if (format === 'alpaca') {
            formatted = `Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n${sysVal}\n\n### Input:\n${userVal}\n\n### Response:\n${assistantVal}`;
        } else if (format === 'sharegpt') {
            const obj = [
                { from: "system", value: sysVal },
                { from: "human", value: userVal },
                { from: "gpt", value: assistantVal }
            ];
            formatted = JSON.stringify(obj, null, 2);
        }
        
        previewerCode.textContent = formatted;
        
        // Token and progress bar metrics
        const estTokens = Math.ceil(formatted.length / 4);
        if (previewerTokensBadge) {
            previewerTokensBadge.innerText = `${estTokens} tokens (est)`;
        }
        const pct = Math.min(100, Math.round((estTokens / 2048) * 100));
        if (previewerPctText) {
            previewerPctText.innerText = `${pct}% of 2048`;
        }
        if (previewerPctBar) {
            previewerPctBar.style.width = `${pct}%`;
        }
    }

    [previewerFormat, previewerSystem, previewerUser, previewerAssistant].forEach(el => {
        if (el) {
            el.addEventListener('input', updatePreviewer);
            el.addEventListener('change', updatePreviewer);
        }
    });

    // --- 14K. Sri Lanka "Safe Travel" Smart Route Simulator ---
    const DISTRICT_GRAPH = {
        "colombo": ["gampaha", "kalutara", "kegalle"],
        "gampaha": ["colombo", "puttalam", "kurunegala", "kegalle"],
        "kalutara": ["colombo", "ratnapura", "galle", "kegalle"],
        "kandy": ["matale", "kegalle", "nuwara eliya", "badulla", "kurunegala"],
        "matale": ["kandy", "kurunegala", "anuradhapura", "polonnaruwa", "badulla"],
        "nuwara eliya": ["kandy", "badulla", "ratnapura", "kegalle"],
        "galle": ["kalutara", "matara", "ratnapura"],
        "matara": ["galle", "hambantota", "ratnapura"],
        "hambantota": ["matara", "moneragala", "ratnapura", "ampara"],
        "jaffna": ["kilinochchi"],
        "kilinochchi": ["jaffna", "mannar", "mullaitivu"],
        "mannar": ["kilinochchi", "mullaitivu", "anuradhapura", "puttalam"],
        "vavuniya": ["mannar", "mullaitivu", "anuradhapura"],
        "mullaitivu": ["kilinochchi", "mannar", "vavuniya", "trincomalee"],
        "batticaloa": ["polonnaruwa", "ampara"],
        "ampara": ["batticaloa", "moneragala", "badulla", "hambantota"],
        "trincomalee": ["mullaitivu", "anuradhapura", "polonnaruwa", "batticaloa"],
        "kurunegala": ["puttalam", "gampaha", "kegalle", "kandy", "matale", "anuradhapura"],
        "puttalam": ["mannar", "anuradhapura", "kurunegala", "gampaha"],
        "anuradhapura": ["puttalam", "mannar", "vavuniya", "trincomalee", "polonnaruwa", "matale", "kurunegala"],
        "polonnaruwa": ["anuradhapura", "trincomalee", "batticaloa", "matale"],
        "badulla": ["matale", "kandy", "nuwara eliya", "ratnapura", "moneragala", "ampara"],
        "moneragala": ["badulla", "ampara", "hambantota", "ratnapura"],
        "ratnapura": ["kegalle", "kalutara", "galle", "matara", "hambantota", "moneragala", "badulla", "nuwara eliya"],
        "kegalle": ["gampaha", "colombo", "kalutara", "ratnapura", "nuwara eliya", "kandy", "kurunegala"]
    };

    const btnToggleRouting = document.getElementById('btn-toggle-routing');
    const btnClearRoute = document.getElementById('btn-clear-route');
    const routingInfoCard = document.getElementById('routing-info-card');
    const routingInstruction = document.getElementById('routing-instruction');
    const routingDetails = document.getElementById('routing-details');

    let routingMode = false;
    let routingPoints = [];
    let routePolyline = null;
    let routingMarkers = [];

    function findShortestDistrictPath(startDistrict, endDistrict) {
        const start = startDistrict.toLowerCase().trim();
        const end = endDistrict.toLowerCase().trim();
        
        if (start === end) return [start];
        if (!DISTRICT_GRAPH[start] || !DISTRICT_GRAPH[end]) return null;
        
        // Find Red evac warning districts
        const redDistricts = new Set(
            activeDmcAlerts
                .filter(a => a.level === 'Red')
                .map(a => a.district.toLowerCase().trim())
        );

        function runBFS(avoidRed) {
            const queue = [[start]];
            const visited = new Set([start]);
            
            while (queue.length > 0) {
                const path = queue.shift();
                const node = path[path.length - 1];
                
                const neighbors = DISTRICT_GRAPH[node] || [];
                for (const neighbor of neighbors) {
                    if (avoidRed && redDistricts.has(neighbor) && neighbor !== end && neighbor !== start) {
                        continue;
                    }
                    if (!visited.has(neighbor)) {
                        visited.add(neighbor);
                        const newPath = [...path, neighbor];
                        if (neighbor === end) {
                            return newPath;
                        }
                        queue.push(newPath);
                    }
                }
            }
            return null;
        }

        let path = runBFS(true);
        let usedDetour = true;
        if (!path) {
            path = runBFS(false);
            usedDetour = false;
        }
        
        if (path) {
            path.isDetour = usedDetour;
            path.fallback = !usedDetour && redDistricts.size > 0;
        }
        return path;
    }

    function plotAllRoutingMarkers() {
        // Remove standard map markers
        mapMarkers.forEach(m => playgroundMap.removeLayer(m));
        mapMarkers = [];
        
        // Remove previous routing markers
        routingMarkers.forEach(m => playgroundMap.removeLayer(m));
        routingMarkers = [];

        if (routePolyline) {
            playgroundMap.removeLayer(routePolyline);
            routePolyline = null;
        }

        cachedPlaces.forEach(place => {
            const lat = parseFloat(place.lat);
            const lng = parseFloat(place.lng);
            if (isNaN(lat) || isNaN(lng)) return;

            let markerColor = '#10b981'; // Green
            if (place.safety_level === 'Moderate') markerColor = '#f59e0b';
            if (place.safety_level === 'Dangerous') markerColor = '#ef4444';

            const customIcon = L.divIcon({
                className: 'custom-map-pin routing-pin',
                html: `<div class="pin-pulse" style="background-color: ${markerColor}; width: 12px; height: 12px; border-radius: 50%; box-shadow: 0 0 8px ${markerColor}; border: 1.5px solid white;"></div>`,
                iconSize: [12, 12],
                iconAnchor: [6, 6]
            });

            const marker = L.marker([lat, lng], { icon: customIcon })
                .addTo(playgroundMap);
                
            marker.on('click', () => {
                handleRoutingPointSelect(place);
            });
            
            marker.bindTooltip(`<strong>${place.name}</strong> (${place.district_id})<br><span style="font-size:10px; color:#a855f7;">Click to select as waypoint</span>`, {
                sticky: true
            });
            
            routingMarkers.push(marker);
        });

        // Fit bounds of all database places
        if (cachedPlaces.length > 0) {
            const bounds = cachedPlaces.map(p => [parseFloat(p.lat), parseFloat(p.lng)]).filter(coord => !isNaN(coord[0]));
            if (bounds.length > 0) {
                playgroundMap.fitBounds(bounds, { padding: [30, 30] });
            }
        }
    }

    function handleRoutingPointSelect(place) {
        if (routingPoints.length >= 2) {
            routingPoints = [];
            if (routePolyline) {
                playgroundMap.removeLayer(routePolyline);
                routePolyline = null;
            }
        }
        
        routingPoints.push(place);
        
        if (routingPoints.length === 1) {
            if (routingInstruction) {
                routingInstruction.innerHTML = `Start: <span style="color: #c084fc;">${place.name} (${place.district_id})</span>. Click destination...`;
            }
            if (routingDetails) {
                routingDetails.innerText = "Select destination waypoint marker on the map.";
            }
        } else if (routingPoints.length === 2) {
            const startPlace = routingPoints[0];
            const endPlace = routingPoints[1];
            
            if (routingInstruction) {
                routingInstruction.innerHTML = `Path: <span style="color: #c084fc;">${startPlace.name}</span> ➜ <span style="color: #34d399;">${endPlace.name}</span>`;
            }
            
            calculateAndDrawRoute(startPlace, endPlace);
        }
    }

    function calculateAndDrawRoute(startPlace, endPlace) {
        const startDist = startPlace.district_id.toLowerCase().trim();
        const endDist = endPlace.district_id.toLowerCase().trim();
        
        const path = findShortestDistrictPath(startDist, endDist);
        if (!path) {
            if (routingDetails) {
                routingDetails.innerText = `Could not compute routing path between ${startDist.toUpperCase()} and ${endDist.toUpperCase()}. No adjacent path found.`;
            }
            return;
        }

        const routeCoordinates = [];
        routeCoordinates.push([parseFloat(startPlace.lat), parseFloat(startPlace.lng)]);
        
        for (let i = 1; i < path.length - 1; i++) {
            const distName = path[i];
            const coords = DISTRICT_COORDINATES[distName];
            if (coords) {
                routeCoordinates.push(coords);
            }
        }
        
        routeCoordinates.push([parseFloat(endPlace.lat), parseFloat(endPlace.lng)]);
        
        let routeSafety = 'Safe';
        let cautions = [];
        
        // Spot-level safety checks from local DB
        path.forEach(dist => {
            const spots = cachedPlaces.filter(p => (p.district_id || "").toLowerCase().trim() === dist);
            spots.forEach(s => {
                if (s.safety_level === 'Dangerous') {
                    routeSafety = 'Dangerous';
                    cautions.push(`⚠️ Dangerous spot in ${dist.toUpperCase()}: ${s.name} (${s.wildlife_hazard || 'Wildlife caution'})`);
                } else if (s.safety_level === 'Moderate' && routeSafety !== 'Dangerous') {
                    routeSafety = 'Moderate';
                    cautions.push(`⚠️ Caution in ${dist.toUpperCase()}: ${s.name} (${s.rain_sensitivity || 'Rain caution'})`);
                }
            });
        });

        // DMC warning hazard alerts checks
        let hasDmcRed = false;
        path.forEach(dist => {
            const dmcAlert = activeDmcAlerts.find(a => a.district.toLowerCase().trim() === dist);
            if (dmcAlert) {
                if (dmcAlert.level === 'Red') {
                    hasDmcRed = true;
                    routeSafety = 'Dangerous';
                    cautions.push(`🚨 Evacuate immediately in ${dist.toUpperCase()}: ${dmcAlert.hazard} (${dmcAlert.bulletin})`);
                } else if (dmcAlert.level === 'Amber' && routeSafety !== 'Dangerous') {
                    routeSafety = 'Moderate';
                    cautions.push(`⚠️ DMC Warning in ${dist.toUpperCase()}: ${dmcAlert.hazard} (${dmcAlert.bulletin})`);
                }
            }
        });
        
        let lineColor = '#10b981'; // Green
        if (routeSafety === 'Moderate') lineColor = '#f59e0b'; // Gold
        if (routeSafety === 'Dangerous' || hasDmcRed) lineColor = '#ef4444'; // Red
        
        if (routePolyline) {
            playgroundMap.removeLayer(routePolyline);
        }

        routePolyline = L.polyline(routeCoordinates, {
            color: lineColor,
            weight: 5,
            opacity: 0.8,
            dashArray: '8, 8'
        }).addTo(playgroundMap);
        
        playgroundMap.fitBounds(routePolyline.getBounds(), { padding: [50, 50] });
        
        if (routingDetails) {
            let html = `<strong>Shortest District Path:</strong> ${path.map(d => d.toUpperCase()).join(' ➜ ')}<br><br>`;
            
            if (path.fallback) {
                html += `<div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid #ef4444; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px; color: #ef4444; font-weight: 600; font-size: 11.5px; display: flex; flex-direction: column; gap: 3px;">
                    <div>🚨 CRITICAL FALLBACK WARNING:</div>
                    <div style="font-weight: normal; color: var(--text-secondary);">No alternative detour routes exist! The path must cross active Red evac warning areas (e.g. Nuwara Eliya). Proceed with extreme safety caution.</div>
                </div>`;
            } else if (path.isDetour && activeDmcAlerts.some(a => a.level === 'Red')) {
                html += `<div style="background: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px; color: #34d399; font-weight: 600; font-size: 11.5px; display: flex; flex-direction: column; gap: 3px;">
                    <div>🛡️ LANDSLIDE DETOUR ACTIVATED:</div>
                    <div style="font-weight: normal; color: var(--text-secondary);">This route was automatically adjusted to avoid active Red disaster zones (e.g. Nuwara Eliya). Safe detour calculated.</div>
                </div>`;
            }
            
            if (cautions.length > 0) {
                html += `<div style="color: #fbbf24; display:flex; flex-direction:column; gap:4px; max-height: 80px; overflow-y: auto; padding-right: 5px;">${cautions.map(c => `<div>${c}</div>`).join('')}</div>`;
            } else {
                html += `<span style="color: #34d399; font-weight:600;">🟢 Safe Route! No weather alerts or hazard warnings on this path. Enjoy your journey!</span>`;
            }
            routingDetails.innerHTML = html;
        }
    }

    function clearRoutingState() {
        routingPoints = [];
        if (routePolyline) {
            playgroundMap.removeLayer(routePolyline);
            routePolyline = null;
        }
        if (routingInstruction) {
            routingInstruction.innerText = "Select two markers on the map to compute a route.";
        }
        if (routingDetails) {
            routingDetails.innerText = "Select a start and destination waypoint marker on the map.";
        }
    }

    if (btnToggleRouting) {
        btnToggleRouting.addEventListener('click', () => {
            routingMode = !routingMode;
            if (routingMode) {
                // Turn OFF heatmap if active
                if (heatmapMode && btnToggleHeatmap) {
                    btnToggleHeatmap.click();
                }
                
                btnToggleRouting.classList.add('active');
                btnToggleRouting.innerHTML = `<i data-lucide="navigation" style="width: 12px; height: 12px;"></i> <span>Exit Route Mode</span>`;
                if (routingInfoCard) routingInfoCard.classList.remove('hidden');
                
                plotAllRoutingMarkers();
            } else {
                btnToggleRouting.classList.remove('active');
                btnToggleRouting.innerHTML = `<i data-lucide="navigation" style="width: 12px; height: 12px;"></i> <span>Smart Route Mode</span>`;
                if (routingInfoCard) routingInfoCard.classList.add('hidden');
                
                clearRoutingState();
                // Clear routing markers
                routingMarkers.forEach(m => playgroundMap.removeLayer(m));
                routingMarkers = [];
                
                // Restore standard empty state
                if (playgroundPrompt && playgroundPrompt.value) {
                    const bubbles = playgroundResponse.querySelectorAll('.chat-bubble');
                    if (bubbles.length > 0) {
                        const lastBubble = bubbles[bubbles.length - 1];
                        parseAndPlotLocations(lastBubble.innerText || lastBubble.textContent);
                    }
                }
            }
            lucide.createIcons();
        });
    }

    if (btnClearRoute) {
        btnClearRoute.addEventListener('click', (e) => {
            e.stopPropagation();
            clearRoutingState();
        });
    }

    // --- 14L. Multi-Turn Interactive Chat-to-SFT Dataset Exporter ---
    const btnAddChatToSft = document.getElementById('btn-add-chat-to-sft');
    if (btnAddChatToSft) {
        btnAddChatToSft.addEventListener('click', async () => {
            const bubbles = playgroundResponse.querySelectorAll('.chat-bubble');
            if (bubbles.length === 0 || (bubbles.length === 1 && bubbles[0].classList.contains('empty-state'))) {
                alert("No chat conversation to save to SFT dataset.");
                return;
            }
            
            const messages = [];
            const customSystemPrompt = systemPromptPreview ? systemPromptPreview.value.trim() : '';
            if (customSystemPrompt) {
                messages.push({ role: 'system', content: customSystemPrompt });
            } else {
                messages.push({ role: 'system', content: "You are Lumen-1, an advanced AI travel assistant for Sri Lanka." });
            }
            
            bubbles.forEach(b => {
                if (b.classList.contains('empty-state')) return;
                const isUser = b.classList.contains('user');
                const role = isUser ? 'user' : 'assistant';
                
                let content = '';
                const divs = b.querySelectorAll('div');
                const textParts = [];
                divs.forEach(d => {
                    if (!d.classList.contains('chat-bubble-mode-tag') && 
                        !d.classList.contains('chat-bubble-meta') && 
                        !d.classList.contains('chat-bubble-media') &&
                        !d.classList.contains('spinner') &&
                        d.style.display !== 'none') {
                        if (d.style.whiteSpace === 'pre-wrap' || d.textContent.trim().length > 0) {
                            textParts.push(d.innerText || d.textContent);
                        }
                    }
                });
                content = textParts.join('\n').trim();
                if (!content) content = b.innerText;
                
                messages.push({ role, content });
            });

            if (messages.length <= 1) {
                alert("Conversation must contain user and assistant messages.");
                return;
            }

            if (!confirm(`Are you sure you want to add this conversation (${messages.length} messages including system prompt) directly to the SFT training dataset?`)) {
                return;
            }

            try {
                btnAddChatToSft.disabled = true;
                const res = await fetch('/api/dataset/append-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages })
                });
                const data = await res.json();
                if (res.status === 200) {
                    alert("Success! Chat successfully appended to 'data/sft.jsonl'!");
                    loadDatasetInfo();
                } else {
                    alert(`Error appending chat: ${data.detail || 'Save failed'}`);
                }
            } catch (e) {
                alert(`Network error: ${e.message}`);
            } finally {
                btnAddChatToSft.disabled = false;
            }
        });
    }

    // --- 14M. DPO Length Bias Diagnostics Chart ---
    let dpoBiasChartInstance = null;
    
    async function loadDpoDiagnostics() {
        const diagTotal = document.getElementById('dpo-diag-total');
        const diagChosen = document.getElementById('dpo-diag-chosen');
        const diagRejected = document.getElementById('dpo-diag-rejected');
        const diagRatioBadge = document.getElementById('dpo-diagnostics-ratio-badge');
        const verbosityAlert = document.getElementById('dpo-verbosity-alert');
        
        try {
            const res = await fetch('/api/dpo/diagnostics');
            const data = await res.json();
            
            if (res.status === 200 && data.status === 'success') {
                if (diagTotal) diagTotal.innerText = data.total_pairs;
                if (diagChosen) diagChosen.innerText = `${data.avg_chosen_words} words`;
                if (diagRejected) diagRejected.innerText = `${data.avg_rejected_words} words`;
                if (diagRatioBadge) {
                    diagRatioBadge.innerText = `${data.length_ratio}x Length Ratio`;
                    if (data.verbosity_warning) {
                        diagRatioBadge.className = "badge badge-red";
                    } else {
                        diagRatioBadge.className = "badge badge-purple";
                    }
                }
                
                if (verbosityAlert) {
                    if (data.verbosity_warning) {
                        verbosityAlert.classList.remove('hidden');
                    } else {
                        verbosityAlert.classList.add('hidden');
                    }
                }
                
                drawDpoBiasChart(data.avg_chosen_words, data.avg_rejected_words);
            }
        } catch (e) {
            console.error("Error loading DPO diagnostics:", e);
        }
    }

    function drawDpoBiasChart(chosenAvg, rejectedAvg) {
        const canvasEl = document.getElementById('dpo-bias-chart');
        if (!canvasEl) return;
        
        const ctx = canvasEl.getContext('2d');
        if (dpoBiasChartInstance) {
            dpoBiasChartInstance.destroy();
        }
        
        dpoBiasChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Chosen (Preferred)', 'Rejected'],
                datasets: [{
                    data: [chosenAvg, rejectedAvg],
                    backgroundColor: ['rgba(52, 211, 153, 0.45)', 'rgba(248, 113, 113, 0.45)'],
                    borderColor: ['#34d399', '#f87171'],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#8e9bb3', font: { family: 'Outfit', size: 11 } },
                        title: { display: true, text: 'Average Word Count', color: '#8e9bb3', font: { size: 10 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f1f3f9', font: { family: 'Outfit', size: 11, weight: '600' } }
                    }
                }
            }
        });
    }

    // --- 14N. Model Arena Controller ---
    const btnSendArena = document.getElementById('btn-send-arena');
    const arenaPromptInput = document.getElementById('arena-prompt-input');
    const arenaAResponse = document.getElementById('arena-a-response');
    const arenaBResponse = document.getElementById('arena-b-response');
    const btnChooseModelA = document.getElementById('btn-choose-model-a');
    const btnChooseModelB = document.getElementById('btn-choose-model-b');
    const arenaALatency = document.getElementById('arena-model-a-latency');
    const arenaBLatency = document.getElementById('arena-model-b-latency');

    let arenaLastPrompt = '';
    let arenaLastResponseA = '';
    let arenaLastResponseB = '';

    if (btnSendArena) {
        btnSendArena.addEventListener('click', async () => {
            const prompt = arenaPromptInput.value.trim();
            if (!prompt) return;

            btnSendArena.disabled = true;
            btnChooseModelA.disabled = true;
            btnChooseModelB.disabled = true;
            arenaLastPrompt = prompt;

            arenaAResponse.innerHTML = '<div class="chat-bubble assistant"><div class="spinner"></div> Generating Response A...</div>';
            arenaBResponse.innerHTML = '<div class="chat-bubble assistant"><div class="spinner"></div> Generating Response B...</div>';
            arenaALatency.innerText = "Generating...";
            arenaBLatency.innerText = "Generating...";

            const modeA = document.getElementById('arena-a-mode').value;
            const tempA = parseFloat(document.getElementById('arena-a-temp').value);
            const sysA = document.getElementById('arena-a-system').value.trim();

            const modeB = document.getElementById('arena-b-mode').value;
            const tempB = parseFloat(document.getElementById('arena-b-temp').value);
            const sysB = document.getElementById('arena-b-system').value.trim();

            const startA = performance.now();
            const promiseA = fetch('/api/test-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt, use_rag: true, mode: modeA, system_prompt: sysA, temperature: tempA })
            }).then(r => r.json()).then(data => {
                const elapsed = Math.round(performance.now() - startA);
                arenaALatency.innerText = `${elapsed}ms`;
                if (data.status === 'success') {
                    arenaLastResponseA = data.response;
                    arenaAResponse.innerHTML = `<div class="chat-bubble assistant" style="white-space: pre-wrap;">${data.response}</div>`;
                } else {
                    arenaAResponse.innerHTML = `<div class="chat-bubble assistant text-red">Error: ${data.detail || 'Inference failed'}</div>`;
                }
            }).catch(err => {
                arenaAResponse.innerHTML = `<div class="chat-bubble assistant text-red">Network Error</div>`;
                arenaALatency.innerText = "Error";
            });

            const startB = performance.now();
            const promiseB = fetch('/api/test-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt, use_rag: true, mode: modeB, system_prompt: sysB, temperature: tempB })
            }).then(r => r.json()).then(data => {
                const elapsed = Math.round(performance.now() - startB);
                arenaBLatency.innerText = `${elapsed}ms`;
                if (data.status === 'success') {
                    arenaLastResponseB = data.response;
                    arenaBResponse.innerHTML = `<div class="chat-bubble assistant" style="white-space: pre-wrap;">${data.response}</div>`;
                } else {
                    arenaBResponse.innerHTML = `<div class="chat-bubble assistant text-red">Error: ${data.detail || 'Inference failed'}</div>`;
                }
            }).catch(err => {
                arenaBResponse.innerHTML = `<div class="chat-bubble assistant text-red">Network Error</div>`;
                arenaBLatency.innerText = "Error";
            });

            await Promise.all([promiseA, promiseB]);
            btnSendArena.disabled = false;
            
            if (arenaLastResponseA && arenaLastResponseB) {
                btnChooseModelA.disabled = false;
                btnChooseModelB.disabled = false;
            }
        });
    }

    if (btnChooseModelA) {
        btnChooseModelA.addEventListener('click', async () => {
            await submitArenaVote(arenaLastPrompt, arenaLastResponseA, arenaLastResponseB);
            btnChooseModelA.disabled = true;
            btnChooseModelB.disabled = true;
        });
    }

    if (btnChooseModelB) {
        btnChooseModelB.addEventListener('click', async () => {
            await submitArenaVote(arenaLastPrompt, arenaLastResponseB, arenaLastResponseA);
            btnChooseModelA.disabled = true;
            btnChooseModelB.disabled = true;
        });
    }

    async function submitArenaVote(prompt, chosen, rejected) {
        try {
            const res = await fetch('/api/dpo/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, chosen, rejected })
            });
            const data = await res.json();
            if (res.status === 200 && data.status === 'success') {
                alert("Preferences submitted directly to DPO Dataset Compiler! 🚀");
                if (typeof loadCompilerDpoList === 'function') loadCompilerDpoList();
                if (typeof loadDpoDiagnostics === 'function') loadDpoDiagnostics();
            } else {
                alert(`Error saving preferences: ${data.detail || 'Unknown error'}`);
            }
        } catch (e) {
            console.error(e);
            alert("Connection error when submitting preferences.");
        }
    }

    // --- 14O. Voice Tour Guide Controller ---
    const voiceTourOverlay = document.getElementById('voice-tour-overlay');
    const voiceTourTitle = document.getElementById('voice-tour-title');
    const btnVoicePlaypause = document.getElementById('btn-voice-playpause');
    const btnVoiceStop = document.getElementById('btn-voice-stop');
    const voiceTourRate = document.getElementById('voice-tour-rate');
    const voiceTourRateVal = document.getElementById('voice-tour-rate-val');
    const voiceWaveVisualizer = document.getElementById('voice-wave-visualizer');

    let currentUtterance = null;
    let synthPlaying = false;
    let synthPaused = false;

    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.btn-voice-tour');
        if (btn) {
            const placeId = btn.getAttribute('data-place-id');
            const place = cachedPlaces.find(p => p.id === placeId);
            if (place) {
                startVoiceTour(place);
            }
        }
    });

    function startVoiceTour(place) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();

            const textToSpeak = `Welcome to ${place.name}. Located in the ${place.district_id} district of the ${place.province_id} province. This is a ${place.tourist_popularity} popularity tourist destination. Road conditions are ${place.road_condition} and mobile signal is ${place.mobile_signal}. ${place.description} Activities you can do here include ${place.activities}. The safety level of this spot is classified as ${place.safety_level}. Weather note: ${place.rain_sensitivity}. Monsoon note: ${place.monsoon_note}.`;
            
            currentUtterance = new SpeechSynthesisUtterance(textToSpeak);
            
            const voices = window.speechSynthesis.getVoices();
            const ukVoice = voices.find(v => v.lang.includes('GB') || v.lang.includes('EN'));
            if (ukVoice) currentUtterance.voice = ukVoice;

            currentUtterance.rate = parseFloat(voiceTourRate ? voiceTourRate.value : 1.0);
            
            currentUtterance.onstart = () => {
                synthPlaying = true;
                synthPaused = false;
                if (voiceTourOverlay) voiceTourOverlay.classList.remove('hidden');
                if (voiceWaveVisualizer) voiceWaveVisualizer.classList.add('playing');
                updateVoiceControls();
            };

            currentUtterance.onend = () => {
                stopVoiceTour();
            };

            currentUtterance.onerror = (event) => {
                console.error("SpeechSynthesis error", event);
                stopVoiceTour();
            };

            if (voiceTourTitle) voiceTourTitle.innerText = place.name;
            window.speechSynthesis.speak(currentUtterance);
        } else {
            alert("Text-to-Speech is not supported in this browser.");
        }
    }

    function stopVoiceTour() {
        window.speechSynthesis.cancel();
        synthPlaying = false;
        synthPaused = false;
        if (voiceTourOverlay) voiceTourOverlay.classList.add('hidden');
        if (voiceWaveVisualizer) voiceWaveVisualizer.classList.remove('playing');
        updateVoiceControls();
    }

    function updateVoiceControls() {
        if (!btnVoicePlaypause) return;
        if (synthPaused) {
            btnVoicePlaypause.innerHTML = '<i data-lucide="play" style="width:14px; height:14px;"></i>';
        } else {
            btnVoicePlaypause.innerHTML = '<i data-lucide="pause" style="width:14px; height:14px;"></i>';
        }
        lucide.createIcons();
    }

    if (btnVoicePlaypause) {
        btnVoicePlaypause.addEventListener('click', () => {
            if (synthPlaying) {
                if (synthPaused) {
                    window.speechSynthesis.resume();
                    synthPaused = false;
                    if (voiceWaveVisualizer) voiceWaveVisualizer.classList.add('playing');
                } else {
                    window.speechSynthesis.pause();
                    synthPaused = true;
                    if (voiceWaveVisualizer) voiceWaveVisualizer.classList.remove('playing');
                }
                updateVoiceControls();
            }
        });
    }

    if (btnVoiceStop) {
        btnVoiceStop.addEventListener('click', () => {
            stopVoiceTour();
        });
    }

    if (voiceTourRate) {
        voiceTourRate.addEventListener('input', () => {
            const val = parseFloat(voiceTourRate.value);
            if (voiceTourRateVal) voiceTourRateVal.innerText = `${val.toFixed(1)}x`;
        });
    }

    // --- 14P. Synthetic Dataset Generator Controller ---
    const btnGenerateSynthetic = document.getElementById('btn-generate-synthetic');
    if (btnGenerateSynthetic) {
        btnGenerateSynthetic.addEventListener('click', async () => {
            const category = document.getElementById('synthetic-category').value;
            const datasetType = document.getElementById('synthetic-dataset-type').value;
            const numSamples = parseInt(document.getElementById('synthetic-num-samples').value);

            btnGenerateSynthetic.disabled = true;
            const originalHTML = btnGenerateSynthetic.innerHTML;
            btnGenerateSynthetic.innerHTML = `<span class="spinner"></span> Synthesizing...`;

            try {
                const res = await fetch('/api/dataset/synthetic-generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ category, dataset_type: datasetType, num_samples: numSamples })
                });
                const data = await res.json();
                
                if (res.status === 200 && data.status === 'success') {
                    alert(`Successfully synthesized and appended ${numSamples} ${datasetType.toUpperCase()} samples! 🎉`);
                    if (typeof loadCompilerDpoList === 'function') loadCompilerDpoList();
                    if (typeof loadDpoDiagnostics === 'function') loadDpoDiagnostics();
                    if (typeof loadDatasetInfo === 'function') loadDatasetInfo();
                } else {
                    alert(`Synthesis failed: ${data.detail || 'Unknown error'}`);
                }
            } catch (err) {
                console.error(err);
                alert("Error during synthetic generation query.");
            } finally {
                btnGenerateSynthetic.disabled = false;
                btnGenerateSynthetic.innerHTML = originalHTML;
                lucide.createIcons();
            }
        });
    }

    // --- 14Q. Speculative Decoding Diagnostics Charts ---
    let specHistogramChartInstance = null;
    let specLatencyChartInstance = null;

    async function loadSpeculativeDiagnostics() {
        try {
            const res = await fetch('/api/system/speculative-diagnostics');
            const data = await res.json();
            if (res.status === 200 && data.status === 'success') {
                const totalRunsEl = document.getElementById('spec-diag-total-runs');
                const avgSpeedupEl = document.getElementById('spec-diag-avg-speedup');
                const avgAcceptanceEl = document.getElementById('spec-diag-avg-acceptance');

                if (totalRunsEl) totalRunsEl.innerText = `${data.total_runs} runs`;
                if (avgSpeedupEl) avgSpeedupEl.innerText = `${data.avg_speedup}x Speedup`;
                if (avgAcceptanceEl) avgAcceptanceEl.innerText = `${data.avg_acceptance_rate}% Acceptance`;

                drawSpeculativeHistogram(data.global_histogram);
                drawSpeculativeLatency(data.history);
            }
        } catch (e) {
            console.error("Error loading speculative diagnostics:", e);
        }
    }

    function drawSpeculativeHistogram(globalHistogram) {
        const canvas = document.getElementById('speculative-histogram-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (specHistogramChartInstance) {
            specHistogramChartInstance.destroy();
        }

        specHistogramChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['0 Tok', '1 Tok', '2 Tok', '3 Tok', '4 Tok', '5 Tok', '6 Tok'],
                datasets: [{
                    label: 'Accepted Steps Count',
                    data: globalHistogram,
                    backgroundColor: 'rgba(139, 92, 246, 0.45)',
                    borderColor: '#8b5cf6',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8e9bb3', font: { family: 'Outfit', size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#8e9bb3', font: { family: 'Outfit', size: 10 } }
                    }
                }
            }
        });
    }

    function drawSpeculativeLatency(history) {
        const canvas = document.getElementById('speculative-latency-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (specLatencyChartInstance) {
            specLatencyChartInstance.destroy();
        }

        const recentHistory = history.slice(-10);
        const labels = recentHistory.map((_, i) => `Run ${i + 1}`);
        const specTimes = recentHistory.map(r => r.spec_time * 1000);
        const stdTimes = recentHistory.map(r => r.std_time * 1000);

        specLatencyChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Speculative Time',
                        data: specTimes,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.05)',
                        borderWidth: 2,
                        tension: 0.35,
                        fill: true
                    },
                    {
                        label: 'Standard Time',
                        data: stdTimes,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.05)',
                        borderWidth: 2,
                        tension: 0.35,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#8e9bb3', font: { family: 'Outfit', size: 10 } }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8e9bb3', font: { family: 'Outfit', size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#8e9bb3', font: { family: 'Outfit', size: 10 } },
                        title: { display: true, text: 'Time (ms)', color: '#8e9bb3', font: { size: 9 } }
                    }
                }
            }
        });
    }

    // --- Phase 11: Evaluation Scorer, Custom Tuner, Multimodal Sandbox & Live Weather Alerts ---
    
    // Global active DMC alerts state
    let activeDmcAlerts = [];
    let alertMapCircles = [];
    let evaluationRadarChart = null;

    // Load evaluation datasets from the backend
    async function loadEvaluationDatasets() {
        try {
            const res = await fetch('/api/evaluation/datasets');
            const data = await res.json();
            if (data.status === 'success') {
                const select = document.getElementById('eval-dataset-select');
                if (select) {
                    select.innerHTML = '';
                    if (data.datasets.length === 0) {
                        select.innerHTML = '<option value="">No datasets available</option>';
                        return;
                    }
                    data.datasets.forEach(ds => {
                        select.innerHTML += `<option value="${ds.filename}">${ds.filename} (${ds.count} records)</option>`;
                    });
                }
            }
        } catch (e) {
            console.error("Error loading evaluation datasets:", e);
        }
    }

    // Run evaluation scorer
    async function runEvaluationScorer() {
        const datasetSelect = document.getElementById('eval-dataset-select');
        const limitInput = document.getElementById('eval-limit-input');
        const runBtn = document.getElementById('btn-run-evaluation');
        const progressContainer = document.getElementById('eval-progress-container');
        const progressStatus = document.getElementById('eval-progress-status');
        const progressBar = document.getElementById('eval-progress-bar');
        const progressPercent = document.getElementById('eval-progress-percent');

        if (!datasetSelect || !datasetSelect.value) {
            alert("Please select a benchmark dataset first.");
            return;
        }

        const dataset = datasetSelect.value;
        const limit = parseInt(limitInput.value) || 5;

        // Reset UI
        runBtn.disabled = true;
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressPercent.innerText = '0%';
        progressStatus.innerText = `Preparing evaluation run for ${dataset}...`;

        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += 5;
                progressBar.style.width = `${progress}%`;
                progressPercent.innerText = `${progress}%`;
                progressStatus.innerText = `Evaluating checkpoint model on ${dataset}... (${progress}%)`;
            }
        }, 300);

        try {
            const res = await fetch('/api/evaluation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, limit, temperature: 0.7 })
            });
            const data = await res.json();
            
            clearInterval(progressInterval);

            if (res.status === 200 && data.status === 'success') {
                progressBar.style.width = '100%';
                progressPercent.innerText = '100%';
                progressStatus.innerText = `Evaluation complete! Loaded ${data.summary.count} test pairs.`;

                // Update score cards
                document.getElementById('eval-avg-bleu').innerText = data.summary.avg_bleu.toFixed(3);
                document.getElementById('eval-avg-rouge').innerText = data.summary.avg_rouge.toFixed(3);
                document.getElementById('eval-avg-similarity').innerText = data.summary.avg_similarity.toFixed(3);

                // Update detailed comparisons table
                const tableBody = document.getElementById('eval-results-table-body');
                if (tableBody) {
                    tableBody.innerHTML = '';
                    data.scores.forEach(score => {
                        const tr = document.createElement('tr');
                        tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                        
                        tr.innerHTML = `
                            <td style="padding: 10px; font-weight: 500;">${score.question}</td>
                            <td style="padding: 10px; color: var(--text-secondary);">${score.reference}</td>
                            <td style="padding: 10px; color: #a78bfa;">${score.generated}</td>
                            <td style="padding: 10px; text-align: center; color: #a855f7; font-weight: 600;">${score.bleu.toFixed(3)}</td>
                            <td style="padding: 10px; text-align: center; color: #ec4899; font-weight: 600;">${score.rouge.toFixed(3)}</td>
                            <td style="padding: 10px; text-align: center; color: #10b981; font-weight: 600;">${score.similarity.toFixed(3)}</td>
                        `;
                        tableBody.appendChild(tr);
                    });
                }

                // Render radar metrics chart
                drawEvaluationChart(data.summary.avg_bleu, data.summary.avg_rouge, data.summary.avg_similarity);
                
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                }, 3000);
            } else {
                alert(`Evaluation run failed: ${data.detail || 'Unknown error'}`);
                progressContainer.classList.add('hidden');
            }
        } catch (e) {
            clearInterval(progressInterval);
            alert(`Error during evaluation run: ${e.message}`);
            progressContainer.classList.add('hidden');
        } finally {
            runBtn.disabled = false;
        }
    }

    function drawEvaluationChart(bleu, rouge, sim) {
        const ctx = document.getElementById('evaluation-radar-chart').getContext('2d');
        if (evaluationRadarChart) {
            evaluationRadarChart.destroy();
        }
        evaluationRadarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['BLEU Overlap', 'ROUGE-L Overlap', 'Embedding Cosine Similarity'],
                datasets: [{
                    label: 'Pipeline Checkpoint Metrics Profile',
                    data: [bleu, rouge, sim],
                    backgroundColor: 'rgba(139, 92, 246, 0.15)',
                    borderColor: 'rgba(139, 92, 246, 0.85)',
                    borderWidth: 2,
                    pointBackgroundColor: '#ec4899',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 1.5,
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(139, 92, 246, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#a3a6b8', font: { family: 'Outfit', size: 10 } }
                    }
                },
                scales: {
                    r: {
                        min: 0,
                        max: 1.0,
                        ticks: { stepSize: 0.2, color: 'rgba(255,255,255,0.4)', showLabelBackdrop: false },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                        angleLines: { color: 'rgba(255,255,255,0.06)' },
                        pointLabels: { color: '#a3a6b8', font: { family: 'Outfit', size: 9, weight: '600' } }
                    }
                }
            }
        });
    }

    // Sync Live DMC Alerts
    async function syncDmcAlerts() {
        try {
            const btn = document.getElementById('btn-sync-dmc');
            if (btn) btn.disabled = true;
            
            const res = await fetch('/api/database/alerts-sync');
            const data = await res.json();
            
            if (btn) btn.disabled = false;
            
            if (data.status === 'success') {
                activeDmcAlerts = data.alerts;
                plotAlertCircles();
                
                // Re-calculate route if waypoints are selected
                if (routingPoints.length === 2) {
                    calculateAndDrawRoute(routingPoints[0], routingPoints[1]);
                }
                
                alert(`📡 DMC Alerts Synced! Synchronized ${activeDmcAlerts.length} active landslide & rain bulletins.`);
            } else {
                alert("DMC warnings sync returned failure state.");
            }
        } catch (e) {
            console.error("DMC warning alerts sync failed:", e);
            alert(`Warnings Sync Failed: ${e.message}`);
        }
    }

    // Plot hazard overlays on the Leaflet map
    function plotAlertCircles() {
        // Clear previous circles
        alertMapCircles.forEach(c => playgroundMap.removeLayer(c));
        alertMapCircles = [];

        activeDmcAlerts.forEach(alert => {
            const color = alert.level === 'Red' ? '#ef4444' : (alert.level === 'Amber' ? '#f59e0b' : '#fbbf24');
            const circle = L.circle([alert.lat, alert.lng], {
                color: color,
                fillColor: color,
                fillOpacity: 0.25,
                weight: 2,
                radius: alert.radius
            }).addTo(playgroundMap);
            
            circle.bindTooltip(`<strong>🚨 DMC Hazard Alert (${alert.level})</strong><br>District: ${alert.district}<br>Type: ${alert.hazard}<br>${alert.bulletin}`);
            alertMapCircles.push(circle);
        });
    }

    // Bind Phase 11 click listeners
    const evalRunBtn = document.getElementById('btn-run-evaluation');
    if (evalRunBtn) {
        evalRunBtn.addEventListener('click', runEvaluationScorer);
    }

    const syncDmcBtn = document.getElementById('btn-sync-dmc');
    if (syncDmcBtn) {
        syncDmcBtn.addEventListener('click', syncDmcAlerts);
    }

    // --- GENERIC SUB-TAB NAVIGATION HANDLER ---
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.sub-tab-btn');
        if (btn) {
            const subTabId = btn.getAttribute('data-sub-tab');
            const parentPane = btn.closest('.tab-pane');
            if (parentPane) {
                parentPane.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                parentPane.querySelectorAll('.sub-tab-content').forEach(content => content.classList.remove('active'));
                const targetContent = parentPane.querySelector(`#sub-tab-${subTabId}`);
                if (targetContent) targetContent.classList.add('active');
                
                // Trigger sub-tab specific loaders
                switch (subTabId) {
                    case 'training-exporter':
                        checkExporterStatus();
                        break;
                    case 'training-evaluation':
                        loadEvaluationDatasets();
                        break;
                    case 'data-database':
                        loadDatabaseRecords();
                        loadDatabaseStats();
                        break;
                    case 'data-rag':
                        loadRagDocuments();
                        break;
                    case 'data-sft-dataset':
                        loadCompilerSFTList();
                        break;
                    case 'data-dpo-dataset':
                        loadCompilerDpoList();
                        loadDpoDiagnostics();
                        break;
                    case 'security-config':
                        loadSecurityConfig();
                        break;
                    case 'security-audit-log':
                        loadSecurityLogs();
                        break;
                }
            }
        }
    });

    // Collapsible Sidebar Handler
    const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    const sidebar = document.querySelector('.sidebar');
    const sidebarToggleIcon = document.getElementById('sidebar-toggle-icon');

    if (btnToggleSidebar && sidebar && sidebarToggleIcon) {
        // Load choice from localStorage
        const isCollapsed = localStorage.getItem('lumen_sidebar_collapsed') === 'true';
        if (isCollapsed) {
            sidebar.classList.add('collapsed');
            sidebarToggleIcon.setAttribute('data-lucide', 'chevron-right');
            lucide.createIcons();
        }

        btnToggleSidebar.addEventListener('click', (e) => {
            e.stopPropagation();
            const collapsed = sidebar.classList.toggle('collapsed');
            localStorage.setItem('lumen_sidebar_collapsed', collapsed);
            
            sidebarToggleIcon.setAttribute('data-lucide', collapsed ? 'chevron-right' : 'chevron-left');
            lucide.createIcons();
        });
    }

    // Demo Mode Warning Overlay dismiss listener
    const btnDismissDemoWarning = document.getElementById('btn-dismiss-demo-warning');
    const demoWarningOverlay = document.getElementById('demo-mode-warning-overlay');
    if (btnDismissDemoWarning && demoWarningOverlay) {
        btnDismissDemoWarning.addEventListener('click', () => {
            demoWarningOverlay.classList.remove('show');
        });
    }

    // --- DYNAMIC TOAST NOTIFICATIONS ---
    function showToast(message, type = 'info', title = '') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const card = document.createElement('div');
        card.className = `toast-card toast-${type}`;
        
        let iconName = 'info';
        if (type === 'success') iconName = 'check-circle';
        if (type === 'warning') iconName = 'alert-triangle';
        if (type === 'danger') iconName = 'alert-octagon';

        if (!title) {
            title = type.charAt(0).toUpperCase() + type.slice(1);
            if (type === 'danger') title = 'Security Threat';
        }

        card.innerHTML = `
            <div class="toast-icon">
                <i data-lucide="${iconName}"></i>
            </div>
            <div class="toast-content">
                <span class="toast-title">${title}</span>
                <span class="toast-message">${message}</span>
            </div>
        `;

        container.appendChild(card);
        lucide.createIcons();

        // Auto remove after 5 seconds
        setTimeout(() => {
            card.classList.add('hide');
            card.addEventListener('animationend', () => {
                card.remove();
            });
        }, 5000);
    }

    // --- SECURITY CONTROLS HANDLERS ---
    let securityConfigCache = [];
    async function loadSecurityConfig() {
        try {
            const res = await fetch('/api/security/config');
            const config = await res.json();
            securityConfigCache = config.enabled_layers || [];
            
            const grid = document.getElementById('security-layers-grid');
            if (grid) {
                grid.innerHTML = '';
                const LAYER_DETAILS = {
                    1: { name: "Layer 1: Input Guard", desc: "Prompt injection detection, Unicode normalization, and prompt length restriction." },
                    2: { name: "Layer 2: Rate Limiting", desc: "IP rate limits, burst protection, and GPU token cap enforcement." },
                    3: { name: "Layer 3: File Security", desc: "MIME/magic bytes verification and path traversal prevention." },
                    4: { name: "Layer 4: API Security", desc: "CORS controls, max request size validation, and security headers (CSP, HSTS)." },
                    5: { name: "Layer 5: Security Mode", desc: "Detailed threat classifications and output PII scrubber." },
                    6: { name: "Layer 6: Pipeline Validator", desc: "Hyperparameter boundary validation for SFT/DPO execution." },
                    7: { name: "Layer 7: Encryption & Sessions", desc: "Optional ECDH session keys, nonces, and HMAC signature checks." },
                    8: { name: "Layer 8: Misuse Prevention", desc: "Jailbreak classification and persona hijack guardrails." },
                    9: { name: "Layer 9: Zero Trust Access", desc: "IP address allowlist validation and API token authorization." },
                    10: { name: "Layer 10: Data Leak Protection", desc: "Isolated database views and protected checkpoints downloads." },
                    11: { name: "Layer 11: Cryptographic Integrity", desc: "SHA-256 model signing and secure chain integrity validation." },
                    12: { name: "Layer 12: Exception Sanitizer", desc: "Suppresses raw stack traces and logs debug information securely." }
                };
                
                for (let i = 1; i <= 12; i++) {
                    const isChecked = securityConfigCache.includes(i);
                    const card = document.createElement('div');
                    card.className = 'glass-card security-layer-card';
                    card.style.cssText = 'padding: 15px; display: flex; align-items: center; justify-content: space-between; gap: 15px; border-radius: 12px;';
                    card.innerHTML = `
                        <div style="flex-grow: 1;">
                            <span style="font-size: 14px; font-weight: 700; color: white; display: block;">${LAYER_DETAILS[i].name}</span>
                            <span style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.4; display: block; margin-top: 4px;">${LAYER_DETAILS[i].desc}</span>
                        </div>
                        <label class="switch-container">
                            <input type="checkbox" class="security-layer-checkbox" data-layer="${i}" ${isChecked ? 'checked' : ''}>
                            <span class="switch-slider">
                                <span class="switch-handle"></span>
                            </span>
                        </label>
                    `;
                    grid.appendChild(card);
                }
            }
        } catch (e) {
            console.error("Error loading security config:", e);
        }
    }

    async function saveSecurityConfig() {
        const grid = document.getElementById('security-layers-grid');
        if (!grid) return;
        
        const checkboxes = grid.querySelectorAll('.security-layer-checkbox');
        const enabledLayers = [];
        checkboxes.forEach(cb => {
            if (cb.checked) {
                enabledLayers.push(parseInt(cb.getAttribute('data-layer')));
            }
        });
        
        const saveBtn = document.getElementById('btn-save-security-config');
        if (saveBtn) saveBtn.disabled = true;
        
        try {
            const res = await fetch('/api/security/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: { enabled_layers: enabledLayers } })
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast("12-Layer security settings dynamically saved and updated in-memory!", "success", "Security Configuration");
                securityConfigCache = enabledLayers;
            } else {
                alert(`Error saving security config: ${data.detail || 'Unknown error'}`);
            }
        } catch (e) {
            alert(`Error saving security config: ${e.message}`);
        } finally {
            if (saveBtn) saveBtn.disabled = false;
        }
    }

    let currentSecurityLogsPage = 1;
    async function loadSecurityLogs(page = 1) {
        currentSecurityLogsPage = page;
        try {
            const res = await fetch(`/api/security/audit-log?page=${page}&limit=12`);
            const data = await res.json();
            
            if (data.status === 'success') {
                const badge = document.getElementById('security-integrity-badge');
                if (badge) {
                    badge.innerText = data.is_intact ? "Integrity: Verified intact ✅" : "Integrity: TAMPERED DETECTED ❌";
                    badge.className = data.is_intact ? "badge badge-green" : "badge badge-red";
                }
                
                const tbody = document.getElementById('security-logs-body');
                if (tbody) {
                    tbody.innerHTML = '';
                    if (data.logs.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center">No safety events logged.</td></tr>';
                        return;
                    }
                    data.logs.forEach(log => {
                        const tr = document.createElement('tr');
                        tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                        
                        let badgeClass = 'badge';
                        if (log.threat_level === 'CRITICAL') badgeClass = 'badge badge-red';
                        else if (log.threat_level === 'HIGH' || log.threat_level === 'WARNING') badgeClass = 'badge badge-gold';
                        else if (log.threat_level === 'ERROR') badgeClass = 'badge badge-red';
                        else badgeClass = 'badge badge-blue';
                        
                        tr.innerHTML = `
                            <td style="padding: 10px; font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary);">${log.timestamp}</td>
                            <td style="padding: 10px; font-family: var(--font-mono); font-size: 12px; color: white;">${log.client_ip}</td>
                            <td style="padding: 10px; font-family: var(--font-mono); font-size: 12px; color: #a78bfa;">${log.endpoint}</td>
                            <td style="padding: 10px; text-align: center;">
                                <span class="${badgeClass}">${log.threat_level}</span>
                            </td>
                            <td style="padding: 10px; font-size: 12.5px; color: #c4c8d4; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${log.message}">${log.message}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }
                
                const indicator = document.getElementById('security-logs-page-indicator');
                if (indicator) indicator.innerText = `Page ${data.page}`;
                
                const prevBtn = document.getElementById('btn-security-logs-prev');
                const nextBtn = document.getElementById('btn-security-logs-next');
                
                if (prevBtn) prevBtn.disabled = data.page <= 1;
                if (nextBtn) nextBtn.disabled = (data.page * data.limit) >= data.total;
            }
        } catch (e) {
            console.error("Error loading security logs:", e);
        }
    }

    const btnSaveSecurityConfig = document.getElementById('btn-save-security-config');
    if (btnSaveSecurityConfig) {
        btnSaveSecurityConfig.addEventListener('click', saveSecurityConfig);
    }
    const btnSecurityLogsPrev = document.getElementById('btn-security-logs-prev');
    const btnSecurityLogsNext = document.getElementById('btn-security-logs-next');
    const btnRefreshSecurityLogs = document.getElementById('btn-refresh-security-logs');
    
    if (btnSecurityLogsPrev) {
        btnSecurityLogsPrev.addEventListener('click', () => {
            if (currentSecurityLogsPage > 1) {
                loadSecurityLogs(currentSecurityLogsPage - 1);
            }
        });
    }
    if (btnSecurityLogsNext) {
        btnSecurityLogsNext.addEventListener('click', () => {
            loadSecurityLogs(currentSecurityLogsPage + 1);
        });
    }
    if (btnRefreshSecurityLogs) {
        btnRefreshSecurityLogs.addEventListener('click', () => {
            loadSecurityLogs(currentSecurityLogsPage);
        });
    }

    // 15. Initial Loads
    renderCustomModesUI();
    setPlaygroundMode('default');
    loadConfigs();
    loadDatasetInfo();
    loadDatabaseRecords();
    loadDatabaseStats();
    loadRagDocuments();
    initPlaygroundMap();
    connectWebSocket();
    updatePreviewer();
});
