
        // --- Configuration ---
        const API_BASE_URL = "http://localhost:8000";

        // --- UI Elements ---
        const micBtn = document.getElementById('micBtn');
        const audioUpload = document.getElementById('audioUpload');
        const submitAudioBtn = document.getElementById('submitAudioBtn');
        const audioPlayback = document.getElementById('audioPlayback');
        const textQuery = document.getElementById('textQuery');
        const submitTextBtn = document.getElementById('submitTextBtn');
        
        const transcriptionOut = document.getElementById('transcriptionOut');
        const answerOut = document.getElementById('answerOut');

        // --- State Variables ---
        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];
        let recordedBlob = null;
        
        // Timers
        let recordTimerInterval;
        let statusCyclerInterval;

        // --- Real-time Status Simulator ---
        function startStatusCycler(inputType) {
            // 1. Dynamic pipeline text based on input type
            const pipelineSteps = inputType === 'audio' 
                ? [
                    "🎙️ Extracting & Transcribing Audio (Sarvam STT)...",
                    "🛡️ Running Guardrail 1: Input Safety...",
                    "🔍 Executing Hybrid Search (Qdrant + SQLite)...",
                    "📊 Cross-Encoder Reranking...",
                    "🧠 Generating Response (Sarvam-105B)..."
                  ]
                : [
                    "🛡️ Running Guardrail 1: Input Safety...",
                    "🔍 Executing Hybrid Search (Qdrant + SQLite)...",
                    "📊 Cross-Encoder Reranking...",
                    "🧠 Generating Response (Sarvam-105B)..."
                  ];

            let stepIndex = 0;
            answerOut.innerHTML = `<em>${pipelineSteps[stepIndex]}</em>`;

            // Cycle to the next status every 1.8 seconds to simulate pipeline progress
            statusCyclerInterval = setInterval(() => {
                stepIndex++;
                if (stepIndex < pipelineSteps.length) {
                    answerOut.innerHTML = `<em>${pipelineSteps[stepIndex]}</em>`;
                } else {
                    // Stick on the last step until the fetch resolves
                    answerOut.innerHTML = `<em>${pipelineSteps[pipelineSteps.length - 1]}<br>(Awaiting final payload...)</em>`;
                    clearInterval(statusCyclerInterval);
                }
            }, 1800);
        }

        function stopStatusCycler() {
            clearInterval(statusCyclerInterval);
        }

        // --- Core Fetch Function ---
        async function executeBackendRequest(endpoint, fetchOptions, inputType) {
            transcriptionOut.textContent = inputType === 'audio' ? "Transmitting audio blob..." : textQuery.value;
            
            // Start UI feedback
            startStatusCycler(inputType);

            try {
                const response = await fetch(`${API_BASE_URL}${endpoint}`, fetchOptions);
                
                if (!response.ok) {
                    throw new Error(`HTTP Error: ${response.status} - ${response.statusText}`);
                }

                const data = await response.json();
                
                // Stop the cycler once we have the data
                stopStatusCycler();

                // Render the transcribed query (useful for audio)
                transcriptionOut.textContent = data.query || "No transcription available.";

                // Handle Guardrail Failures vs Success
                if (!data.safe) {
                    answerOut.innerHTML = `<span style="color: var(--goa-pink); font-weight: bold;">[Pipeline Blocked]</span> Error: ${data.error}<br><br>${data.response}`;
                } else {
                    answerOut.textContent = data.response;
                }

            } catch (error) {
                stopStatusCycler();
                console.error("Pipeline Request Failed:", error);
                transcriptionOut.textContent = "Request Failed.";
                answerOut.innerHTML = `<span style="color: red;">Connection Error: Ensure FastAPI is running at ${API_BASE_URL}</span>`;
            }
        }

        // --- Event Listeners ---

        // 1. Microphone Logic with Timer
        micBtn.addEventListener('click', async () => {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const audioUrl = URL.createObjectURL(recordedBlob);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = 'block';
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.classList.add('recording');

                    // 3. Start Timer Display
                    const startTime = Date.now();
                    micBtn.textContent = `⏹ Stop (00:00)`;
                    recordTimerInterval = setInterval(() => {
                        const elapsed = Math.floor((Date.now() - startTime) / 1000);
                        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                        const secs = String(elapsed % 60).padStart(2, '0');
                        micBtn.textContent = `⏹ Stop (${mins}:${secs})`;
                    }, 1000);

                } catch (err) {
                    alert("Microphone access denied. Please use localhost or HTTPS.");
                }
            } else {
                // Stop Recording & Clean up Timer
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                clearInterval(recordTimerInterval);
                micBtn.textContent = "🎤 Record Again";
                micBtn.classList.remove('recording');
            }
        });

        // 2. Submit Audio -> POST /chat/audio
        submitAudioBtn.addEventListener('click', () => {
            const formData = new FormData();

            if (audioUpload.files.length > 0) {
                formData.append("audio", audioUpload.files[0]);
            } else if (recordedBlob) {
                formData.append("audio", recordedBlob, "recording.webm");
            } else {
                alert("Please record something or upload a file first.");
                return;
            }

            executeBackendRequest("/chat/audio", {
                method: "POST",
                body: formData
            }, 'audio'); // Pass 'audio' flag
        });

        // 3. Submit Text -> POST /chat/text
        submitTextBtn.addEventListener('click', () => {
            const textValue = textQuery.value.trim();
            if (textValue === "") {
                alert("Please type a query.");
                return;
            }

            executeBackendRequest("/chat/text", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: textValue })
            }, 'text'); // Pass 'text' flag
        });