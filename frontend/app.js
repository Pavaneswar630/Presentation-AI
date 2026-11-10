/**
 * Presentation AI — Client Controller & Live HUD Telemetry
 */

(() => {
  // Application State
  const state = {
    sessionId: null,
    isActive: false,
    ws: null,
    mediaStream: null,
    frameIntervalId: null,
    timerIntervalId: null,
    sessionSeconds: 0,
    radarChart: null,
    timeSeriesChart: null,
    recognition: null,
    totalFillers: 0,
    totalWords: 0
  };

  // DOM Elements
  const el = {
    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    btnRequestProbe: document.getElementById('btnRequestProbe'),
    btnGenerateQuestions: document.getElementById('btnGenerateQuestions'),
    liveProbePanel: document.getElementById('liveProbePanel'),
    liveProbeContext: document.getElementById('liveProbeContext'),
    liveProbeText: document.getElementById('liveProbeText'),
    qaStatus: document.getElementById('qaStatus'),
    qaQuestions: document.getElementById('qaQuestions'),
    tabLiveBtn: document.getElementById('tabLiveBtn'),
    tabReportBtn: document.getElementById('tabReportBtn'),
    btnReturnLive: document.getElementById('btnReturnLive'),
    viewLive: document.getElementById('viewLive'),
    viewReport: document.getElementById('viewReport'),
    chkSimulated: document.getElementById('chkSimulated'),
    chkAudioStt: document.getElementById('chkAudioStt'),
    statusPill: document.getElementById('statusPill'),
    headerSlaDot: document.getElementById('headerSlaDot'),
    headerLatencyVal: document.getElementById('headerLatencyVal'),
    headerFpsVal: document.getElementById('headerFpsVal'),
    headerTimer: document.getElementById('headerTimer'),
    webcamVideo: document.getElementById('webcamVideo'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    alertToast: document.getElementById('alertToast'),
    alertCode: document.getElementById('alertCode'),
    alertMessage: document.getElementById('alertMessage'),
    hudShoulderTilt: document.getElementById('hudShoulderTilt'),
    hudSpineDev: document.getElementById('hudSpineDev'),
    hudHandJitter: document.getElementById('hudHandJitter'),
    hudStabilityVal: document.getElementById('hudStabilityVal'),
    barStability: document.getElementById('barStability'),
    hudWpmVal: document.getElementById('hudWpmVal'),
    barWpm: document.getElementById('barWpm'),
    hudFillerCount: document.getElementById('hudFillerCount'),
    hudFillerRatio: document.getElementById('hudFillerRatio'),
    transcriptBox: document.getElementById('transcriptBox'),
    // Report Elements
    repSessionId: document.getElementById('repSessionId'),
    repOverallScore: document.getElementById('repOverallScore'),
    repPostureScore: document.getElementById('repPostureScore'),
    repCadenceWpm: document.getElementById('repCadenceWpm'),
    repFillerCount: document.getElementById('repFillerCount'),
    repFillerRatio: document.getElementById('repFillerRatio'),
    repWordsSpoken: document.getElementById('repWordsSpoken'),
    repDuration: document.getElementById('repDuration'),
    repDefectCount: document.getElementById('repDefectCount'),
    defectsList: document.getElementById('defectsList'),
    coachingList: document.getElementById('coachingList')
  };

  // Hidden offscreen canvas for JPEG encoding
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = 320;
  offscreenCanvas.height = 240;
  const offscreenCtx = offscreenCanvas.getContext('2d');

  // Event Listeners
  el.btnStart.addEventListener('click', startSession);
  el.btnStop.addEventListener('click', stopSession);
  el.btnRequestProbe.addEventListener('click', requestLiveProbe);
  el.btnGenerateQuestions.addEventListener('click', generateAudienceQuestions);
  el.tabLiveBtn.addEventListener('click', () => switchTab('live'));
  el.tabReportBtn.addEventListener('click', () => switchTab('report'));
  el.btnReturnLive.addEventListener('click', () => switchTab('live'));

  function switchTab(tab) {
    if (tab === 'live') {
      el.viewLive.classList.remove('hidden');
      el.viewReport.classList.add('hidden');
      el.tabLiveBtn.className = 'px-3.5 py-1.5 rounded-lg font-medium transition-all bg-blue-600 text-white shadow';
      el.tabReportBtn.className = 'px-3.5 py-1.5 rounded-lg font-medium transition-all text-gray-400 hover:text-white';
    } else {
      el.viewLive.classList.add('hidden');
      el.viewReport.classList.remove('hidden');
      el.tabReportBtn.className = 'px-3.5 py-1.5 rounded-lg font-medium transition-all bg-blue-600 text-white shadow';
      el.tabLiveBtn.className = 'px-3.5 py-1.5 rounded-lg font-medium transition-all text-gray-400 hover:text-white';
    }
  }

  /**
   * 1. Start Coaching Session
   */
  async function startSession() {
    try {
      el.statusPill.textContent = 'Status: Initializing session...';
      el.btnStart.disabled = true;

      // REST API: POST /api/v1/sessions/start
      const res = await fetch('/api/v1/sessions/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'web_speaker',
          presentation_title: 'Live Keynote Rehearsal',
          config: { target_fps: 30.0, max_latency_sla_ms: 22.0 }
        })
      });

      if (!res.ok) throw new Error(`Server returned status ${res.status}`);
      const data = await res.json();
      state.sessionId = data.session_id;

      // Start Camera / Simulated Feed
      if (el.chkSimulated.checked) {
        startSimulatedVideo();
      } else {
        await startWebcam();
      }

      // Initialize WebSocket connection
      initWebSocket(data.session_id);

      // Start Speech Recognition (STT) if enabled
      if (el.chkAudioStt.checked) {
        initSpeechRecognition();
      }

      // Start 30 FPS Frame Capture Loop
      startFrameStreaming();

      // Start Timer
      state.sessionSeconds = 0;
      state.timerIntervalId = setInterval(updateTimer, 1000);

      state.isActive = true;
      el.btnStart.classList.add('opacity-50', 'cursor-not-allowed');
      el.btnStop.disabled = false;
      el.btnStop.classList.remove('opacity-50', 'cursor-not-allowed');
      el.statusPill.textContent = `Status: Streaming (${state.sessionId.slice(0, 12)}...)`;
      el.statusPill.classList.add('text-emerald-400');

    } catch (err) {
      console.error('Session start error:', err);
      alert(`Could not start coaching session: ${err.message}`);
      el.btnStart.disabled = false;
      el.statusPill.textContent = 'Status: Start Error';
    }
  }

  /**
   * 2. Stop & Analyze Session
   */
  async function stopSession() {
    if (!state.isActive) return;

    try {
      el.statusPill.textContent = 'Status: Analyzing via PyTorch & FAISS...';
      el.btnStop.disabled = true;

      // Halt capture, recognition and streaming
      clearInterval(state.frameIntervalId);
      clearInterval(state.timerIntervalId);
      if (state.recognition) state.recognition.stop();
      if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(t => t.stop());
      }
      if (state.ws) {
        state.ws.close();
      }

      // REST API: POST /api/v1/sessions/{id}/stop
      const stopRes = await fetch(`/api/v1/sessions/${state.sessionId}/stop`, {
        method: 'POST'
      });
      if (!stopRes.ok) throw new Error(`Stop failed: ${stopRes.status}`);

      // REST API: GET /api/v1/sessions/{id}/report
      const repRes = await fetch(`/api/v1/sessions/${state.sessionId}/report`);
      if (!repRes.ok) throw new Error(`Report fetch failed: ${repRes.status}`);
      const report = await repRes.json();

      // Render Evaluation Dashboard
      renderPostSessionReport(report);
      switchTab('report');

      state.isActive = false;
      el.btnStart.disabled = false;
      el.btnStart.classList.remove('opacity-50', 'cursor-not-allowed');
      el.statusPill.textContent = 'Status: Session Completed';
      el.statusPill.classList.remove('text-emerald-400');

    } catch (err) {
      console.error('Session stop error:', err);
      alert(`Error analyzing session: ${err.message}`);
    }
  }

  /**
   * Webcam Media Stream
   */
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false
      });
      state.mediaStream = stream;
      el.webcamVideo.srcObject = stream;
      await el.webcamVideo.play();
    } catch (err) {
      console.warn('Webcam not available or permission denied. Falling back to simulated feed:', err);
      el.chkSimulated.checked = true;
      startSimulatedVideo();
    }
  }

  /**
   * Synthetic Demo Video Canvas Generator (Fallback when no webcam)
   */
  function startSimulatedVideo() {
    const simCanvas = document.createElement('canvas');
    simCanvas.width = 640;
    simCanvas.height = 480;
    const ctx = simCanvas.getContext('2d');
    let frame = 0;

    function renderSimFrame() {
      if (!state.isActive && state.mediaStream) return;
      frame++;
      // Draw background studio
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 640, 480);

      // Draw simulated presenter silhouette
      ctx.fillStyle = '#334155';
      const sway = Math.sin(frame * 0.05) * 15;
      // Head
      ctx.beginPath();
      ctx.arc(320 + sway, 130, 45, 0, Math.PI * 2);
      ctx.fill();
      // Torso
      ctx.fillRect(250 + sway, 180, 140, 200);

      requestAnimationFrame(renderSimFrame);
    }
    renderSimFrame();
    el.webcamVideo.srcObject = simCanvas.captureStream(30);
  }

  /**
   * WebSocket Streaming & Telemetry Ingestion
   */
  function initWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/v1/session/${sessionId}`;
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      console.log('WebSocket connection established:', wsUrl);
    };

    state.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'live_metrics') {
          handleLiveMetrics(msg);
        } else if (msg.event === 'coaching_alert') {
          handleCoachingAlert(msg);
        } else if (msg.event === 'live_probe_question') {
          handleLiveProbe(msg.probe);
        }
      } catch (e) {
        console.warn('Malformed WS message:', e);
      }
    };

    state.ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
    };
  }

  function requestLiveProbe() {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      showLiveProbe('Live questions are available after the session connects.', 'Connection required');
      return;
    }

    state.ws.send(JSON.stringify({ type: 'request_live_probes' }));
  }

  function handleLiveProbe(probe) {
    if (!probe) return;
    showLiveProbe(probe.prompt_text || 'Can you support that point with a concrete example?', probe.context || 'Live audience probe');
  }

  function showLiveProbe(prompt, context) {
    el.liveProbeContext.textContent = context;
    el.liveProbeText.textContent = prompt;
    el.liveProbePanel.classList.remove('hidden');
  }

  /**
   * Continuous 30 FPS Frame Capture Loop
   */
  function startFrameStreaming() {
    const targetIntervalMs = 1000 / 30; // 33.3ms for 30 FPS
    let seq = 0;

    state.frameIntervalId = setInterval(() => {
      if (!state.isActive || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

      offscreenCtx.drawImage(el.webcamVideo, 0, 0, 320, 240);
      const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.6);

      seq++;
      const packet = {
        type: 'frame_packet',
        seq_id: seq,
        data: base64Data
      };
      state.ws.send(JSON.stringify(packet));
    }, targetIntervalMs);
  }

  /**
   * Real-Time Speech Recognition (STT) via Web Speech API
   */
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('SpeechRecognition API not supported in this browser.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const lastIdx = event.results.length - 1;
      const transcript = event.results[lastIdx][0].transcript.trim();
      const words = transcript.split(/\s+/);

      state.totalWords += words.length;

      // Filler word matching
      const fillers = ['um', 'uh', 'like', 'actually', 'basically'];
      for (const w of words) {
        if (fillers.includes(w.toLowerCase())) {
          state.totalFillers++;
        }
        // Send word token to backend over WebSocket
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
          state.ws.send(JSON.stringify({
            type: 'audio_packet',
            word: w
          }));
        }
      }

      // Update Live UI
      el.hudFillerCount.textContent = state.totalFillers;
      const ratio = state.totalWords > 0 ? ((state.totalFillers / state.totalWords) * 100).toFixed(1) : '0.0';
      el.hudFillerRatio.textContent = `${ratio}% ratio`;

      // Live WPM calculation
      const mins = Math.max(state.sessionSeconds / 60.0, 0.05);
      const curWpm = Math.round(state.totalWords / mins);
      el.hudWpmVal.textContent = `${curWpm} WPM`;
      el.barWpm.style.width = `${Math.min(100, (curWpm / 200) * 100)}%`;

      // Append to live speech box
      el.transcriptBox.innerHTML += ` <span class="text-white">${transcript}</span>`;
      el.transcriptBox.scrollTop = el.transcriptBox.scrollHeight;
    };

    recognition.onerror = (e) => console.warn('STT Error:', e);
    recognition.start();
    state.recognition = recognition;
  }

  /**
   * Process Inbound Live Telemetry Metrics
   */
  function handleLiveMetrics(msg) {
    const lat = msg.latency || {};
    const totalMs = lat.total_frame_latency_ms || 0.0;
    const slaMet = lat.sla_satisfied !== undefined ? lat.sla_satisfied : totalMs <= 22.0;

    // Header SLA & FPS Updates
    el.headerLatencyVal.textContent = `${totalMs.toFixed(1)} ms`;
    el.headerFpsVal.textContent = msg.pipeline_fps ? msg.pipeline_fps.toFixed(1) : '30.0';

    if (slaMet) {
      el.headerSlaDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500 sla-indicator-good';
      el.headerLatencyVal.className = 'text-emerald-400 font-semibold';
    } else {
      el.headerSlaDot.className = 'w-2.5 h-2.5 rounded-full bg-rose-500 sla-indicator-bad';
      el.headerLatencyVal.className = 'text-rose-400 font-semibold';
    }

    // Posture Metrics HUD
    const p = msg.posture || {};
    el.hudShoulderTilt.textContent = `${(p.shoulder_tilt_degrees || 0).toFixed(1)}°`;
    el.hudSpineDev.textContent = `${(p.spine_deviation_degrees || 0).toFixed(1)}°`;
    el.hudHandJitter.textContent = (p.hand_movement_velocity || 0).toFixed(3);

    const stab = p.stability_index !== undefined ? p.stability_index : 95.0;
    el.hudStabilityVal.textContent = stab.toFixed(1);
    el.barStability.style.width = `${stab}%`;

    // Draw Skeleton on HUD Canvas
    drawSkeletonOverlay(msg.landmarks);
  }

  /**
   * Draw Pose Landmark Overlay on Canvas
   */
  function drawSkeletonOverlay(landmarks) {
    if (!landmarks) return;

    const canvas = el.overlayCanvas;
    canvas.width = el.webcamVideo.videoWidth || 640;
    canvas.height = el.webcamVideo.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    function pt(lm) {
      return lm ? { x: lm.x * w, y: lm.y * h } : null;
    }

    const lSh = pt(landmarks.left_shoulder);
    const rSh = pt(landmarks.right_shoulder);
    const mSh = pt(landmarks.mid_shoulder);
    const mHp = pt(landmarks.mid_hip);
    const lWr = pt(landmarks.left_wrist);
    const rWr = pt(landmarks.right_wrist);
    const nose = pt(landmarks.nose);

    // 1. Draw Shoulder Alignment Line
    if (lSh && rSh) {
      ctx.beginPath();
      ctx.moveTo(lSh.x, lSh.y);
      ctx.lineTo(rSh.x, rSh.y);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 4;
      ctx.stroke();
    }

    // 2. Draw Spine Plumb Line
    if (mSh && mHp) {
      ctx.beginPath();
      ctx.moveTo(mHp.x, mHp.y);
      ctx.lineTo(mSh.x, mSh.y);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 4;
      ctx.stroke();
    }

    // 3. Draw Landmark Joints
    const joints = [lSh, rSh, mSh, mHp, lWr, rWr, nose];
    ctx.fillStyle = '#10b981';
    for (const j of joints) {
      if (j) {
        ctx.beginPath();
        ctx.arc(j.x, j.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  /**
   * Handle Real-Time Coaching Alert Toast
   */
  let alertTimeout = null;
  function handleCoachingAlert(msg) {
    el.alertCode.textContent = msg.code || 'COACHING ALERT';
    el.alertMessage.textContent = msg.message || 'Defect detected.';
    el.alertToast.classList.remove('hidden');

    clearTimeout(alertTimeout);
    alertTimeout = setTimeout(() => {
      el.alertToast.classList.add('hidden');
    }, 3500);
  }

  /**
   * Session Timer Updater
   */
  function updateTimer() {
    state.sessionSeconds++;
    const mins = Math.floor(state.sessionSeconds / 60);
    const secs = state.sessionSeconds % 60;
    const formatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    el.headerTimer.textContent = formatted;
  }

  /**
   * Render Comprehensive Post-Session Evaluation Report
   */
  function renderPostSessionReport(report) {
    const s = report.summary || {};
    const r = report.radar_dimensions || {};
    const ts = report.time_series_metrics || {};

    el.repSessionId.textContent = report.session_id || 'sess_...';
    el.repOverallScore.textContent = (s.overall_score || 85.0).toFixed(1);
    el.repPostureScore.textContent = (s.posture_stability_score || 88.0).toFixed(1);
    el.repCadenceWpm.textContent = (s.average_cadence_wpm || 145.0).toFixed(0);
    el.repFillerCount.textContent = s.filler_word_count || 0;
    el.repFillerRatio.textContent = `${(s.filler_word_ratio_pct || 0.0).toFixed(1)}% of speech`;
    el.repWordsSpoken.textContent = s.total_words_spoken || 0;

    const dur = s.duration_seconds || 0;
    const dMin = Math.floor(dur / 60);
    const dSec = Math.floor(dur % 60);
    el.repDuration.textContent = `${dMin}:${String(dSec).padStart(2, '0')}`;

    // Render 5-Axis Radar Chart
    renderRadarChart(r);

    // Render Time-Series Chart
    renderTimeSeriesChart(ts);

    // Render Detected Defects List
    renderDefects(report.detected_defects || []);

    // Render RAG Coaching Recommendations
    renderCoachingAdvice(report.rag_coaching_recommendations || []);

    el.qaStatus.classList.add('hidden');
    el.qaStatus.textContent = '';
    el.qaQuestions.replaceChildren();
    const qaPrompt = document.createElement('p');
    qaPrompt.className = 'text-xs text-gray-500 italic';
    qaPrompt.textContent = 'Generate questions based on the transcript from this session.';
    el.qaQuestions.appendChild(qaPrompt);
  }

  async function generateAudienceQuestions() {
    if (!state.sessionId) {
      el.qaStatus.textContent = 'Complete a session before generating audience questions.';
      el.qaStatus.classList.remove('hidden');
      return;
    }

    el.btnGenerateQuestions.disabled = true;
    el.qaStatus.textContent = 'Analyzing your transcript and coaching context...';
    el.qaStatus.classList.remove('hidden');
    try {
      const response = await fetch(`/api/v1/sessions/${encodeURIComponent(state.sessionId)}/generate-questions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question_count: 3, difficulty: 'probing' })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Server returned status ${response.status}`);
      }

      renderGeneratedQuestions(payload);
      el.qaStatus.textContent = payload.transcript_summary || 'Questions generated from your presentation.';
    } catch (err) {
      console.error('Question generation error:', err);
      el.qaStatus.textContent = `Could not generate questions: ${err.message}`;
    } finally {
      el.qaStatus.classList.remove('hidden');
      el.btnGenerateQuestions.disabled = false;
    }
  }

  function renderGeneratedQuestions(result) {
    const fragment = document.createDocumentFragment();
    for (const question of result.targeted_questions || []) {
      const card = document.createElement('article');
      card.className = 'p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2';

      const badge = document.createElement('span');
      badge.className = 'inline-flex px-2 py-1 rounded-full text-[10px] font-semibold bg-blue-950 text-blue-300 border border-blue-800';
      badge.textContent = question.category || 'Audience question';

      const prompt = document.createElement('p');
      prompt.className = 'text-sm font-semibold text-white';
      prompt.textContent = question.question_text || '';

      const rationale = document.createElement('p');
      rationale.className = 'text-xs text-gray-300';
      rationale.textContent = `Why they ask: ${question.rationale || 'Tests the depth of your presentation.'}`;

      const answerFocus = document.createElement('p');
      answerFocus.className = 'text-xs text-emerald-300';
      answerFocus.textContent = `Answer strategy: ${question.suggested_answer_focus || 'Support your response with clear evidence.'}`;

      card.append(badge, prompt, rationale, answerFocus);
      fragment.appendChild(card);
    }

    if (!fragment.childNodes.length) {
      const emptyState = document.createElement('p');
      emptyState.className = 'text-xs text-amber-300';
      emptyState.textContent = 'The question generator returned no questions. Please try again.';
      fragment.appendChild(emptyState);
    }
    el.qaQuestions.replaceChildren(fragment);
  }

  function renderRadarChart(radar) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    if (state.radarChart) state.radarChart.destroy();

    state.radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Posture', 'Pacing', 'Gestures', 'Clarity', 'Energy'],
        datasets: [{
          label: 'Your Delivery',
          data: [
            radar.posture || 85,
            radar.pacing || 90,
            radar.gesture_expressiveness || 80,
            radar.verbal_clarity || 88,
            radar.energy_consistency || 82
          ],
          backgroundColor: 'rgba(59, 130, 246, 0.25)',
          borderColor: '#3b82f6',
          pointBackgroundColor: '#60a5fa',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { display: false },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            pointLabels: { color: '#9ca3af', font: { size: 11, family: 'Inter' } }
          }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  function renderTimeSeriesChart(ts) {
    const ctx = document.getElementById('timeSeriesChart').getContext('2d');
    if (state.timeSeriesChart) state.timeSeriesChart.destroy();

    const labels = (ts.timestamps || []).map(t => `${t.toFixed(0)}s`);
    const stabilityData = ts.posture_stability_rolling || [];
    const wpmData = ts.wpm_rolling || [];

    state.timeSeriesChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels.length ? labels : ['0s', '15s', '30s', '45s', '60s'],
        datasets: [
          {
            label: 'Posture Stability',
            data: stabilityData.length ? stabilityData : [92, 88, 76, 89, 94],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Speaking Cadence (WPM)',
            data: wpmData.length ? wpmData : [140, 155, 165, 142, 145],
            borderColor: '#10b981',
            tension: 0.3,
            yAxisID: 'yWpm'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af' }
          },
          yWpm: {
            position: 'right',
            min: 50,
            max: 220,
            grid: { display: false },
            ticks: { color: '#10b981' }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af' }
          }
        },
        plugins: {
          legend: { labels: { color: '#e5e7eb', font: { family: 'Inter' } } }
        }
      }
    });
  }

  function renderDefects(defects) {
    el.repDefectCount.textContent = `${defects.length} detected`;
    if (!defects.length) {
      el.defectsList.innerHTML = '<p class="text-xs text-emerald-400 font-medium">✓ Outstanding form! No persistent defects flagged.</p>';
      return;
    }

    el.defectsList.innerHTML = defects.map(d => {
      const isHigh = d.severity === 'HIGH';
      const badgeCls = isHigh ? 'bg-rose-950 text-rose-400 border-rose-800' : 'bg-amber-950 text-amber-400 border-amber-800';
      return `
        <div class="p-3.5 rounded-xl bg-gray-900/60 border border-gray-800 space-y-1">
          <div class="flex items-center justify-between">
            <span class="font-bold text-xs text-white">${d.defect_type}</span>
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${badgeCls}">${d.severity}</span>
          </div>
          <p class="text-xs text-gray-300">${d.metric_value}</p>
          <p class="text-[10px] font-mono text-gray-500">Timeline: ${d.timestamp_range[0]}s — ${d.timestamp_range[1]}s</p>
        </div>
      `;
    }).join('');
  }

  function renderCoachingAdvice(adviceList) {
    if (!adviceList.length) {
      el.coachingList.innerHTML = '<p class="text-xs text-gray-500 italic">No specific recommendations required.</p>';
      return;
    }

    el.coachingList.innerHTML = adviceList.map(a => `
      <div class="p-4 rounded-xl bg-gray-900/60 border border-blue-900/30 space-y-2">
        <div class="flex items-center justify-between">
          <span class="font-bold text-xs text-blue-300">${a.topic}</span>
          <span class="text-[10px] font-mono text-gray-400">Match ${(a.similarity_score * 100).toFixed(0)}%</span>
        </div>
        <p class="text-xs text-gray-300 leading-relaxed">${a.recommendation}</p>
        <p class="text-[10px] font-mono text-gray-500">Source: ${a.source_transcript_id}</p>
      </div>
    `).join('');
  }

})();
