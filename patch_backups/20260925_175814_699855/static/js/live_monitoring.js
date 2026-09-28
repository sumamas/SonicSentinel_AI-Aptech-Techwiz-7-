(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    start: $('startMonitoring'), pause: $('pauseMonitoring'), stop: $('stopMonitoring'),
    micStatus: $('micStatus'), micDot: $('micStateDot'), timer: $('sessionTimer'), count: $('windowCount'),
    liveIndicator: $('liveIndicator'), streamState: $('streamState'), signalCaption: $('signalCaption'),
    canvas: $('waveformCanvas'), placeholder: $('waveformPlaceholder'), inputBar: $('inputLevelBar'), inputText: $('inputLevelText'),
    qualityBadge: $('qualityBadge'), empty: $('currentWindowEmpty'), content: $('currentWindowContent'),
    qualityLabel: $('qualityLabel'), qualityDescription: $('qualityDescription'), qualityIcon: $('qualityIcon'),
    duration: $('liveDuration'), sampleRate: $('liveSampleRate'), rms: $('liveRms'), silence: $('liveSilence'), clipping: $('liveClipping'), windowId: $('liveWindowId'),
    tbody: $('recentWindowsBody'), clear: $('clearWindows'), toast: $('liveToast'), detection: $('liveDetection'), detectionModel: $('liveDetectionModel'),
    confidence: $('liveConfidence'), severity: $('liveSeverity'), eventStatus: $('liveEventStatus'),
    pythonResult: $('pythonResult'), pythonDetail: $('pythonDetail'), pythonStatus: $('pythonStatus'),
    alertCard: $('alertCard'), alertIndicator: $('alertIndicator'), alertIcon: $('alertIcon'), alertTitle: $('alertTitle'), alertMessage: $('alertMessage'),
    confirmationState: $('confirmationState'), confirmationDetail: $('confirmationDetail'), captureMode: $('captureMode'),
  };

  if (!els.start || !els.canvas) return;

  // --- Live capture tuning ---
  const WINDOW_SECONDS = 3;
  const ROLLING_SECONDS = 6;
  const PEAK_PRE_SECONDS = 1.2;
  const PEAK_POST_SECONDS = 1.8;
  const SECOND_WINDOW_SHIFT_SECONDS = 0.35;
  const PEAK_COOLDOWN_SECONDS = 1.15;
  const CONTINUOUS_INTERVAL_SECONDS = 2.5;
  const MAX_RECENT_ROWS = 10;
  const HISTORY_MAX = 6;
  const HISTORY_MAX_AGE_MS = 9000;
  const REQUIRED_HITS = 2;

  const MIN_PEAK = 0.08;
  const MIN_RMS = 0.012;
  const MIN_TRANSIENT_CREST = 2.0;
  const CONFIRM_THRESHOLDS = {
    'Gunshot': 0.70,
    'Glass Breaking': 0.64,
    'Alarm or Siren': 0.68,
  };
  const SEVERITY_MAP = {
    'Gunshot': 'Critical',
    'Glass Breaking': 'High',
    'Alarm or Siren': 'High',
  };

  let stream = null, audioContext = null, sourceNode = null, analyserNode = null, processorNode = null, silentGain = null;
  let rafId = null, timerId = null, sessionStartedAt = null, pausedAt = null, pausedTotalMs = 0;
  let monitoring = false, paused = false, processedWindows = 0, queueBusy = false, serverWindowSequence = 0;
  let uploadQueue = [];

  // Rolling microphone chunks use absolute sample indexes so we can extract a
  // peak-centered 3-second window after enough post-event audio has arrived.
  let rollingChunks = [];
  let absoluteSampleCursor = 0;
  let pendingPeakEvents = [];
  let lastPeakTriggerSample = -Infinity;
  let nextContinuousSample = 0;
  let noiseFloorRms = 0.008;
  let recentRms = 0;
  let eventCounter = 0;
  let confirmationHistory = [];

  function showToast(message, isError = false) {
    if (!els.toast) return;
    els.toast.textContent = message;
    els.toast.classList.toggle('error', isError);
    els.toast.classList.add('show');
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => els.toast.classList.remove('show'), 2800);
  }

  function setMicState(label, state) {
    els.micStatus.textContent = label;
    els.micDot.className = `mic-state-dot state-${state}`;
  }

  function setStreamState(text) {
    els.streamState.textContent = text;
    els.liveIndicator.classList.toggle('active', text === 'Live');
  }

  function formatTimer(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    return `${String(Math.floor(total / 3600)).padStart(2,'0')}:${String(Math.floor((total % 3600) / 60)).padStart(2,'0')}:${String(total % 60).padStart(2,'0')}`;
  }

  function startTimer() {
    sessionStartedAt = Date.now(); pausedTotalMs = 0;
    timerId = window.setInterval(() => {
      const pauseAdjustment = paused && pausedAt ? Date.now() - pausedAt : 0;
      els.timer.textContent = formatTimer(Date.now() - sessionStartedAt - pausedTotalMs - pauseAdjustment);
    }, 250);
  }
  function stopTimer() { if (timerId) window.clearInterval(timerId); timerId = null; }

  function qualityDescription(label) {
    return ({
      Good: 'Signal is suitable for analysis.',
      Acceptable: 'Usable, but some quality limitations were detected.',
      Poor: 'Quality is weak and may require manual review.',
      Unusable: 'Signal is not suitable for reliable classification.',
    })[label] || 'Quality result received.';
  }
  function qualityClass(label) { return `quality-${String(label || 'waiting').toLowerCase().replace(/\s+/g,'-')}`; }
  function pct(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

  function renderAlert(data) {
    if (!els.alertCard) return;
    const active = Boolean(data.active_alert);
    els.alertCard.classList.toggle('alert-card-active', active);
    els.alertIndicator.className = active ? 'alert-danger-dot' : 'alert-safe-dot';
    if (active) {
      els.alertIcon.className = 'alert-icon-active';
      els.alertIcon.innerHTML = '<i class="bi bi-exclamation-triangle-fill"></i>';
      els.alertTitle.textContent = `${data.severity || 'High'} alert — ${data.python_model?.prediction || 'Sound detected'}`;
      els.alertMessage.textContent = data.decision_note || 'Repeated windows confirmed this event.';
    } else {
      els.alertIcon.className = '';
      els.alertIcon.innerHTML = '<i class="bi bi-shield-check"></i>';
      els.alertTitle.textContent = data.event_status?.startsWith('Confirming') ? 'Candidate sound — confirming' : 'No active alert';
      els.alertMessage.textContent = data.decision_note || 'No confirmed alert is active.';
    }
  }

  function topMargin(model) {
    const top = Array.isArray(model.top3) ? model.top3 : [];
    if (!top.length) return 0;
    return Number(top[0]?.confidence || 0) - Number(top[1]?.confidence || 0);
  }

  function applyRepeatedConfirmation(data, meta) {
    const model = data.python_model || {};
    if (!data.prediction_enabled || !model.prediction) return data;

    const now = Date.now();
    const item = {
      time: now,
      label: model.prediction,
      confidence: Number(model.confidence || 0),
      margin: topMargin(model),
      token: meta?.eventToken || '',
      mode: meta?.mode || 'unknown',
    };
    confirmationHistory.push(item);
    confirmationHistory = confirmationHistory.filter(x => now - x.time <= HISTORY_MAX_AGE_MS).slice(-HISTORY_MAX);

    const recent = confirmationHistory.slice(-3);
    const same = recent.filter(x => x.label === item.label);
    const threshold = CONFIRM_THRESHOLDS[item.label] ?? 0.72;
    const avgConfidence = same.length ? same.reduce((a,b) => a + b.confidence, 0) / same.length : 0;
    const avgMargin = same.length ? same.reduce((a,b) => a + b.margin, 0) / same.length : 0;

    // Two matching overlapping/adjacent windows are required. A small
    // top-1-vs-top-2 margin protects against ambiguous phone-speaker captures.
    // Event-shape gating adds another safeguard: impulsive Glass/Gunshot needs
    // peak evidence, while Alarm/Siren needs at least one sustained window.
    let modeCompatible = true;
    if (item.label === 'Alarm or Siren') modeCompatible = same.some(x => x.mode === 'continuous');
    if (item.label === 'Glass Breaking' || item.label === 'Gunshot') modeCompatible = same.some(x => x.mode === 'peak');
    const confirmed = same.length >= REQUIRED_HITS && avgConfidence >= threshold && avgMargin >= 0.12 && modeCompatible;
    const hits = Math.min(same.length, REQUIRED_HITS);

    data.confirmation = {
      confirmed,
      hits,
      required: REQUIRED_HITS,
      label: item.label,
      average_confidence: avgConfidence,
      average_margin: avgMargin,
      capture_mode: item.mode,
      mode_compatible: modeCompatible,
    };

    if (confirmed) {
      const severity = SEVERITY_MAP[item.label] || data.severity || 'Informational';
      data.severity = severity;
      data.event_status = severity === 'Critical' || severity === 'High' ? 'Confirmed Alert' : 'Confirmed';
      data.active_alert = severity === 'Critical' || severity === 'High';
      data.decision_note = `${item.label} confirmed in ${hits}/${REQUIRED_HITS} recent windows (avg ${pct(avgConfidence)}).`;
    } else {
      data.active_alert = false;
      data.event_status = `Confirming ${hits}/${REQUIRED_HITS}`;
      data.decision_note = `${item.label} is a candidate. Waiting for a consistent repeated window before alerting.`;
    }
    return data;
  }

  function renderConfirmation(data) {
    const c = data.confirmation;
    if (!c) {
      if (els.confirmationState) els.confirmationState.textContent = 'Waiting';
      if (els.confirmationDetail) els.confirmationDetail.textContent = '2 matching windows are required.';
      return;
    }
    if (els.confirmationState) {
      els.confirmationState.textContent = c.confirmed ? `Confirmed ${c.hits}/${c.required}` : `Confirming ${c.hits}/${c.required}`;
      els.confirmationState.className = c.confirmed ? 'confirmation-value confirmed' : 'confirmation-value pending';
    }
    if (els.confirmationDetail) {
      const gate = c.mode_compatible ? 'event shape OK' : 'waiting for matching event shape';
      els.confirmationDetail.textContent = `Avg ${pct(c.average_confidence)} • margin ${pct(c.average_margin)} • ${gate}`;
    }
    if (els.captureMode) els.captureMode.textContent = c.capture_mode === 'peak' ? 'Peak-centered' : 'Continuous check';
  }

  function renderPrediction(data) {
    const model = data.python_model || {};
    if (!data.prediction_enabled || !model.prediction) {
      els.detection.textContent = 'Prediction unavailable';
      els.detectionModel.textContent = data.message || 'Trained model not available';
      els.confidence.textContent = '—'; els.severity.textContent = '—';
      els.eventStatus.textContent = data.event_status || 'Inspected';
      els.pythonResult.textContent = 'Model unavailable';
      els.pythonDetail.textContent = data.message || 'Copy the trained model files into /models.';
      els.pythonStatus.textContent = 'Unavailable'; els.pythonStatus.className = 'model-chip model-chip-error';
      renderConfirmation(data); renderAlert(data); return;
    }
    els.detection.textContent = model.prediction;
    els.detectionModel.textContent = `Python model: ${String(model.model || 'selected').toUpperCase()}`;
    els.confidence.textContent = pct(model.confidence);
    els.severity.textContent = data.severity || '—';
    els.eventStatus.textContent = data.event_status || 'Classified';
    els.pythonResult.textContent = `${model.prediction} • ${pct(model.confidence)}`;
    els.pythonDetail.textContent = `${String(model.model || 'selected').toUpperCase()} • ${data.event_status || 'Classified'}`;
    els.pythonStatus.textContent = data.confirmation?.confirmed ? 'Confirmed' : 'Candidate';
    els.pythonStatus.className = data.confirmation?.confirmed ? 'model-chip model-chip-ready' : 'model-chip model-chip-waiting';
    renderConfirmation(data); renderAlert(data);
  }

  function renderInspection(data, meta) {
    applyRepeatedConfirmation(data, meta);
    const q = data.quality || {}; const label = q.label || 'Unknown';
    processedWindows += 1; serverWindowSequence += 1; els.count.textContent = String(processedWindows);
    els.empty.hidden = true; els.content.hidden = false;
    els.qualityBadge.className = `quality-badge ${qualityClass(label)}`; els.qualityBadge.textContent = label;
    els.qualityLabel.textContent = label; els.qualityDescription.textContent = qualityDescription(label);
    els.duration.textContent = `${Number(data.duration_seconds || 0).toFixed(2)} sec`;
    els.sampleRate.textContent = `${Number(data.sample_rate || 0).toLocaleString()} Hz`;
    els.rms.textContent = Number(q.rms || 0).toFixed(6);
    els.silence.textContent = `${(Number(q.silence_ratio || 0) * 100).toFixed(1)}%`;
    els.clipping.textContent = `${(Number(q.clipping_ratio || 0) * 100).toFixed(2)}%`;
    els.windowId.textContent = `#${serverWindowSequence}`;
    els.qualityIcon.className = `quality-icon ${qualityClass(label)}`;
    els.qualityIcon.innerHTML = label === 'Good' ? '<i class="bi bi-check2-circle"></i>' : label === 'Acceptable' ? '<i class="bi bi-exclamation-circle"></i>' : '<i class="bi bi-exclamation-triangle"></i>';
    renderPrediction(data); addRecentRow(serverWindowSequence, label, q, data, meta);
  }

  function addRecentRow(number, label, q, data, meta) {
    const emptyRow = $('recentWindowsEmpty'); if (emptyRow) emptyRow.remove();
    const model = data.python_model || {}; const row = document.createElement('tr'); const now = new Date();
    const confirmation = data.confirmation?.confirmed ? 'Confirmed' : `${data.confirmation?.hits || 0}/${REQUIRED_HITS}`;
    row.innerHTML = `
      <td>#${number}</td>
      <td>${now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</td>
      <td><strong>${escapeHtml(model.prediction || 'Unavailable')}</strong><small class="capture-sub">${escapeHtml(meta?.mode === 'peak' ? 'Peak-centered' : 'Continuous')}</small></td>
      <td>${model.confidence == null ? '—' : pct(model.confidence)}</td>
      <td><span class="confirm-pill ${data.confirmation?.confirmed ? 'is-confirmed' : ''}">${escapeHtml(confirmation)}</span></td>
      <td><span class="severity-pill severity-${escapeHtml(String(data.severity || 'none').toLowerCase())}">${escapeHtml(data.severity || '—')}</span></td>
      <td><span class="live-quality-pill ${qualityClass(label)}">${escapeHtml(label)}</span></td>
      <td>${Number(q.rms || 0).toFixed(5)}</td>`;
    els.tbody.prepend(row);
    while (els.tbody.children.length > MAX_RECENT_ROWS) els.tbody.removeChild(els.tbody.lastElementChild);
  }

  async function checkMicrophoneAvailability() {
    if (!navigator.mediaDevices?.getUserMedia) { setMicState('Not supported in this browser','error'); els.start.disabled = true; return; }
    if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') { setMicState('HTTPS is required for microphone access','error'); els.start.disabled = true; return; }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasMic = devices.some(d => d.kind === 'audioinput');
      setMicState(hasMic ? 'Available — ready to start' : 'No microphone detected', hasMic ? 'ready' : 'error'); els.start.disabled = !hasMic;
    } catch (_) { setMicState('Ready — permission required','ready'); }
  }

  function blockStats(samples) {
    let sum = 0, peak = 0, peakIndex = 0;
    for (let i=0;i<samples.length;i++) { const v=Math.abs(samples[i]); sum += v*v; if (v>peak) {peak=v; peakIndex=i;} }
    const rms = Math.sqrt(sum / Math.max(samples.length,1));
    return {rms, peak, peakIndex, crest: peak / Math.max(rms, 1e-6)};
  }

  function appendRolling(samples) {
    const start = absoluteSampleCursor, end = start + samples.length;
    rollingChunks.push({start, end, data: samples}); absoluteSampleCursor = end;
    const keepFrom = absoluteSampleCursor - Math.round(audioContext.sampleRate * ROLLING_SECONDS);
    while (rollingChunks.length && rollingChunks[0].end < keepFrom) rollingChunks.shift();
  }

  function extractAbsolute(start, end) {
    const length = end - start; if (length <= 0) return null;
    const out = new Float32Array(length); let written = 0;
    for (const chunk of rollingChunks) {
      const a = Math.max(start, chunk.start), b = Math.min(end, chunk.end);
      if (b <= a) continue;
      const srcStart = a - chunk.start, dstStart = a - start, count = b - a;
      out.set(chunk.data.subarray(srcStart, srcStart + count), dstStart); written += count;
    }
    return written >= length * 0.98 ? out : null;
  }

  function queueWindow(samples, meta) {
    if (!samples || !audioContext) return;
    uploadQueue.push({blob: encodeWav(samples, audioContext.sampleRate), meta});
    if (uploadQueue.length > 6) uploadQueue = uploadQueue.slice(-6);
    processUploadQueue();
  }

  function schedulePeak(globalPeakSample, stats) {
    const sr = audioContext.sampleRate;
    if ((globalPeakSample - lastPeakTriggerSample) < sr * PEAK_COOLDOWN_SECONDS) return;
    lastPeakTriggerSample = globalPeakSample;
    eventCounter += 1;
    pendingPeakEvents.push({
      token: `peak-${eventCounter}-${Date.now()}`,
      peakSample: globalPeakSample,
      readySample: globalPeakSample + Math.round(sr * (PEAK_POST_SECONDS + SECOND_WINDOW_SHIFT_SECONDS + 0.08)),
      peakScore: stats.peak,
    });
    els.signalCaption.textContent = 'Transient detected • collecting peak-centered windows…';
  }

  function processPendingPeakEvents() {
    if (!audioContext || !pendingPeakEvents.length) return;
    const sr = audioContext.sampleRate, windowLength = Math.round(sr * WINDOW_SECONDS);
    const ready = [], waiting = [];
    for (const ev of pendingPeakEvents) (absoluteSampleCursor >= ev.readySample ? ready : waiting).push(ev);
    pendingPeakEvents = waiting;

    for (const ev of ready) {
      // Two overlapping views of the same transient give genuine repeated-window
      // confirmation while keeping the acoustic peak inside both 3-sec windows.
      const start1 = Math.round(ev.peakSample - sr * PEAK_PRE_SECONDS);
      const start2 = Math.round(start1 - sr * SECOND_WINDOW_SHIFT_SECONDS);
      const w1 = extractAbsolute(start1, start1 + windowLength);
      const w2 = extractAbsolute(start2, start2 + windowLength);
      queueWindow(w1, {mode:'peak', eventToken:ev.token, peakScore:ev.peakScore, variant:'A'});
      queueWindow(w2, {mode:'peak', eventToken:ev.token, peakScore:ev.peakScore, variant:'B'});
    }
  }

  function maybeContinuousWindow() {
    if (!audioContext || absoluteSampleCursor < nextContinuousSample) return;
    const sr = audioContext.sampleRate;
    nextContinuousSample = absoluteSampleCursor + Math.round(sr * CONTINUOUS_INTERVAL_SECONDS);
    // Fallback for sirens/alarms that are sustained rather than impulsive.
    const activityThreshold = Math.max(MIN_RMS, noiseFloorRms * 1.7);
    if (recentRms < activityThreshold) return;
    const end = absoluteSampleCursor, start = end - Math.round(sr * WINDOW_SECONDS);
    const samples = extractAbsolute(start, end);
    if (samples) queueWindow(samples, {mode:'continuous', eventToken:`continuous-${Date.now()}`, peakScore:0});
  }

  function handleAudioBlock(samples) {
    if (!monitoring || paused || !audioContext) return;
    const stats = blockStats(samples);
    const blockStart = absoluteSampleCursor;
    appendRolling(samples);

    recentRms = recentRms * 0.82 + stats.rms * 0.18;
    // Noise floor follows quiet audio slowly but does not chase a loud event.
    if (stats.rms < Math.max(0.035, noiseFloorRms * 2.2)) noiseFloorRms = noiseFloorRms * 0.985 + stats.rms * 0.015;
    noiseFloorRms = Math.min(Math.max(noiseFloorRms, 0.003), 0.035);

    const adaptivePeak = Math.max(MIN_PEAK, noiseFloorRms * 5.5);
    const adaptiveRms = Math.max(MIN_RMS, noiseFloorRms * 2.1);
    const transient = stats.peak >= adaptivePeak && stats.rms >= adaptiveRms && stats.crest >= MIN_TRANSIENT_CREST;
    if (transient) schedulePeak(blockStart + stats.peakIndex, stats);

    processPendingPeakEvents();
    maybeContinuousWindow();
  }

  function encodeWav(samples, sampleRate) {
    const bytesPerSample=2, buffer=new ArrayBuffer(44+samples.length*bytesPerSample), view=new DataView(buffer);
    const writeString=(offset,value)=>{for(let i=0;i<value.length;i++) view.setUint8(offset+i,value.charCodeAt(i));};
    writeString(0,'RIFF'); view.setUint32(4,36+samples.length*bytesPerSample,true); writeString(8,'WAVE'); writeString(12,'fmt ');
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true); view.setUint32(24,sampleRate,true);
    view.setUint32(28,sampleRate*bytesPerSample,true); view.setUint16(32,bytesPerSample,true); view.setUint16(34,16,true);
    writeString(36,'data'); view.setUint32(40,samples.length*bytesPerSample,true);
    let offset=44; for(let i=0;i<samples.length;i++){const s=Math.max(-1,Math.min(1,samples[i])); view.setInt16(offset,s<0?s*0x8000:s*0x7fff,true); offset+=2;}
    return new Blob([view],{type:'audio/wav'});
  }

  async function processUploadQueue() {
    if (queueBusy || uploadQueue.length===0 || !monitoring) return;
    queueBusy=true; const item=uploadQueue.shift(); const formData=new FormData();
    formData.append('audio', item.blob, `live-window-${Date.now()}.wav`);
    formData.append('capture_mode', item.meta?.mode || 'unknown');
    formData.append('event_token', item.meta?.eventToken || '');
    formData.append('peak_score', String(item.meta?.peakScore || 0));
    try {
      els.signalCaption.textContent = item.meta?.mode === 'peak' ? 'Classifying peak-centered transient…' : 'Classifying sustained audio…';
      const response=await fetch('/api/live/inspect',{method:'POST',body:formData,credentials:'same-origin'}); const data=await response.json();
      if(!response.ok) throw new Error(data.error || 'Live audio inspection failed.');
      renderInspection(data,item.meta); els.signalCaption.textContent='Microphone active • transient + repeated confirmation';
    } catch(error) { showToast(error.message || 'Could not inspect live audio.',true); els.signalCaption.textContent='Microphone active • backend inspection error'; }
    finally { queueBusy=false; if(uploadQueue.length) processUploadQueue(); }
  }

  function drawWaveform() {
    if(!analyserNode||!monitoring)return; const canvas=els.canvas, rect=canvas.getBoundingClientRect(), ratio=Math.max(1,window.devicePixelRatio||1);
    const width=Math.max(1,Math.floor(rect.width*ratio)), height=Math.max(1,Math.floor(rect.height*ratio)); if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    const ctx=canvas.getContext('2d'), data=new Uint8Array(analyserNode.fftSize); analyserNode.getByteTimeDomainData(data); ctx.clearRect(0,0,width,height);
    ctx.lineWidth=2.1*ratio; const gradient=ctx.createLinearGradient(0,0,width,0); gradient.addColorStop(0,'#2fb8a8'); gradient.addColorStop(.45,'#2c9fd4'); gradient.addColorStop(1,'#317fce'); ctx.strokeStyle=gradient; ctx.beginPath();
    let sum=0; for(let i=0;i<data.length;i++){const n=(data[i]-128)/128; sum+=n*n; const x=(i/(data.length-1))*width, y=(.5+n*.38)*height; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);} ctx.stroke();
    const rms=Math.sqrt(sum/data.length), level=Math.min(100,Math.round(rms*260)); els.inputBar.style.width=`${level}%`; els.inputText.textContent=`${level}%`; rafId=window.requestAnimationFrame(drawWaveform);
  }

  async function startMonitoring() {
    if(monitoring)return; if(!navigator.mediaDevices?.getUserMedia){showToast('This browser does not support microphone capture.',true);return;}
    try {
      setMicState('Requesting permission…','ready');
      stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
      const AudioContextCtor=window.AudioContext||window.webkitAudioContext; audioContext=new AudioContextCtor(); if(audioContext.state==='suspended')await audioContext.resume();
      sourceNode=audioContext.createMediaStreamSource(stream); analyserNode=audioContext.createAnalyser(); analyserNode.fftSize=1024; analyserNode.smoothingTimeConstant=.72;
      processorNode=audioContext.createScriptProcessor(4096,1,1); silentGain=audioContext.createGain(); silentGain.gain.value=0;
      sourceNode.connect(analyserNode); sourceNode.connect(processorNode); processorNode.connect(silentGain); silentGain.connect(audioContext.destination);
      processorNode.onaudioprocess=(event)=>{if(!monitoring||paused)return; handleAudioBlock(new Float32Array(event.inputBuffer.getChannelData(0)));};

      monitoring=true; paused=false; uploadQueue=[]; queueBusy=false; processedWindows=0; serverWindowSequence=0;
      rollingChunks=[]; absoluteSampleCursor=0; pendingPeakEvents=[]; lastPeakTriggerSample=-Infinity; nextContinuousSample=Math.round(audioContext.sampleRate*WINDOW_SECONDS);
      noiseFloorRms=.008; recentRms=0; eventCounter=0; confirmationHistory=[]; els.count.textContent='0';
      els.start.disabled=true; els.pause.disabled=false; els.stop.disabled=false; els.pause.innerHTML='<i class="bi bi-pause-fill"></i><span>Pause</span>'; els.placeholder.hidden=true;
      setMicState('Active — listening now','active'); setStreamState('Live'); els.signalCaption.textContent='Microphone active • transient detector armed';
      if(els.confirmationState){els.confirmationState.textContent='Waiting';els.confirmationState.className='confirmation-value pending';}
      if(els.confirmationDetail)els.confirmationDetail.textContent='A sound needs 2 matching windows before confirmation.';
      startTimer(); drawWaveform(); showToast('Peak-based live monitoring started.');
      const track=stream.getAudioTracks()[0]; if(track)track.addEventListener('ended',()=>{if(monitoring){showToast('Microphone disconnected.',true);stopMonitoring('disconnected');}});
    } catch(error) {
      const denied=error?.name==='NotAllowedError'||error?.name==='PermissionDeniedError'; setMicState(denied?'Permission denied':'Microphone unavailable','error'); setStreamState('Error');
      showToast(denied?'Microphone permission was denied. Allow access in your browser settings.':(error.message||'Could not start microphone.'),true);
    }
  }

  async function togglePause() {
    if(!monitoring||!audioContext)return;
    if(!paused){paused=true;pausedAt=Date.now();await audioContext.suspend();els.pause.innerHTML='<i class="bi bi-play-fill"></i><span>Resume</span>';setMicState('Paused','paused');setStreamState('Paused');els.signalCaption.textContent='Monitoring paused';if(rafId)cancelAnimationFrame(rafId);rafId=null;}
    else{await audioContext.resume();paused=false;if(pausedAt)pausedTotalMs+=Date.now()-pausedAt;pausedAt=null;els.pause.innerHTML='<i class="bi bi-pause-fill"></i><span>Pause</span>';setMicState('Active — listening now','active');setStreamState('Live');els.signalCaption.textContent='Microphone active • transient detector armed';drawWaveform();}
  }

  async function stopMonitoring(reason='stopped') {
    if(!monitoring&&!stream)return; monitoring=false;paused=false;uploadQueue=[];rollingChunks=[];pendingPeakEvents=[];confirmationHistory=[];
    if(rafId)cancelAnimationFrame(rafId);rafId=null;stopTimer();
    if(processorNode){processorNode.onaudioprocess=null;try{processorNode.disconnect();}catch(_){}} if(sourceNode){try{sourceNode.disconnect();}catch(_){}} if(analyserNode){try{analyserNode.disconnect();}catch(_){}} if(silentGain){try{silentGain.disconnect();}catch(_){}}
    if(stream)stream.getTracks().forEach(track=>track.stop()); if(audioContext&&audioContext.state!=='closed'){try{await audioContext.close();}catch(_){}}
    stream=null;audioContext=null;sourceNode=null;analyserNode=null;processorNode=null;silentGain=null;pausedAt=null;
    els.start.disabled=false;els.pause.disabled=true;els.stop.disabled=true;els.pause.innerHTML='<i class="bi bi-pause-fill"></i><span>Pause</span>';els.inputBar.style.width='0%';els.inputText.textContent='0%';els.placeholder.hidden=false;
    setStreamState(reason==='disconnected'?'Disconnected':'Stopped');setMicState(reason==='disconnected'?'Disconnected':'Available — ready to start',reason==='disconnected'?'error':'ready');els.signalCaption.textContent=reason==='disconnected'?'Microphone disconnected':'Monitoring stopped';if(reason!=='disconnected')showToast('Live monitoring stopped.');
  }

  els.start.addEventListener('click',startMonitoring); els.pause.addEventListener('click',togglePause); els.stop.addEventListener('click',()=>stopMonitoring('stopped'));
  els.clear?.addEventListener('click',()=>{els.tbody.innerHTML='<tr id="recentWindowsEmpty"><td colspan="8" class="live-empty-row"><i class="bi bi-soundwave"></i> No windows processed yet.</td></tr>';confirmationHistory=[];if(els.confirmationState)els.confirmationState.textContent='Waiting';});
  window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(track=>track.stop());});
  checkMicrophoneAvailability();
})();
