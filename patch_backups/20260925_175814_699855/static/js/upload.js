(() => {
  const form = document.getElementById('audioInspectForm');
  if (!form) return;

  const fileInput = document.getElementById('audioFile');
  const dropzone = document.getElementById('dropzone');
  const selected = document.getElementById('selectedFile');
  const state = document.getElementById('resultState');
  const empty = document.getElementById('resultEmpty');
  const content = document.getElementById('resultContent');
  const predictionCard = document.getElementById('predictionCard');
  const notice = document.getElementById('predictionNotice');

  const pct = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));

  const showFile = () => {
    if (fileInput.files?.[0]) {
      const f = fileInput.files[0];
      selected.textContent = `${f.name} • ${(f.size / 1024 / 1024).toFixed(2)} MB`;
    }
  };

  fileInput.addEventListener('change', showFile);
  ['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((name) => dropzone.addEventListener(name, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', (e) => {
    if (!e.dataTransfer.files.length) return;
    fileInput.files = e.dataTransfer.files;
    showFile();
  });

  function renderTop3(items = []) {
    const target = document.getElementById('resultTop3');
    target.innerHTML = items.map((item) => {
      const confidence = Math.max(0, Math.min(1, Number(item.confidence || 0)));
      return `
        <div class="prediction-probability-row">
          <div><span>${escapeHtml(item.label)}</span><strong>${pct(confidence)}</strong></div>
          <div class="prediction-probability-track"><span style="width:${(confidence * 100).toFixed(2)}%"></span></div>
        </div>`;
    }).join('');
  }

  function renderPrediction(data) {
    if (!data.prediction_enabled || !data.prediction) {
      predictionCard.hidden = true;
      notice.classList.remove('prediction-ready');
      notice.classList.add('prediction-warning');
      document.getElementById('predictionNoticeTitle').textContent = 'Prediction unavailable';
      document.getElementById('resultMessage').textContent = data.message || 'Python model is unavailable.';
      state.textContent = 'Inspected';
      return;
    }

    predictionCard.hidden = false;
    notice.classList.remove('prediction-warning');
    notice.classList.add('prediction-ready');

    document.getElementById('resultPrediction').textContent = data.prediction.label;
    document.getElementById('resultConfidence').textContent = pct(data.prediction.confidence);
    document.getElementById('resultModel').textContent = `Python model: ${String(data.prediction.model || 'selected').toUpperCase()}`;
    document.getElementById('resultSeverity').textContent = data.severity || '—';
    document.getElementById('resultStatus').textContent = data.status || 'Classified';
    document.getElementById('resultPredictionQuality').textContent = data.quality?.label || '—';
    document.getElementById('predictionNoticeTitle').textContent = data.active_alert ? 'Alert rule matched' : 'Classification complete';
    document.getElementById('resultMessage').textContent = data.decision_note || data.message || 'Classification complete.';

    state.textContent = data.status || 'Classified';
    state.className = `result-state status-${String(data.status || 'classified').toLowerCase().replace(/\s+/g, '-')}`;
    renderTop3(data.top3 || []);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files?.length) return;

    const button = form.querySelector('button[type="submit"]');
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span>Analyzing audio…</span><i class="bi bi-hourglass-split"></i>';
    state.textContent = 'Processing';

    try {
      const fd = new FormData();
      fd.append('audio', fileInput.files[0]);
      const response = await fetch('/api/audio/inspect', { method: 'POST', body: fd });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Audio analysis failed.');

      empty.hidden = true;
      content.hidden = false;
      document.getElementById('resultFilename').textContent = data.filename;
      document.getElementById('resultEventId').textContent = `#${data.event_id}`;
      document.getElementById('resultDuration').textContent = `${data.duration_seconds}s`;
      document.getElementById('resultSampleRate').textContent = `${data.sample_rate} Hz`;
      document.getElementById('resultQuality').textContent = data.quality.label;
      document.getElementById('resultRms').textContent = data.quality.rms;
      document.getElementById('resultSilence').textContent = pct(data.quality.silence_ratio);
      document.getElementById('resultClipping').textContent = pct(data.quality.clipping_ratio);
      renderPrediction(data);
    } catch (err) {
      state.textContent = 'Error';
      state.className = 'result-state';
      empty.hidden = false;
      content.hidden = true;
      empty.innerHTML = '<i class="bi bi-exclamation-triangle"></i><strong>Analysis failed</strong><p id="uploadErrorText"></p>';
      document.getElementById('uploadErrorText').textContent = err.message;
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  });
})();
