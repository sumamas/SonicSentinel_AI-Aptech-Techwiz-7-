(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const form = $('uploadForm'), input = $('audioFile'), drop = $('dropZone'), audio = $('uploadPreview');
  const state = $('uploadState'), error = $('uploadError'), button = $('analyzeButton'), steps = $('steps');
  const ALLOWED = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
  let file = null, preview = null, stepTimer = null;

  const pct = (v) => (v == null ? '—' : (Number(v) * 100).toFixed(1) + '%');
  const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const agreeClass = (s) => ({'Acceptable Match':'match','Weak Match':'weak','Model Disagreement':'disagree','Uncertain Result':'uncertain'}[s] || 'uncertain');

  function showError(msg) { error.textContent = msg; error.hidden = false; }

  function setFile(f) {
    error.hidden = true;
    if (!f) {
      file = null;
      $('fileCard').hidden = true;
      $('audioPlayer').hidden = true;
      button.disabled = true;
      state.textContent = 'Choose a file to begin.';
      updateStepsList(0);
      return;
    }
    const ext = (f.name.split('.').pop() || '').toLowerCase();
    if (!ALLOWED.includes(ext)) { setFile(null); showError('This file type is not supported. Use WAV, MP3, FLAC, OGG or M4A.'); return; }
    file = f;
    $('fileName').textContent = f.name;
    $('fileInfo').textContent = (f.size / 1048576).toFixed(2) + ' MB · ' + ext.toUpperCase();
    $('fileCard').hidden = false;
    if (preview) URL.revokeObjectURL(preview);
    preview = URL.createObjectURL(f);
    audio.src = preview;
    $('audioPlayer').hidden = false;
    button.disabled = false;
    state.textContent = 'Ready to analyze.';
    updateStepsList(1);
    drawPlaceholderWaveform();
  }

  input.addEventListener('change', () => setFile(input.files[0]));
  $('clearFile').addEventListener('click', () => { input.value = ''; setFile(null); });
  $('clearBtn').addEventListener('click', () => { input.value = ''; setFile(null); });
  ['dragenter', 'dragover'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

  function setStep(n) {
    steps.querySelectorAll('li').forEach((li) => {
      const i = Number(li.dataset.step);
      li.className = i < n ? 'done' : (i === n ? 'active' : '');
      li.querySelector('i').className = i < n ? 'bi bi-check2-circle' : (i === n ? 'bi bi-arrow-repeat' : 'bi bi-circle');
    });
    updateStepsList(n);
  }

  function updateStepsList(n) {
    const items = $('stepsList')?.querySelectorAll('li');
    if (!items) return;
    items.forEach((li, i) => {
      li.className = i < n ? 'done' : (i === n ? 'active' : '');
    });
  }

  function bars(top3, model) {
    return (top3 || []).map((r) => `
      <div class="model-bar-row">
        <span class="bar-label">${esc(r.label)}</span>
        <span class="bar-pct">${pct(r.confidence)}</span>
        <div class="bar-track"><span class="bar-fill ${model}" style="width:${Math.max(0, Math.min(100, r.confidence * 100)).toFixed(1)}%"></span></div>
      </div>`).join('');
  }

  function render(d) {
    const f = d.final || {}, c = d.comparison || {}, py = d.python || {}, tm = d.gtm || {};
    const review = f.manual_review_required;

    // Verdict section
    $('verdict').className = 'verdict-card ' + (review ? 'review ' : '') + 'sev-' + String(f.severity || '').toLowerCase();
    $('rVerdictLabel').textContent = f.display_label || 'Not classified';
    $('rVerdictConf').textContent = pct(f.confidence);
    $('rConfLevel').textContent = (f.confidence_level || '—') + ' confidence · ' + (f.method || '');
    $('rPills').innerHTML = [
      `<span class="pill ${esc(String(f.severity || '').toLowerCase())}">${esc(f.severity || '—')} severity</span>`,
      `<span class="pill ${review ? 'review' : (f.active_alert ? 'alert' : '')}">${esc(f.status || '—')}</span>`,
      `<span class="pill ${esc(String(d.quality?.label || '').toLowerCase())}">${esc(d.quality?.label)} quality</span>`,
      d.duplicate_of ? `<span class="pill uncertain">Duplicate of #${esc(d.duplicate_of)}</span>` : ''
    ].join('');
    $('rNote').textContent = f.decision_note || '';
    $('rOpen').href = d.detail_url;

    // Right sidebar — Current Analysis Result
    $('rLabel').textContent = f.display_label || 'Not classified';
    $('rConf').textContent = pct(f.confidence);
    $('rConfBar').style.width = Math.max(0, Math.min(100, (f.confidence || 0) * 100)) + '%';
    $('rSeverity').textContent = f.severity || '—';
    $('rSeverity').className = 'meta-pill ' + String(f.severity || '').toLowerCase();
    $('rQuality').textContent = d.quality?.label || '—';
    $('rRms').textContent = d.quality?.rms || '—';
    $('rStatus').textContent = f.status || '—';

    // Model comparison
    $('rAgree').textContent = c.status || '—';
    $('rCompare').textContent = c.confidence_difference != null ? `Top-class confidence difference ${pct(c.confidence_difference)}` : 'Comparison needs both models.';

    // Python model
    if (py.status === 'ready') {
      $('rPy').textContent = py.prediction.label;
      $('rPyConf').textContent = pct(py.prediction.confidence);
      $('rPyBars').innerHTML = bars(py.top3, 'python');
    } else {
      $('rPy').textContent = 'Unavailable';
      $('rPyConf').textContent = '—';
      $('rPyBars').innerHTML = '';
    }

    // GTM model
    if (tm.status === 'ready') {
      $('rTm').textContent = tm.prediction.label;
      $('rTmConf').textContent = pct(tm.prediction.confidence);
      $('rTmBars').innerHTML = bars(tm.top3, 'gtm');
    } else {
      $('rTm').textContent = 'Unavailable';
      $('rTmConf').textContent = '—';
      $('rTmBars').innerHTML = '';
    }

    // Agreement details
    if (c.confidence_difference != null) {
      $('rConfDiff').textContent = pct(c.confidence_difference);
      $('rConfDiffSub').textContent = `(${pct(py.prediction?.confidence)} vs ${pct(tm.prediction?.confidence)})`;
    } else {
      $('rConfDiff').textContent = '—';
      $('rConfDiffSub').textContent = '';
    }
    $('rFinalDecision').textContent = f.display_label || '—';
    const agreeScore = c.agreement_score != null ? Math.round(c.agreement_score * 100) : null;
    $('rAgreementScore').textContent = agreeScore != null ? (agreeScore >= 80 ? 'High' : agreeScore >= 50 ? 'Medium' : 'Low') : '—';
    $('rAgreementBar').style.width = (agreeScore || 0) + '%';

    // Show results
    $('result').hidden = false;
    $('verdictSection').hidden = false;
    $('result').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });

    // Add to batch results table
    addBatchRow(d);

    // Draw waveform if audio data available
    if (d.waveform_data) {
      drawWaveform(d.waveform_data);
    }
  }

  function addBatchRow(d) {
    const tbody = $('batchResultsBody');
    if (!tbody) return;
    // Remove empty state if present
    const emptyRow = tbody.querySelector('td[colspan]');
    if (emptyRow) emptyRow.parentElement.remove();

    const f = d.final || {};
    const row = document.createElement('tr');
    const rowNum = tbody.children.length + 1;
    const classLabel = f.display_label || 'Unknown';
    const classClass = classLabel.toLowerCase().includes('siren') || classLabel.toLowerCase().includes('alarm') ? 'alarm'
      : classLabel.toLowerCase().includes('glass') ? 'glass'
      : classLabel.toLowerCase().includes('scream') || classLabel.toLowerCase().includes('panic') ? 'panic'
      : classLabel.toLowerCase().includes('help') ? 'help' : 'noise';

    row.innerHTML = `
      <td class="num">${rowNum}</td>
      <td class="file-name">${esc(d.filename)}</td>
      <td><span class="pill pill-${classClass}">${esc(classLabel)}</span></td>
      <td>${pct(f.confidence)}</td>
      <td><span class="pill pill-${esc(String(d.quality?.label || '').toLowerCase())}">${esc(d.quality?.label || '—')}</span></td>
      <td><span class="pill pill-completed">${esc(f.status || 'Completed')}</span></td>
      <td>${new Date().toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' })} ${new Date().toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit' })}</td>
      <td><a href="${d.detail_url}" class="view-btn"><i class="bi bi-eye"></i> View Details</a></td>
    `;
    tbody.insertBefore(row, tbody.firstChild);
  }

  // Waveform / Spectrogram drawing
  function drawPlaceholderWaveform() {
    const canvas = $('waveformCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Upload audio to see waveform', w / 2, h / 2 - 10);
  }

  function drawWaveform(data) {
    const canvas = $('waveformCanvas');
    if (!canvas || !data) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);

    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#2563eb');
    gradient.addColorStop(1, '#3b82f6');
    ctx.fillStyle = gradient;

    const barWidth = w / data.length;
    data.forEach((val, i) => {
      const barHeight = val * h * 0.9;
      ctx.fillRect(i * barWidth, (h - barHeight) / 2, barWidth * 0.8, barHeight);
    });
  }

  function drawSpectrogram(data) {
    const canvas = $('spectrogramCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);

    // Placeholder spectrogram with gradient
    const gradient = ctx.createLinearGradient(0, h, w, 0);
    gradient.addColorStop(0, '#1e1b4b');
    gradient.addColorStop(0.3, '#7c3aed');
    gradient.addColorStop(0.5, '#f59e0b');
    gradient.addColorStop(0.7, '#ef4444');
    gradient.addColorStop(1, '#fbbf24');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    // Add some visual texture
    ctx.globalAlpha = 0.3;
    for (let i = 0; i < 50; i++) {
      ctx.fillStyle = Math.random() > 0.5 ? '#fbbf24' : '#7c3aed';
      ctx.fillRect(Math.random() * w, Math.random() * h, Math.random() * 80, Math.random() * 4);
    }
    ctx.globalAlpha = 1;
  }

  // Toggle waveform/spectrogram
  $('viewWaveform')?.addEventListener('click', () => {
    $('viewWaveform').classList.add('active');
    $('viewSpectrogram').classList.remove('active');
    $('waveformContainer').hidden = false;
    $('spectrogramContainer').hidden = true;
  });
  $('viewSpectrogram')?.addEventListener('click', () => {
    $('viewSpectrogram').classList.add('active');
    $('viewWaveform').classList.remove('active');
    $('waveformContainer').hidden = true;
    $('spectrogramContainer').hidden = false;
    drawSpectrogram();
  });

  $('rAgain').addEventListener('click', () => {
    $('result').hidden = true;
    $('verdictSection').hidden = true;
    input.value = '';
    setFile(null);
    steps.hidden = true;
    window.scrollTo({ top: 0 });
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!file) { showError('Choose an audio file first.'); return; }
    button.disabled = true; input.disabled = true; error.hidden = true;
    $('result').hidden = true;
    $('verdictSection').hidden = true;
    steps.hidden = false; let n = 0; setStep(0);
    stepTimer = setInterval(() => { if (n < 2) setStep(++n); }, 900);
    state.textContent = 'Analyzing… the first run loads the models and can take a few seconds.';
    try {
      const body = new FormData(); body.append('audio', file); body.append('source', 'upload');
      const r = await fetch('/api/audio/inspect', { method: 'POST', body, credentials: 'same-origin' });
      const data = await r.json().catch(() => ({ error: r.status === 413 ? 'The file is larger than the upload limit.' : 'The server response could not be read. Sign in again if your session expired.' }));
      if (!r.ok) throw new Error(data.error || 'Analysis failed.');
      clearInterval(stepTimer); setStep(4);
      state.textContent = 'Saved as Audio #' + data.event_id + '.';
      render(data);
    } catch (err) {
      clearInterval(stepTimer); steps.hidden = true;
      showError(err.message); state.textContent = 'Analysis was not completed.';
    } finally {
      button.disabled = !file; input.disabled = false;
    }
  });

  // Initialize
  drawPlaceholderWaveform();
})();
