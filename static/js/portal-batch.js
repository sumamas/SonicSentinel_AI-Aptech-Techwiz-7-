(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const input = $('batchFiles'), drop = $('batchDrop'), start = $('batchStart'), stop = $('batchStop'), clear = $('batchClear');
  const rows = $('batchRows'), state = $('batchState'), progress = $('batchProgress');
  const ALLOWED = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
  let queue = [], busy = false, cancel = false;
  const stats = { total: 0, done: 0, alert: 0, review: 0, agree: 0, fail: 0 };

  const pct = (v) => (v == null ? '—' : (Number(v) * 100).toFixed(1) + '%');
  const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const agreeClass = (s) => ({'Acceptable Match':'match','Weak Match':'weak','Model Disagreement':'disagree','Uncertain Result':'uncertain'}[s] || 'uncertain');

  function paintStats() {
    $('sTotal').textContent = stats.total; $('sDone').textContent = stats.done; $('sAlert').textContent = stats.alert;
    $('sReview').textContent = stats.review; $('sAgree').textContent = stats.agree; $('sFail').textContent = stats.fail;
  }
  function resetStats() { Object.keys(stats).forEach((k) => (stats[k] = 0)); stats.total = queue.length; paintStats(); }

  function addFiles(list) {
    if (busy) return;
    for (const f of list) {
      const ext = (f.name.split('.').pop() || '').toLowerCase();
      if (queue.some((q) => q.file.name === f.name && q.file.size === f.size)) continue;
      queue.push({ file: f, ok: ALLOWED.includes(ext), status: 'queued' });
    }
    draw();
  }

  function draw() {
    rows.replaceChildren();
    if (!queue.length) {
      rows.innerHTML = '<tr><td colspan="7"><div class="m-empty"><i class="bi bi-inbox"></i>Add recordings to build your batch.</div></td></tr>';
    }
    queue.forEach((q, i) => {
      const tr = document.createElement('tr'); tr.id = 'row' + i;
      tr.innerHTML = `<td class="file"><b>${esc(q.file.name)}</b><small>${(q.file.size / 1048576).toFixed(2)} MB</small></td>
        <td>${q.ok ? 'Queued' : '<span class="pill review">Unsupported type</span>'}</td><td>—</td><td>—</td><td>—</td><td>—</td><td></td>`;
      rows.append(tr);
    });
    const valid = queue.filter((q) => q.ok).length;
    state.textContent = queue.length ? `${queue.length} file(s) · ${valid} ready` : 'No files selected.';
    start.disabled = !valid || busy; clear.disabled = !queue.length || busy;
    resetStats();
  }

  input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
  ['dragenter', 'dragover'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
  clear.addEventListener('click', () => { queue = []; draw(); progress.style.width = '0%'; });
  stop.addEventListener('click', () => { cancel = true; stop.disabled = true; state.textContent = 'Stopping after the current file…'; });

  function fill(tr, d) {
    const f = d.final || {}, c = d.comparison || {}, py = d.python || {}, tm = d.gtm || {};
    const cells = tr.children;
    cells[1].innerHTML = '<span class="pill match"><span class="d"></span>Complete</span>';
    cells[2].innerHTML = `<div class="duo">
      <span><span class="tag py">PY</span><b>${esc(py.prediction?.label || 'Unavailable')}</b><em>${pct(py.prediction?.confidence)}</em></span>
      <span><span class="tag tm">TM</span><b>${esc(tm.prediction?.label || 'Unavailable')}</b><em>${pct(tm.prediction?.confidence)}</em></span></div>`;
    cells[3].innerHTML = `<span class="pill ${agreeClass(c.status)}">${esc(c.status || '—')}</span>` + (c.confidence_difference != null ? `<small>Δ ${pct(c.confidence_difference)}</small>` : '');
    cells[4].innerHTML = `<b>${esc(f.display_label || '—')}</b><small>${pct(f.confidence)} · <span class="pill ${esc(String(f.severity || '').toLowerCase())}" style="padding:1px 7px">${esc(f.severity || '—')}</span></small>`;
    const review = f.manual_review_required;
    cells[5].innerHTML = `<span class="pill ${review ? 'review' : (f.active_alert ? 'alert' : 'weak')}">${esc(f.status || '—')}</span><small>${esc(d.quality?.label)} quality${d.duplicate_of ? ' · duplicate of #' + esc(d.duplicate_of) : ''}</small>`;
    cells[6].innerHTML = `<a class="m-btn ghost sm" href="${esc(d.detail_url)}">Open #${esc(d.event_id)}</a>`;
    stats.done++; if (f.active_alert) stats.alert++; if (review) stats.review++;
    if (c.status === 'Acceptable Match' || c.status === 'Weak Match') stats.agree++;
  }

  start.addEventListener('click', async () => {
    if (busy) return;
    const work = queue.map((q, i) => ({ q, i })).filter((x) => x.q.ok && x.q.status !== 'done');
    if (!work.length) return;
    busy = true; cancel = false; start.disabled = true; clear.disabled = true; stop.disabled = false; input.disabled = true;
    resetStats(); let processed = 0;
    for (const { q, i } of work) {
      if (cancel) break;
      const tr = $('row' + i), cells = tr.children;
      cells[1].innerHTML = '<span class="pill weak"><i class="bi bi-arrow-repeat"></i> Analyzing…</span>';
      state.textContent = `Analyzing ${processed + 1} of ${work.length}: ${q.file.name}`;
      try {
        const body = new FormData(); body.append('audio', q.file); body.append('source', 'batch');
        const r = await fetch('/api/audio/inspect', { method: 'POST', body, credentials: 'same-origin' });
        const data = await r.json().catch(() => ({ error: r.status === 413 ? 'File is larger than the upload limit.' : 'Server response unreadable; sign in again if your session expired.' }));
        if (!r.ok) throw new Error(data.error || 'Analysis failed.');
        fill(tr, data); q.status = 'done';
      } catch (err) {
        stats.fail++; q.status = 'failed';
        cells[1].innerHTML = '<span class="pill alert">Failed</span>';
        cells[2].colSpan = 1; cells[5].innerHTML = `<small style="color:#8f2a34">${esc(err.message)}</small>`;
      }
      processed++; paintStats();
      progress.style.width = (100 * processed / work.length) + '%';
    }
    work.slice(processed).forEach(({ i }) => { const c = $('row' + i).children; c[1].textContent = 'Not processed'; });
    state.textContent = `${cancel ? 'Stopped' : 'Finished'} · ${stats.done} completed · ${stats.fail} failed · ${work.length - processed} not processed`;
    busy = false; start.disabled = false; clear.disabled = false; stop.disabled = true; input.disabled = false;
  });
})();
