const form = document.getElementById('batchInspectForm');
const input = document.getElementById('batchFiles');
const results = document.getElementById('batchResults');
if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const files = [...input.files];
    if (!files.length) return;
    results.innerHTML = '';
    for (const file of files) {
      const row = document.createElement('div');
      row.className = 'batch-result-row';
      row.innerHTML = `<strong>${file.name}</strong><span>Processing…</span>`;
      results.appendChild(row);
      const data = new FormData(); data.append('audio', file);
      try {
        const response = await fetch('/api/audio/inspect', { method: 'POST', body: data });
        const json = await response.json();
        row.querySelector('span').textContent = response.ok ? `${json.quality.label} • ${json.sample_rate} Hz • ${json.duration_seconds}s` : (json.error || 'Failed');
      } catch (_) {
        row.querySelector('span').textContent = 'Network error';
      }
    }
  });
}
