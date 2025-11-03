(function () {
  // Bootstrap modal helpers
  let analyzeModalEl = document.getElementById('analyzeModal');
  let analyzeModal = analyzeModalEl ? new bootstrap.Modal(analyzeModalEl) : null;
  const analyzeBtn = document.getElementById('analyzeBtn');
  const openAnalyze = document.getElementById('openAnalyze');
  const uploadInput = document.getElementById('imageUpload');
  const previewWrap = document.getElementById('previewWrap');
  const imgPrev = document.getElementById('imagePreview');
  const predResult = document.getElementById('predResult');

  window.NSAI_openAnalyze = () => analyzeModal && analyzeModal.show();
  analyzeBtn && analyzeBtn.addEventListener('click', () => window.NSAI_openAnalyze());
  openAnalyze && openAnalyze.addEventListener('click', (e) => { e.preventDefault(); window.NSAI_openAnalyze(); });

  if (uploadInput) {
    uploadInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        imgPrev.src = ev.target.result;
        previewWrap.style.display = 'block';
      };
      reader.readAsDataURL(file);
      predResult.textContent = '';
    });
  }

  const form = document.getElementById('upload-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!uploadInput.files[0]) return;

      const fd = new FormData();
      fd.append('file', uploadInput.files[0]);

      predResult.textContent = 'Running prediction...';

      const res = await fetch('/predict', { method: 'POST', body: fd });
      const json = await res.json();

      if (json.error) {
        predResult.textContent = 'Error: ' + json.error;
        return;
      }

      predResult.innerHTML = `<b>${json.label}</b> (Confidence: ${json.confidence}%)`;

      // Refresh dashboard KPIs + list + donut via /api/stats
      refreshStats && refreshStats();
    });
  }

  async function refreshStats() {
    const res = await fetch('/api/stats');
    const s = await res.json();

    // KPIs
    document.getElementById('kpi-total') && (document.getElementById('kpi-total').textContent = s.total);
    document.getElementById('kpi-tumor') && (document.getElementById('kpi-tumor').textContent = s.tumors);
    document.getElementById('kpi-non') && (document.getElementById('kpi-non').textContent = s.non_tumors);
    document.getElementById('kpi-avg') && (document.getElementById('kpi-avg').textContent = (s.avg_confidence * 100).toFixed(1) + '%');

    // Recent List
    const list = document.getElementById('recent-list');
    if (list) {
      list.innerHTML = '';
      if (s.recent.length === 0) {
        list.innerHTML = '<li class="list-group-item text-muted">No scans yet.</li>';
      } else {
        s.recent.forEach(r => {
          const li = document.createElement('li');
          li.className = 'list-group-item d-flex align-items-center';
          li.innerHTML = `
            <img src="/static/uploads/${r.filename}" class="rounded me-3" style="width:56px;height:56px;object-fit:cover;">
            <div class="flex-grow-1">
              <div class="fw-semibold">${r.filename}</div>
              <div class="text-muted small">${r.created_at}</div>
            </div>
            <span class="badge ${r.label === 'Tumor' ? 'bg-danger' : 'bg-success'} me-2">${r.label}</span>
            <span class="text-muted small">${(r.confidence * 100).toFixed(1)}%</span>
          `;
          list.appendChild(li);
        });
      }
    }

    // Update donut (Chart instance defined in dashboard.html)
    if (window.renderDonut) window.renderDonut(s.tumors, s.non_tumors);
  }

  window.refreshStats = refreshStats;
})();
