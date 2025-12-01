(function () {
  // Modal helpers 
  let analyzeModal;
  function ensureModal() {
    if (!analyzeModal) {
      const el = document.getElementById('analyzeModal');
      if (!el) return null;
      analyzeModal = (window.bootstrap && window.bootstrap.Modal)
        ? new bootstrap.Modal(el)
        : null;
    }
    return analyzeModal;
  }

  // Open modal
  const btnAnalyze = document.getElementById('analyzeBtn');
  btnAnalyze?.addEventListener('click', () => {
    const m = ensureModal();
    if (m) m.show();
    else document.getElementById('analyzeModal')?.classList.add('show');
  });

  // Preview
  const uploadInput = document.getElementById('imageUpload');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg  = document.getElementById('imagePreview');
  const predResult  = document.getElementById('predResult');
  const form        = document.getElementById('upload-form');

  uploadInput?.addEventListener('change', function () {
    predResult.textContent = '';
    if (this.files && this.files[0]) {
      const reader = new FileReader();
      reader.onload = e => {
        previewImg.src = e.target.result;
        previewWrap.style.display = 'block';
      };
      reader.readAsDataURL(this.files[0]);
    } else {
      previewWrap.style.display = 'none';
      previewImg.src = '';
    }
  });

  // Submit 
  form?.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!uploadInput.files.length) {
      predResult.innerHTML = `<div class="alert alert-warning mb-0">Please choose an image.</div>`;
      return;
    }

    const data = new FormData(form);
    predResult.innerHTML = 'Analyzing…';

    fetch('/predict', { method: 'POST', body: data })
      .then(r => r.json().then(j => ({ ok: r.ok, body: j })))
      .then(({ ok, body }) => {
        if (!ok) throw new Error(body.detail || 'Error');
        const pct = Math.round((body.tumor_probability || 0) * 100);
        predResult.innerHTML = body.has_tumor
          ? `<div class="alert alert-danger mb-0"><b>YES (Tumor)</b> — Confidence: ${pct}%</div>`
          : `<div class="alert alert-success mb-0"><b>NO (Normal)</b> — Confidence: ${pct}%</div>`;

        refreshDashboard();      // KPIs, donut, recent list
        refreshSideConsole();    // side console KPIs + latest patient
      })
      .catch(err => {
        predResult.innerHTML = `<div class="alert alert-danger mb-0">${err.message}</div>`;
      });
  });

  // Recent click-to-reopen 
  function bindRecentClicks() {
    document.querySelectorAll('.recent-item').forEach(li => {
      li.addEventListener('click', () => {
        const id = li.getAttribute('data-id');
        if (!id) return;
        fetch(`/case/${id}`)
          .then(r => r.json().then(j => ({ ok: r.ok, body: j })))
          .then(({ ok, body }) => {
            if (!ok) throw new Error(body.detail || 'Error');

            const m = ensureModal();
            if (!m) return;

            const imgUrl = '/' + (body.image_path || '');
            const yes    = !!body.result?.has_tumor;
            const pct    = Math.round((Number(body.result?.prob_yes || 0)) * 100);

            previewWrap.style.display = 'block';
            previewImg.src = imgUrl;

            const p = body.patient || {};
            const pd = `
              <div class="mt-2">
                <div class="small text-muted">Patient:</div>
                <div class="small"><b>ID:</b> ${p.patient_id || '—'} &nbsp; <b>Name:</b> ${p.name || '—'} &nbsp; <b>Age:</b> ${p.age || '—'} &nbsp; <b>Sex:</b> ${p.sex || '—'}</div>
                <div class="small text-muted">Case ID: ${body.id || '—'} • UTC: ${body.timestamp_utc || '—'}</div>
              </div>
            `;

            predResult.innerHTML = yes
              ? `<div class="alert alert-danger mb-2"><b>YES (Tumor)</b> — Confidence: ${pct}%</div>${pd}`
              : `<div class="alert alert-success mb-2"><b>NO (Normal)</b> — Confidence: ${pct}%</div>${pd}`;

            m.show();
          })
          .catch(err => {
            alert(err.message || 'Failed to open case');
          });
      });
    });
  }

  // Dashboard refresher
  function refreshDashboard() {
    fetch('/recent')
      .then(r => r.json())
      .then(list => {
        // KPIs
        const total = list.length;
        const tumors = list.filter(x => x.result?.has_tumor).length;
        const non   = total - tumors;
        const confs = list.map(x => Number(x.result?.prob_yes || 0));
        const avg   = confs.length ? (confs.reduce((a,b)=>a+b,0)/confs.length) : 0;

        const elTotal = document.getElementById('kpi-total');
        const elTumor = document.getElementById('kpi-tumor');
        const elNon   = document.getElementById('kpi-non');
        const elAvg   = document.getElementById('kpi-avg');

        if (elTotal) elTotal.textContent = total;
        if (elTumor) elTumor.textContent = tumors;
        if (elNon)   elNon.textContent   = non;
        if (elAvg)   elAvg.textContent   = `${(avg*100).toFixed(1)}%`;

        // Donut
        if (window.renderDonut) window.renderDonut(tumors, non);

        // Recent list
        const listEl = document.getElementById('recent-list');
        if (!listEl) return;

        listEl.innerHTML = '';
        const slice = list.slice(0, 8);
        if (!slice.length) {
          listEl.innerHTML = '<li class="list-group-item text-muted">No scans yet.</li>';
        } else {
          slice.forEach(r => {
            const fp = r.image_path || '';
            const fname = fp.split('/').pop();
            const label = r.result?.has_tumor ? 'Tumor' : 'No Tumor';
            const badge = r.result?.has_tumor ? 'bg-danger' : 'bg-success';
            const pct   = Math.round((Number(r.result?.prob_yes || 0))*100);

            const li = document.createElement('li');
            li.className = 'list-group-item d-flex align-items-center recent-item';
            li.setAttribute('data-id', r.id);
            li.innerHTML = `
              <img src="/${fp}" class="rounded me-3" style="width:56px;height:56px;object-fit:cover;">
              <div class="flex-grow-1">
                <div class="fw-semibold">${fname}</div>
                <div class="text-muted small">${r.timestamp_utc || ''}</div>
              </div>
              <span class="badge ${badge} me-2">${label}</span>
              <span class="text-muted small">${pct}%</span>
            `;
            listEl.appendChild(li);
          });
        }
        bindRecentClicks();
      })
      .catch(()=>{});
  }

  // Side Console (
  function refreshSideConsole() {
    fetch('/stats')
      .then(r => r.json())
      .then(data => {
        const k = data.kpis || {};
        const latest = data.latest_case || null;

        const scTotal = document.getElementById('sc-total');
        const scAvg   = document.getElementById('sc-avg');
        const scTumor = document.getElementById('sc-tumor');
        const scNon   = document.getElementById('sc-non');
        const scPat   = document.getElementById('sc-patient');

        if (scTotal) scTotal.textContent = Number(k.total || 0);
        if (scAvg)   scAvg.textContent   = `${Math.round((Number(k.avg_confidence || 0))*100)}%`;
        if (scTumor) scTumor.textContent = Number(k.tumors || 0);
        if (scNon)   scNon.textContent   = Number(k.non_tumors || 0);

        if (scPat) {
          if (!latest) {
            scPat.innerHTML = `<div class="small">No cases yet.</div>`;
          } else {
            const p = latest.patient || {};
            const yes = !!latest.result?.has_tumor;
            const pct = Math.round((Number(latest.result?.prob_yes || 0)) * 100);
            const badge = yes ? 'danger' : 'success';
            const label = yes ? 'YES (Tumor)' : 'NO (Normal)';
            const img  = latest.image_path ? `/<span></span>${latest.image_path}`.replace('/<span></span>', '') : '';

            scPat.innerHTML = `
              <div class="d-flex align-items-start gap-2">
                ${img ? `<img src="/${latest.image_path}" class="rounded" style="width:72px;height:72px;object-fit:cover;">` : ''}
                <div class="flex-grow-1">
                  <div class="small text-muted">Patient</div>
                  <div class="small"><b>ID:</b> ${p.patient_id || '—'} &nbsp; <b>Name:</b> ${p.name || '—'}</div>
                  <div class="small"><b>Age:</b> ${p.age || '—'} &nbsp; <b>Sex:</b> ${p.sex || '—'}</div>
                  <div class="small text-muted">Case: ${latest.id || '—'} • UTC: ${latest.timestamp_utc || '—'}</div>
                  <div class="mt-1"><span class="badge bg-${badge}">${label}</span> <span class="small text-muted ms-1">${pct}%</span></div>
                </div>
              </div>
            `;
          }
        }
      })
      .catch(()=>{});
  }

  // Update side console when it opens
  const sideConsoleEl = document.getElementById('sideConsole');
  sideConsoleEl?.addEventListener('shown.bs.offcanvas', refreshSideConsole);

  // Initial load
  refreshDashboard();
  refreshSideConsole();
  bindRecentClicks();
})();
