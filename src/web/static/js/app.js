/**
 * RoadEye YOLO11 - Local Host Road Damage Detection SPA
 * Vanilla JavaScript Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // State Management
  // =========================================================================
  const state = {
    selectedVideoFile: null,
    isSampleVideo: false,
    selectedImageFile: null,
    isSampleImage: false,
    selectedSampleName: null,
    activeVideoJobId: null,
    videoPollInterval: null,
    webcamStream: null,
    webcamInterval: null,
    webcamRunning: false,
  };

  // =========================================================================
  // DOM Elements
  // =========================================================================
  // Header & Status
  const statusLabelText = document.getElementById('status-label-text');
  const weightsNameDisplay = document.getElementById('weights-name-display');
  const deviceNameDisplay = document.getElementById('device-name-display');

  // Navigation
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanels = document.querySelectorAll('.tab-panel');

  // Controls
  const confSlider = document.getElementById('conf-slider');
  const confValBadge = document.getElementById('conf-val-badge');
  const skipSelect = document.getElementById('skip-select');
  const toggleHud = document.getElementById('toggle-hud');
  const classFilters = {
    D00: document.getElementById('filter-d00'),
    D10: document.getElementById('filter-d10'),
    D20: document.getElementById('filter-d20'),
    D40: document.getElementById('filter-d40'),
  };

  // Video Tab Elements
  const videoDropzone = document.getElementById('video-dropzone');
  const videoFileInput = document.getElementById('video-file-input');
  const btnBrowseVideo = document.getElementById('btn-browse-video');
  const btnSampleVideo = document.getElementById('btn-sample-video');
  const videoSelectedFileBar = document.getElementById('video-selected-file-bar');
  const videoSelectedName = document.getElementById('video-selected-name');
  const videoSelectedSize = document.getElementById('video-selected-size');
  const btnClearVideo = document.getElementById('btn-clear-video');
  const btnStartVideoDetection = document.getElementById('btn-start-video-detection');

  // Video Processing Card
  const videoProcessingCard = document.getElementById('video-processing-card');
  const videoProgressFill = document.getElementById('video-progress-fill');
  const videoProgressPct = document.getElementById('video-progress-pct');
  const videoProgressFrames = document.getElementById('video-progress-frames');
  const videoProgressFps = document.getElementById('video-progress-fps');
  const videoLivePreviewImg = document.getElementById('video-live-preview-img');
  const btnCancelVideo = document.getElementById('btn-cancel-video');
  const liveCountD00 = document.getElementById('live-count-d00');
  const liveCountD10 = document.getElementById('live-count-d10');
  const liveCountD20 = document.getElementById('live-count-d20');
  const liveCountD40 = document.getElementById('live-count-d40');

  // Video Results Card
  const videoResultsCard = document.getElementById('video-results-card');
  const resultsVideoPlayer = document.getElementById('results-video-player');
  const resultsVideoSource = document.getElementById('results-video-source');
  const btnDownloadVideoMp4 = document.getElementById('btn-download-video-mp4');
  const btnDownloadVideoCsv = document.getElementById('btn-download-video-csv');
  const videoPciScore = document.getElementById('video-pci-score');
  const videoPciBadge = document.getElementById('video-pci-badge');
  const videoPciSummary = document.getElementById('video-pci-summary');
  const resD00Count = document.getElementById('res-d00-count');
  const resD10Count = document.getElementById('res-d10-count');
  const resD20Count = document.getElementById('res-d20-count');
  const resD40Count = document.getElementById('res-d40-count');
  const resTotalDet = document.getElementById('res-total-det');
  const resTimeDet = document.getElementById('res-time-det');

  // Image Tab Elements
  const imageDropzone = document.getElementById('image-dropzone');
  const imageFileInput = document.getElementById('image-file-input');
  const btnBrowseImage = document.getElementById('btn-browse-image');
  const btnSampleImage = document.getElementById('btn-sample-image');
  const imageSelectedFileBar = document.getElementById('image-selected-file-bar');
  const imageSelectedName = document.getElementById('image-selected-name');
  const btnClearImage = document.getElementById('btn-clear-image');
  const btnStartImageDetection = document.getElementById('btn-start-image-detection');
  const imageResultsCard = document.getElementById('image-results-card');
  const imgResultOriginal = document.getElementById('img-result-original');
  const imgResultAnnotated = document.getElementById('img-result-annotated');
  const imageInferenceTimeSub = document.getElementById('image-inference-time-sub');
  const btnDownloadImageResult = document.getElementById('btn-download-image-result');
  const imageDetectionsTbody = document.getElementById('image-detections-tbody');

  // Webcam Tab Elements
  const btnStartWebcam = document.getElementById('btn-start-webcam');
  const btnStopWebcam = document.getElementById('btn-stop-webcam');
  const webcamVideoHidden = document.getElementById('webcam-video-hidden');
  const webcamCanvasHidden = document.getElementById('webcam-canvas-hidden');
  const webcamAnnotatedStream = document.getElementById('webcam-annotated-stream');
  const webcamPlaceholderBox = document.getElementById('webcam-placeholder-box');
  const webcamFpsVal = document.getElementById('webcam-fps-val');
  const webcamLatencyVal = document.getElementById('webcam-latency-val');
  const webcamCountVal = document.getElementById('webcam-count-val');

  // =========================================================================
  // Initial Status Check
  // =========================================================================
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        weightsNameDisplay.textContent = data.model_filename || 'best.pt';
        deviceNameDisplay.textContent = data.cuda_available ? `GPU: ${data.gpu_name}` : 'CPU Acceleration';
        statusLabelText.textContent = 'Model Online';
      }
    } catch (err) {
      console.warn('Status fetch error:', err);
      statusLabelText.textContent = 'Local Server Ready';
    }
  }
  fetchSystemStatus();

  // =========================================================================
  // Navigation Tabs Logic
  // =========================================================================
  navTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      navTabs.forEach((t) => t.classList.remove('active'));
      tabPanels.forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add('active');

      // If switching to database, load records
      if (targetId === 'database-panel') {
        loadDatabaseHistory();
      }

      // If switching away from webcam, pause stream
      if (targetId !== 'webcam-panel' && state.webcamRunning) {
        stopWebcamStream();
      }
    });
  });

  // Confidence Slider Update
  confSlider.addEventListener('input', () => {
    confValBadge.textContent = parseFloat(confSlider.value).toFixed(2);
  });

  function getSelectedClasses() {
    const selected = [];
    Object.entries(classFilters).forEach(([cls, elem]) => {
      if (elem && elem.checked) selected.push(cls);
    });
    return selected.join(',');
  }

  // =========================================================================
  // Video Handling Logic
  // =========================================================================
  btnBrowseVideo.addEventListener('click', (e) => {
    e.stopPropagation();
    videoFileInput.click();
  });

  videoDropzone.addEventListener('click', () => {
    if (!state.selectedVideoFile && !state.isSampleVideo) {
      videoFileInput.click();
    }
  });

  videoDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    videoDropzone.classList.add('drag-over');
  });

  videoDropzone.addEventListener('dragleave', () => {
    videoDropzone.classList.remove('drag-over');
  });

  videoDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    videoDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVideoSelection(e.dataTransfer.files[0], false);
    }
  });

  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleVideoSelection(e.target.files[0], false);
    }
  });

  btnSampleVideo.addEventListener('click', (e) => {
    e.stopPropagation();
    selectSampleVideoCard('sample_road.mp4');
  });

  btnClearVideo.addEventListener('click', (e) => {
    e.stopPropagation();
    clearVideoSelection();
  });

  // Direct Sample Video Cards
  const sampleVideoCards = document.querySelectorAll('.sample-video-card');
  sampleVideoCards.forEach((card) => {
    card.addEventListener('click', (e) => {
      e.stopPropagation();
      const sampleVideo = card.dataset.sampleVideo;
      selectSampleVideoCard(sampleVideo);
    });
  });

  function selectSampleVideoCard(sampleVideo) {
    state.isSampleVideo = true;
    state.selectedSampleVideoName = sampleVideo;
    state.selectedVideoFile = null;

    sampleVideoCards.forEach((c) => {
      if (c.dataset.sampleVideo === sampleVideo) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    videoSelectedFileBar.classList.remove('hidden');
    btnStartVideoDetection.disabled = false;
    videoSelectedName.textContent = `${sampleVideo} (Sample video ready to detect)`;
    videoSelectedSize.textContent = '60 Frames / 3.0s';
  }

  function handleVideoSelection(file, isSample) {
    state.isSampleVideo = isSample;
    state.selectedVideoFile = file;
    state.selectedSampleVideoName = isSample ? 'sample_road.mp4' : null;

    sampleVideoCards.forEach((c) => c.classList.remove('active'));

    videoSelectedFileBar.classList.remove('hidden');
    btnStartVideoDetection.disabled = false;

    if (isSample) {
      videoSelectedName.textContent = 'sample_road.mp4 (Bundled 60-frame road clip)';
      videoSelectedSize.textContent = '2.1 MB';
    } else if (file) {
      videoSelectedName.textContent = file.name;
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      videoSelectedSize.textContent = `${sizeMb} MB`;
    }
  }

  function clearVideoSelection() {
    state.selectedVideoFile = null;
    state.isSampleVideo = false;
    state.selectedSampleVideoName = null;
    sampleVideoCards.forEach((c) => c.classList.remove('active'));
    videoFileInput.value = '';
    videoSelectedFileBar.classList.add('hidden');
    btnStartVideoDetection.disabled = true;
  }

  // Start Video Detection
  btnStartVideoDetection.addEventListener('click', async () => {
    if (!state.selectedVideoFile && !state.isSampleVideo) return;

    btnStartVideoDetection.disabled = true;
    videoResultsCard.classList.add('hidden');
    videoProcessingCard.classList.remove('hidden');

    // Reset progress UI
    videoProgressFill.style.width = '0%';
    videoProgressPct.textContent = '0%';
    videoProgressFrames.textContent = 'Frame: 0 / ...';
    videoProgressFps.textContent = '0.0 FPS';
    liveCountD00.textContent = '0';
    liveCountD10.textContent = '0';
    liveCountD20.textContent = '0';
    liveCountD40.textContent = '0';

    const formData = new FormData();
    formData.append('conf', confSlider.value);
    formData.append('skip', skipSelect.value);
    formData.append('show_hud', toggleHud.checked ? 'true' : 'false');
    formData.append('classes', getSelectedClasses());

    if (state.isSampleVideo) {
      formData.append('use_sample', 'true');
      if (state.selectedSampleVideoName) {
        formData.append('sample_name', state.selectedSampleVideoName);
      }
    } else {
      formData.append('video', state.selectedVideoFile);
    }

    try {
      const response = await fetch('/api/detect/video/start', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to start video inference job');
      }

      const resData = await response.json();
      state.activeVideoJobId = resData.job_id;
      startVideoProgressPolling(state.activeVideoJobId);
    } catch (err) {
      alert(`Error starting video detection: ${err.message}`);
      videoProcessingCard.classList.add('hidden');
      btnStartVideoDetection.disabled = false;
    }
  });

  // Cancel Job
  btnCancelVideo.addEventListener('click', async () => {
    if (state.activeVideoJobId) {
      await fetch(`/api/detect/video/cancel/${state.activeVideoJobId}`, { method: 'POST' });
      clearInterval(state.videoPollInterval);
      videoProcessingCard.classList.add('hidden');
      btnStartVideoDetection.disabled = false;
    }
  });

  // Poll Video Progress
  function startVideoProgressPolling(jobId) {
    if (state.videoPollInterval) clearInterval(state.videoPollInterval);

    state.videoPollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/detect/video/progress/${jobId}`);
        if (!res.ok) return;

        const data = await res.json();

        // Update progress bar & stats
        videoProgressFill.style.width = `${data.progress}%`;
        videoProgressPct.textContent = `${data.progress}%`;
        videoProgressFrames.textContent = `Frame: ${data.current_frame} / ${data.total_frames}`;
        videoProgressFps.textContent = `${data.processing_fps} FPS`;

        // Update counters
        if (data.counts) {
          liveCountD00.textContent = data.counts.D00 || 0;
          liveCountD10.textContent = data.counts.D10 || 0;
          liveCountD20.textContent = data.counts.D20 || 0;
          liveCountD40.textContent = data.counts.D40 || 0;
        }

        // Update preview image
        if (data.preview_frame) {
          videoLivePreviewImg.src = data.preview_frame;
        }

        // Job completed
        if (data.status === 'completed') {
          clearInterval(state.videoPollInterval);
          renderVideoResults(data);
        } else if (data.status === 'error' || data.status === 'canceled') {
          clearInterval(state.videoPollInterval);
          videoProcessingCard.classList.add('hidden');
          btnStartVideoDetection.disabled = false;
          if (data.error) alert(`Error: ${data.error}`);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 250);
  }

  function renderVideoResults(data) {
    videoProcessingCard.classList.add('hidden');
    videoResultsCard.classList.remove('hidden');
    btnStartVideoDetection.disabled = false;

    // Load video player
    resultsVideoSource.src = data.video_url;
    resultsVideoPlayer.load();

    // Set download buttons
    btnDownloadVideoMp4.href = data.video_url;
    btnDownloadVideoCsv.href = data.csv_url;

    // PCI Metrics
    if (data.pci) {
      videoPciScore.textContent = data.pci.pci_score;
      videoPciScore.style.color = data.pci.color;
      videoPciBadge.textContent = `${data.pci.rating} Condition`;
      videoPciBadge.style.color = data.pci.color;
      videoPciBadge.style.background = `${data.pci.color}22`;
      videoPciSummary.textContent = data.pci.summary;
    }

    // Counters
    resD00Count.textContent = data.counts.D00 || 0;
    resD10Count.textContent = data.counts.D10 || 0;
    resD20Count.textContent = data.counts.D20 || 0;
    resD40Count.textContent = data.counts.D40 || 0;
    resTotalDet.textContent = data.total_detections || 0;
    resTimeDet.textContent = `${data.elapsed_sec || 0}s`;
  }

  // =========================================================================
  // Image Handling Logic
  // =========================================================================
  btnBrowseImage.addEventListener('click', (e) => {
    e.stopPropagation();
    imageFileInput.click();
  });

  imageDropzone.addEventListener('click', () => {
    if (!state.selectedImageFile && !state.isSampleImage) {
      imageFileInput.click();
    }
  });

  imageDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageDropzone.classList.add('drag-over');
  });

  imageDropzone.addEventListener('dragleave', () => {
    imageDropzone.classList.remove('drag-over');
  });

  imageDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    imageDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageSelection(e.dataTransfer.files[0], false);
    }
  });

  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageSelection(e.target.files[0], false);
    }
  });

  // Direct Sample Cards
  const sampleCards = document.querySelectorAll('.sample-card');
  sampleCards.forEach(card => {
    card.addEventListener('click', (e) => {
      e.stopPropagation();
      const sampleFile = card.dataset.sample;
      selectSampleCard(sampleFile);
    });
  });

  function selectSampleCard(sampleFile) {
    state.isSampleImage = true;
    state.selectedSampleName = sampleFile;
    state.selectedImageFile = null;

    sampleCards.forEach(c => {
      if (c.dataset.sample === sampleFile) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    imageSelectedFileBar.classList.remove('hidden');
    btnStartImageDetection.disabled = false;
    imageSelectedName.textContent = `${sampleFile} (Sample ready to detect)`;
  }

  btnSampleImage.addEventListener('click', (e) => {
    e.stopPropagation();
    selectSampleCard('multi_defect_road.jpg');
  });

  btnClearImage.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImageSelection();
  });

  function handleImageSelection(file, isSample) {
    state.isSampleImage = isSample;
    state.selectedImageFile = file;
    state.selectedSampleName = isSample ? 'multi_defect_road.jpg' : null;

    sampleCards.forEach(c => c.classList.remove('active'));

    imageSelectedFileBar.classList.remove('hidden');
    btnStartImageDetection.disabled = false;

    if (isSample) {
      imageSelectedName.textContent = 'multi_defect_road.jpg (Sample ready to detect)';
    } else if (file) {
      imageSelectedName.textContent = file.name;
    }
  }

  function clearImageSelection() {
    state.selectedImageFile = null;
    state.isSampleImage = false;
    state.selectedSampleName = null;
    sampleCards.forEach(c => c.classList.remove('active'));
    imageFileInput.value = '';
    imageSelectedFileBar.classList.add('hidden');
    btnStartImageDetection.disabled = true;
  }

  btnStartImageDetection.addEventListener('click', async () => {
    if (!state.selectedImageFile && !state.isSampleImage) return;

    btnStartImageDetection.disabled = true;
    btnStartImageDetection.textContent = 'Detecting Faults...';

    const formData = new FormData();
    formData.append('conf', confSlider.value);
    formData.append('classes', getSelectedClasses());

    if (state.isSampleImage) {
      formData.append('use_sample', 'true');
      if (state.selectedSampleName) {
        formData.append('sample_name', state.selectedSampleName);
      }
    } else {
      formData.append('image', state.selectedImageFile);
    }

    try {
      const res = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Image detection failed');

      const data = await res.json();
      renderImageResults(data);
    } catch (err) {
      alert(`Error during image detection: ${err.message}`);
    } finally {
      btnStartImageDetection.disabled = false;
      btnStartImageDetection.textContent = 'Run Image Damage Detection';
    }
  });

  function renderImageResults(data) {
    imageResultsCard.classList.remove('hidden');

    imgResultOriginal.src = data.image_original;
    imgResultAnnotated.src = data.image_annotated;
    btnDownloadImageResult.href = data.image_annotated;
    imageInferenceTimeSub.textContent = `Inference completed in ${data.inference_time_ms} ms (${data.total_detections} faults found)`;

    // Populate Detections Table
    imageDetectionsTbody.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((det) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${det.class_name}</strong></td>
          <td>${det.nice_name}</td>
          <td>${(det.confidence * 100).toFixed(1)}%</td>
          <td><code>[${det.bbox.join(', ')}]</code></td>
        `;
        imageDetectionsTbody.appendChild(tr);
      });
    } else {
      imageDetectionsTbody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 20px;">
            No road damage detected above the confidence threshold (${confSlider.value}).
          </td>
        </tr>
      `;
    }
  }

  // =========================================================================
  // Live Webcam Logic
  // =========================================================================
  btnStartWebcam.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });

      state.webcamStream = stream;
      webcamVideoHidden.srcObject = stream;
      await webcamVideoHidden.play();

      state.webcamRunning = true;
      btnStartWebcam.disabled = true;
      btnStopWebcam.disabled = false;
      webcamPlaceholderBox.classList.add('hidden');
      webcamAnnotatedStream.classList.remove('hidden');

      startWebcamLoop();
    } catch (err) {
      alert(`Could not access webcam: ${err.message}`);
    }
  });

  btnStopWebcam.addEventListener('click', () => {
    stopWebcamStream();
  });

  function stopWebcamStream() {
    state.webcamRunning = false;
    btnStartWebcam.disabled = false;
    btnStopWebcam.disabled = true;

    if (state.webcamInterval) {
      clearInterval(state.webcamInterval);
      state.webcamInterval = null;
    }

    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((track) => track.stop());
      state.webcamStream = null;
    }

    webcamPlaceholderBox.classList.remove('hidden');
    webcamAnnotatedStream.classList.add('hidden');
    webcamFpsVal.textContent = '0.0';
    webcamLatencyVal.textContent = '0 ms';
    webcamCountVal.textContent = '0';
  }

  function startWebcamLoop() {
    const ctx = webcamCanvasHidden.getContext('2d');
    let isProcessingFrame = false;
    let lastTime = performance.now();
    let frameCounter = 0;
    let fpsTimer = performance.now();

    state.webcamInterval = setInterval(async () => {
      if (!state.webcamRunning || isProcessingFrame) return;

      const vW = webcamVideoHidden.videoWidth || 640;
      const vH = webcamVideoHidden.videoHeight || 480;

      webcamCanvasHidden.width = vW;
      webcamCanvasHidden.height = vH;
      ctx.drawImage(webcamVideoHidden, 0, 0, vW, vH);

      const frameBase64 = webcamCanvasHidden.toDataURL('image/jpeg', 0.7);

      isProcessingFrame = true;
      const reqStart = performance.now();

      try {
        const res = await fetch('/api/detect/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: frameBase64,
            conf: parseFloat(confSlider.value),
          }),
        });

        if (res.ok) {
          const data = await res.json();
          webcamAnnotatedStream.src = data.image;

          const latency = Math.round(performance.now() - reqStart);
          webcamLatencyVal.textContent = `${latency} ms`;
          webcamCountVal.textContent = data.total_detections || 0;

          frameCounter++;
          const now = performance.now();
          if (now - fpsTimer >= 1000) {
            webcamFpsVal.textContent = ((frameCounter * 1000) / (now - fpsTimer)).toFixed(1);
            frameCounter = 0;
            fpsTimer = now;
          }
        }
      } catch (err) {
        console.warn('Webcam frame inference failed:', err);
      } finally {
        isProcessingFrame = false;
      }
    }, 120); // ~8-10 FPS rate
  }

  // =========================================================================
  // Database & Inspection History Logic
  // =========================================================================
  let cachedInspections = [];

  async function loadDatabaseHistory() {
    try {
      // 1. Fetch Stats
      const statsRes = await fetch('/api/database/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        const statTotal = document.getElementById('db-stat-total-inspections');
        const statDefects = document.getElementById('db-stat-total-defects');
        const statPci = document.getElementById('db-stat-avg-pci');
        if (statTotal) statTotal.textContent = statsData.total_inspections || 0;
        if (statDefects) statDefects.textContent = statsData.total_detections || 0;
        if (statPci) {
          const avg = statsData.average_pci || 100.0;
          statPci.textContent = avg.toFixed(1);
          statPci.style.color = avg >= 70 ? '#10b981' : avg >= 50 ? '#f59e0b' : '#ef4444';
        }
      }

      // 2. Fetch Inspections
      const inspRes = await fetch('/api/database/inspections');
      if (inspRes.ok) {
        cachedInspections = await inspRes.json();
        renderDatabaseTable(cachedInspections);
      }
    } catch (err) {
      console.warn('Failed to load database history:', err);
    }
  }

  function renderDatabaseTable(inspections) {
    const tbody = document.getElementById('db-inspections-tbody');
    if (!tbody) return;

    if (!inspections || inspections.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 25px;">
            No inspections logged yet. Run an image or video inspection to log results.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = inspections.slice(0, 50).map((insp) => {
      const dt = insp.created_at ? new Date(insp.created_at).toLocaleString() : 'Just now';
      const badgeCls = (insp.pci_rating || 'Good').toLowerCase();
      const pciScore = Math.round(insp.pci_score || 100);
      const isVideo = insp.media_type === 'video';
      return `
        <tr>
          <td><code style="font-size: 0.72rem; color: var(--accent-cyan);">${insp.id}</code></td>
          <td><strong>${insp.name || 'Inspection'}</strong></td>
          <td><span class="type-pill ${isVideo ? 'pill-video' : 'pill-image'}">${(insp.media_type || 'image').toUpperCase()}</span></td>
          <td style="font-size: 0.78rem; color: var(--text-muted);">${dt}</td>
          <td>
            <span style="color:#008cff; font-weight:600;" title="D00 Longitudinal">${insp.d00_count || 0}</span> |
            <span style="color:#32cd32; font-weight:600;" title="D10 Transverse">${insp.d10_count || 0}</span> |
            <span style="color:#ffa500; font-weight:600;" title="D20 Alligator">${insp.d20_count || 0}</span> |
            <span style="color:#dc2222; font-weight:700;" title="D40 Pothole">${insp.d40_count || 0}</span>
          </td>
          <td><strong>${insp.total_defects || 0}</strong></td>
          <td><strong style="color: ${pciScore >= 70 ? '#10b981' : pciScore >= 50 ? '#f59e0b' : '#ef4444'}">${pciScore}</strong></td>
          <td><span class="pci-badge badge-${badgeCls}">${insp.pci_rating || 'Good'}</span></td>
        </tr>
      `;
    }).join('');
  }

  const dbSearchInput = document.getElementById('db-search-input');
  if (dbSearchInput) {
    dbSearchInput.addEventListener('input', () => {
      const q = dbSearchInput.value.toLowerCase().trim();
      if (!q) {
        renderDatabaseTable(cachedInspections);
      } else {
        const filtered = cachedInspections.filter((item) =>
          (item.name || '').toLowerCase().includes(q) ||
          (item.id || '').toLowerCase().includes(q) ||
          (item.pci_rating || '').toLowerCase().includes(q) ||
          (item.media_type || '').toLowerCase().includes(q)
        );
        renderDatabaseTable(filtered);
      }
    });
  }

  const btnRefreshDb = document.getElementById('btn-refresh-db');
  if (btnRefreshDb) {
    btnRefreshDb.addEventListener('click', () => {
      loadDatabaseHistory();
    });
  }
});
