/* ============================================
   Venom Watch — Frontend JavaScript
   Handles: Drag & Drop, Preview, AJAX Upload,
   Result Display, Reset
   ============================================ */

(function () {
    'use strict';

    // ============================================
    // DOM ELEMENTS
    // ============================================
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const actionButtons = document.getElementById('actionButtons');
    const detectBtn = document.getElementById('detectBtn');
    const resetBtn = document.getElementById('resetBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultCard = document.getElementById('resultCard');
    const resultBanner = document.getElementById('resultBanner');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const annotatedImage = document.getElementById('annotatedImage');
    const annotatedImageContainer = document.getElementById('annotatedImageContainer');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    const newDetectionBtn = document.getElementById('newDetectionBtn');

    // Detail fields
    const detailDetection = document.getElementById('detailDetection');
    const detailBbox = document.getElementById('detailBbox');
    const detailConfidence = document.getElementById('detailConfidence');
    const detailRecordId = document.getElementById('detailRecordId');

    // ============================================
    // STATE
    // ============================================
    let selectedFile = null;
    const MAX_SIZE_MB = parseInt(document.querySelector('.drop-zone').dataset.maxSize || '10', 10);
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
    const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];

    // ============================================
    // EVENT: Browse Button Click
    // ============================================
    if (browseBtn) {
        browseBtn.addEventListener('click', function () {
            fileInput.click();
        });
    }

    // ============================================
    // EVENT: Drop Zone Click
    // ============================================
    if (dropZone) {
        dropZone.addEventListener('click', function (e) {
            if (e.target.tagName !== 'BUTTON') {
                fileInput.click();
            }
        });
    }

    // ============================================
    // EVENT: File Input Change
    // ============================================
    if (fileInput) {
        fileInput.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }

    // ============================================
    // DRAG & DROP EVENTS
    // ============================================
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, function (e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, function (e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', function (e) {
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    // ============================================
    // HANDLE FILE SELECTION
    // ============================================
    function handleFile(file) {
        hideError();

        // Validate file type
        const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            showError(
                `Unsupported file format "${ext}". Please upload JPG, JPEG, or PNG.`
            );
            return;
        }

        // Validate file type via MIME
        if (!ALLOWED_TYPES.includes(file.type) && file.type !== '') {
            showError(
                `Invalid file type. Please upload JPG, JPEG, or PNG images.`
            );
            return;
        }

        // Validate file size
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > MAX_SIZE_MB) {
            showError(
                `File size (${sizeMB.toFixed(1)} MB) exceeds the maximum allowed size of ${MAX_SIZE_MB} MB.`
            );
            return;
        }

        // Valid file — store and preview
        selectedFile = file;
        showPreview(file);
        showActionButtons();
    }

    // ============================================
    // SHOW IMAGE PREVIEW
    // ============================================
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewContainer.classList.remove('d-none');
            previewContainer.style.display = 'block';
            fileNameDisplay.textContent =
                `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        };
        reader.readAsDataURL(file);
    }

    // ============================================
    // SHOW/HIDE ACTION BUTTONS
    // ============================================
    function showActionButtons() {
        actionButtons.style.display = 'flex';
    }

    function hideActionButtons() {
        actionButtons.style.display = 'none';
    }

    // ============================================
    // EVENT: Detect Button Click
    // ============================================
    if (detectBtn) {
        detectBtn.addEventListener('click', function () {
            if (!selectedFile) {
                showError('Please select an image first.');
                return;
            }
            performDetection();
        });
    }

    // ============================================
    // PERFORM DETECTION (AJAX to /api/detect/)
    // ============================================
    function performDetection() {
        // Show loading
        hideError();
        hideResultCard();
        hideActionButtons();
        loadingSpinner.classList.remove('d-none');
        detectBtn.disabled = true;

        // Build FormData
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('csrfmiddlewaretoken', getCSRFToken());

        // Send request
        fetch('/api/detect/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCSRFToken(),
            },
        })
        .then(response => {
            return response.json().then(data => {
                if (!response.ok) {
                    throw new Error(data.error || 'Detection request failed.');
                }
                return data;
            });
        })
        .then(data => {
            loadingSpinner.classList.add('d-none');
            detectBtn.disabled = false;
            showResult(data);
        })
        .catch(error => {
            loadingSpinner.classList.add('d-none');
            detectBtn.disabled = false;
            showActionButtons();
            showError(error.message || 'An unexpected error occurred.');
        });
    }

    // ============================================
    // SHOW RESULT
    // ============================================
    function showResult(data) {
        // Build banner
        if (data.snake) {
            resultBanner.className = 'result-banner text-center mb-4 snake-found';
            resultBanner.innerHTML = `
                <i class="bi bi-exclamation-triangle-fill result-icon text-danger"></i>
                <h3>🐍 Snake Detected!</h3>
                <p class="text-muted mb-0 mt-1">A snake was found in the image.</p>
            `;
        } else {
            resultBanner.className = 'result-banner text-center mb-4 no-snake';
            resultBanner.innerHTML = `
                <i class="bi bi-check-circle-fill result-icon text-success"></i>
                <h3>✅ No Snake Detected</h3>
                <p class="text-muted mb-0 mt-1">No snake was found in the image.</p>
            `;
        }

        // Confidence bar
        const conf = data.confidence || 0;
        confidenceText.textContent = conf.toFixed(2) + '%';
        confidenceBar.style.width = '0%';

        // Color the bar based on result
        if (data.snake) {
            confidenceBar.className = 'progress-bar bg-danger';
        } else {
            confidenceBar.className = 'progress-bar bg-success';
        }

        // Animate confidence bar
        setTimeout(() => {
            confidenceBar.style.width = conf + '%';
        }, 300);

        // Annotated image
        if (data.annotated_image_url) {
            annotatedImage.src = data.annotated_image_url;
            annotatedImageContainer.classList.remove('d-none');
            annotatedImageContainer.style.display = 'block';
        } else {
            annotatedImageContainer.classList.add('d-none');
            annotatedImageContainer.style.display = 'none';
        }

        // Details
        detailDetection.innerHTML = data.snake
            ? '<span class="text-danger fw-bold">Snake Detected</span>'
            : '<span class="text-success fw-bold">No Snake</span>';

        detailBbox.textContent = data.bbox
            ? `[${data.bbox.join(', ')}]`
            : 'N/A';

        detailConfidence.innerHTML = `<span class="text-warning fw-bold">${conf.toFixed(2)}%</span>`;

        detailRecordId.textContent = data.record_id || 'N/A';

        // Show result card with animation
        resultCard.classList.remove('d-none');
        resultCard.style.animation = 'fadeInUp 0.5s ease';

        // Scroll to result
        setTimeout(() => {
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    // ============================================
    // HIDE RESULT
    // ============================================
    function hideResultCard() {
        resultCard.classList.add('d-none');
    }

    // ============================================
    // ERROR HANDLING
    // ============================================
    function showError(message) {
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorAlert.classList.add('d-none');
    }

    // ============================================
    // RESET
    // ============================================
    function resetAll() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.add('d-none');
        previewContainer.style.display = 'none';
        hideActionButtons();
        hideResultCard();
        hideError();
        loadingSpinner.classList.add('d-none');
        detectBtn.disabled = false;

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetAll);
    }

    if (newDetectionBtn) {
        newDetectionBtn.addEventListener('click', resetAll);
    }

    // ============================================
    // CSRF TOKEN HELPER
    // ============================================
    function getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith('csrftoken=')) {
                return cookie.substring('csrftoken='.length);
            }
        }
        // Fallback: look for hidden input
        const csrfInput = document.querySelector('[name=csrfmiddlewaretoken]');
        return csrfInput ? csrfInput.value : '';
    }

    // Set data attribute for max size on dropzone
    if (dropZone) {
        const maxMb = document.querySelector('.drop-zone p').textContent.match(/Max Size:\s*(\d+)/);
        if (maxMb) {
            dropZone.dataset.maxSize = maxMb[1];
        }
    }

})();