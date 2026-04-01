// UI Architecture Nodes
const welcomePage = document.getElementById('welcome-page');
const dashboardPage = document.getElementById('dashboard-page');
const getStartedBtn = document.getElementById('getStartedBtn');

// Dashboard Input Nodes
const dropZone = document.getElementById('drop-zone');
const imageUpload = document.getElementById('imageUpload');
const inputView = document.getElementById('input-view');
const previewView = document.getElementById('preview-view');
const imagePreview = document.getElementById('imagePreview');
const resetImageBtn = document.getElementById('resetImageBtn');

// Camera Nodes
const alternateActions = document.getElementById('alternate-actions-container');
const cameraBtn = document.getElementById('cameraBtn');
const cameraBox = document.getElementById('camera-box');
const videoElement = document.getElementById('videoElement');
const captureBtn = document.getElementById('captureBtn');
const cancelCameraBtn = document.getElementById('cancelCameraBtn');
const canvasElement = document.getElementById('canvasElement');

// Analytics Nodes
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultSection = document.getElementById('resultSection');
const resetAppBtns = document.querySelectorAll('.reset-app-btn');

// Result Metric Nodes
const healthScoreDisplay = document.getElementById('healthScoreDisplay');
const scoreBar = document.getElementById('scoreBar');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');

let selectedFile = null;
let cameraStream = null;

// Routing: Welcome -> Dashboard
getStartedBtn.addEventListener('click', () => {
    // Add slide out animation out of welcome page
    welcomePage.classList.remove('active');
    welcomePage.classList.add('fade-out');
    
    // Smooth transition
    setTimeout(() => {
        welcomePage.classList.add('hidden');
        dashboardPage.classList.remove('hidden');
        
        // Force reflow
        void dashboardPage.offsetWidth; 
        
        dashboardPage.classList.remove('opacity-0', 'translate-y-8');
        dashboardPage.classList.add('active');
    }, 500);
});

/* ====================================
   FILE INPUT (DRAG & DROP)
==================================== */

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight effect
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('border-brand-green', 'bg-gray-800/80');
        dropZone.classList.remove('border-gray-600', 'bg-gray-900/50');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('border-brand-green', 'bg-gray-800/80');
        dropZone.classList.add('border-gray-600', 'bg-gray-900/50');
    }, false);
});

// Handle dropped files
dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFile(files[0]);
});

// Handle clicked files
imageUpload.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        // Set user's image in UI
        imagePreview.src = e.target.result;
        
        // Transition card to 'Preview Mode'
        inputView.classList.add('hidden');
        previewView.classList.remove('hidden');
        
        // Finalize state
        analyzeBtn.disabled = false;
        resultSection.classList.remove('result-show');
        setTimeout(() => resultSection.classList.add('hidden'), 500);
    }
    reader.readAsDataURL(file);
}

resetImageBtn.addEventListener('click', () => {
    goBackToInput();
});

function goBackToInput() {
    selectedFile = null;
    imagePreview.src = '';
    
    // Transition card back to 'Input Mode'
    previewView.classList.add('hidden');
    inputView.classList.remove('hidden');
    
    analyzeBtn.disabled = true;
    
    // Hide results if they exist
    resultSection.classList.remove('result-show');
    setTimeout(() => { 
        if(!resultSection.classList.contains('result-show')) {
            resultSection.classList.add('hidden');
        }
    }, 500);
    
    stopCamera();
}

/* ====================================
   CAMERA INTEGRATION 
==================================== */

cameraBtn.addEventListener('click', async () => {
    try {
        cameraBox.classList.remove('hidden');
        alternateActions.classList.add('hidden'); // Hide Or... buttons
        dropZone.classList.add('hidden'); // Hide drop zone for focus
        
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        videoElement.srcObject = cameraStream;
    } catch (err) {
        alert("Unable to access camera. Please allow permissions.");
        stopCamera();
    }
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraBox.classList.add('hidden');
    alternateActions.classList.remove('hidden');
    dropZone.classList.remove('hidden');
}

cancelCameraBtn.addEventListener('click', stopCamera);

captureBtn.addEventListener('click', () => {
    const context = canvasElement.getContext('2d');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    stopCamera();

    canvasElement.toBlob((blob) => {
        const file = new File([blob], "camera_snap.png", { type: "image/png" });
        handleFile(file);
    }, 'image/png');
});


/* ====================================
   API INFERENCE & RESULT LOGIC
==================================== */

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Set Loading State
    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('hidden');
    loadingIndicator.style.display = 'flex'; // Force flex via tailwind override precaution
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        // Core Backend Pipeline Link
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error('Prediction API rejected the request.');

        const data = await res.json();
        const score = data.health_score;

        // Artificial delay so the processing animation can be experienced by the user
        setTimeout(() => {
            
            // 1. Unmount loading logic
            loadingIndicator.classList.add('hidden');
            loadingIndicator.style.display = 'none';

            // 2. Clear old result values
            scoreBar.style.width = '0%';
            confidenceBar.style.width = '0%';

            // 3. Mount text results
            healthScoreDisplay.innerText = score;
            predictionText.innerText = data.prediction;
            confidenceText.innerText = data.confidence.toFixed(2);
            
            // 4. Update the color themes based on the health integer
            updateScoreStyles(score);

            // 5. Trigger CSS animations
            resultSection.classList.remove('hidden'); 
            
            // Allow DOM repainting before executing CSS transition scale/opacity
            setTimeout(() => {
                resultSection.classList.add('result-show');
                scoreBar.style.width = `${score}%`;
                confidenceBar.style.width = `${data.confidence.toFixed(0)}%`;
            }, 50);
            
        }, 1200);

    } catch (err) {
        alert("Backend network error. Check if the Python API is running.");
        loadingIndicator.classList.add('hidden');
        loadingIndicator.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function updateScoreStyles(score) {
    // Wipe previous applied metric utility classes
    healthScoreDisplay.className = 'text-6xl md:text-7xl font-tech font-bold tracking-tighter transition-colors duration-500';
    scoreBar.className = 'h-full rounded-full transition-all duration-1000 ease-out shadow-lg relative ' + 
                         'w-0'; // Ensure width is 0 before animating

    if (score >= 70) {
        healthScoreDisplay.classList.add('text-theme-green');
        scoreBar.classList.add('bg-theme-green');
    } else if (score >= 40) {
        healthScoreDisplay.classList.add('text-theme-yellow');
        scoreBar.classList.add('bg-theme-yellow');
    } else {
        healthScoreDisplay.classList.add('text-theme-red');
        scoreBar.classList.add('bg-theme-red');
    }
}

// Global reset
resetAppBtns.forEach(btn => btn.addEventListener('click', () => {
    goBackToInput();
}));
