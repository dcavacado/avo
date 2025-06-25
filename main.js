
const classNames = ['avocado', 'stem'];
const modelPath = "best.onnx";
let session = null;

async function initModel() {
    session = await ort.InferenceSession.create(modelPath);
    console.log("Model loaded");
}

async function runDetection(frame) {
    // Placeholder — actual YOLOv8 preprocessing and postprocessing needed
    // You can integrate with actual ONNX model here
    const ctx = document.getElementById("canvas").getContext("2d");
    ctx.drawImage(frame, 0, 0);
    ctx.fillStyle = "lime";
    ctx.fillText("Stub detection: avocado", 20, 20);
}

async function startCamera() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        detectLoop(video);
    };
}

async function detectLoop(video) {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    while (true) {
        ctx.drawImage(video, 0, 0);
        await runDetection(video);
        await new Promise(r => setTimeout(r, 100));  // 10 fps
    }
}

initModel();
startCamera();
