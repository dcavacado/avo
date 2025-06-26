
const classNames = ['avocado', 'stem'];
const modelPath = "best.onnx";
let session = null;

const inputSize = 640; // YOLOv8 default

async function initModel() {
    session = await ort.InferenceSession.create(modelPath);
    console.log("✅ Model loaded");
}

function preprocess(video, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, inputSize, inputSize);
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
    const { data } = imageData;

    const input = new Float32Array(inputSize * inputSize * 3);
    for (let i = 0; i < inputSize * inputSize; i++) {
        input[i] = data[i * 4] / 255;         // R
        input[i + inputSize * inputSize] = data[i * 4 + 1] / 255; // G
        input[i + 2 * inputSize * inputSize] = data[i * 4 + 2] / 255; // B
    }

    return new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]);
}

function drawBoxes(ctx, boxes, scaleX, scaleY) {
    ctx.lineWidth = 2;
    ctx.font = "16px sans-serif";
    boxes.forEach(obj => {
        const [x1, y1, x2, y2] = obj.box;
        const label = classNames[obj.class];
        const score = obj.score.toFixed(2);
        ctx.strokeStyle = "lime";
        ctx.fillStyle = "lime";
        ctx.beginPath();
        ctx.rect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
        ctx.stroke();
        ctx.fillText(`${label} ${score}`, x1 * scaleX + 5, y1 * scaleY + 20);
    });
}

function processOutput(output, confThreshold = 0.4) {
    const boxes = [];
    const preds = output.data;
    const numDetections = preds.length / 84;

    for (let i = 0; i < numDetections; i++) {
        const offset = i * 84;
        const objConf = preds[offset + 4];
        if (objConf < confThreshold) continue;

        const classScores = preds.slice(offset + 5, offset + 84);
        const maxScore = Math.max(...classScores);
        const classId = classScores.indexOf(maxScore);
        if (maxScore * objConf < confThreshold) continue;

        const cx = preds[offset];
        const cy = preds[offset + 1];
        const w = preds[offset + 2];
        const h = preds[offset + 3];
        const x1 = cx - w / 2;
        const y1 = cy - h / 2;
        const x2 = cx + w / 2;
        const y2 = cy + h / 2;

        boxes.push({
            box: [x1, y1, x2, y2],
            class: classId,
            score: maxScore * objConf,
        });
    }
    return boxes;
}

async function startCamera() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            video.play();
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
        };
    });
}

async function detectLoop(video, canvas) {
    const ctx = canvas.getContext("2d");

    while (true) {
        const inputTensor = preprocess(video, canvas);
        const output = await session.run({ images: inputTensor });
        const outputTensor = Object.values(output)[0];
        console.log("Output shape:", outputTensor.dims);
        console.log("Output data sample:", outputTensor.data.slice(0, 10));
        const boxes = processOutput(outputTensor);

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        drawBoxes(ctx, boxes, canvas.width / inputSize, canvas.height / inputSize);

        await new Promise(r => setTimeout(r, 100)); // ~10 fps
    }
}

(async () => {
    await initModel();
    await startCamera();
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    detectLoop(video, canvas);
})();
