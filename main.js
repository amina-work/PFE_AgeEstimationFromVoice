let isRecording = false;
let isRecordingText = document.getElementById("isRecording");
let micToggle = document.getElementById("micToggle");
let micIcon = document.getElementById("micIcon");
let audioChunks = [];
let rec;

micToggle.addEventListener("click", toggleRecording);

async function getUserMedia(constraints) {
    if (navigator.mediaDevices) {
        return navigator.mediaDevices.getUserMedia(constraints);
    }
    let legacyApi =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
    if (legacyApi) {
        return new Promise(function (resolve, reject) {
            legacyApi.bind(navigator)(constraints, resolve, reject);
        });
    } else {
        alert("User media API not supported");
    }
}

function handlerFunction(stream) {
    rec = new MediaRecorder(stream);
    rec.start();
    rec.ondataavailable = (e) => {
        audioChunks.push(e.data);
        if (rec.state == "inactive") {
            let blob = new Blob(audioChunks, { type: "audio/mp3" });
            console.log(blob);
            document.getElementById("audioElement").src = URL.createObjectURL(blob);
        }
    };
}

function startRecording() {
    getUserMedia({ audio: true }).then((stream) => {
        handlerFunction(stream);
    });
    isRecording = true;
    isRecordingText.textContent = "Recording...";
    micIcon.textContent = "stop";
}

function stopRecording() {
    rec.stop();
    isRecording = false;
    isRecordingText.textContent = "Click the microphone button to start recording";
    micIcon.textContent = "mic";
}

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}
