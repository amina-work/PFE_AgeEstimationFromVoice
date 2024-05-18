let isRecording = false;
let isRecordingText = document.getElementById("isRecording");
let micToggle = document.getElementById("micToggle");
let micIcon = document.getElementById("micIcon");
let audioChunks = [];
let rec;

let submitDetailsButton = document.getElementById("submitDetails");
let goBackButton = document.getElementById("goback");
let pageOne = document.getElementById("pageone");
let pageThree = document.getElementById("pagethree");

submitDetailsButton.style.display = "none";
pageThree.style.display = "none";

micToggle.addEventListener("click", toggleRecording);
submitDetailsButton.addEventListener("click", submitDetails);
goBackButton.addEventListener("click", showPageOne);

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
    audioChunks = [];
    rec.ondataavailable = (e) => {
        audioChunks.push(e.data);
    };
    rec.onstop = () => {
        let blob = new Blob(audioChunks, { type: "audio/mp3" });
        document.getElementById("audioElement").src = URL.createObjectURL(blob);
        submitDetailsButton.style.display = "block"; // Ensure submit button is displayed when recording stops
    };
}

function sendBlobToFlask(blob) {
    let formData = new FormData();
    formData.append('audio', blob, 'recording.mp3');
    formData.append('gender', document.getElementById('gender').value);
    formData.append('accent', document.getElementById('accent').value);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        console.log(data);
        updateResult(data.predicted_age);
        showPageThree();
    })
    .catch(error => {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    });
}

function updateResult(predictedAge) {
    document.querySelector('.result').textContent = `We estimate you are ${predictedAge.toFixed(2)} years old`;
}

navigator.mediaDevices.getUserMedia({ audio: true }).then(handlerFunction).catch(e => console.error('getUserMedia Error:', e));

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        handlerFunction(stream);
    });
    isRecording = true;
    isRecordingText.textContent = "Recording...";
    micIcon.textContent = "stop";
}

function stopRecording() {
    rec.stop();
    isRecording = false;
    isRecordingText.textContent = "Click the microphone to start re-recording";
    micIcon.textContent = "mic";
}

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function submitDetails() {
    if (isRecording) {
        stopRecording();
    }
    let blob = new Blob(audioChunks, { type: "audio/mp3" });
    sendBlobToFlask(blob);
}

function showPageThree() {
    pageOne.style.display = "none";
    pageThree.style.display = "flex";
}

function showPageOne() {
    pageThree.style.display = "none";
    pageOne.style.display = "flex";
}