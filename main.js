let isRecording = false;
let isRecordingText = document.getElementById("isRecording");
let micToggle = document.getElementById("micToggle");
let micIcon = document.getElementById("micIcon");
let audioChunks = [];
let rec;

// Get references to page elements
let nextPageButton = document.getElementById("nextpage");
let goBackButton = document.getElementById("goback");
let pageOne = document.getElementById("pageone");
let pageTwo = document.getElementById("pagetwo");

// Initially hide the "See results" button and pagetwo
nextPageButton.style.display = "none";
pageTwo.style.display = "none";

micToggle.addEventListener("click", toggleRecording);
nextPageButton.addEventListener("click", showPageTwo);
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
    audioChunks = []; // Clearing the audioChunks array before starting a new recording
    rec.ondataavailable = (e) => {
        audioChunks.push(e.data);
        if (rec.state == "inactive") {
            let blob = new Blob(audioChunks, { type: "audio/mp3" });
            console.log(blob);
            document.getElementById("audioElement").src = URL.createObjectURL(blob);
            // Show the "See results" button when a recording is available
            nextPageButton.style.display = "block";
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

function showPageTwo() {
    pageOne.style.display = "none";
    pageTwo.style.display = "flex";
}

function showPageOne() {
    pageTwo.style.display = "none";
    pageOne.style.display = "flex";
}
