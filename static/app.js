const $ = id => document.getElementById(id);
let sessionId = localStorage.getItem("voice-router-session") || null;
let callActive = false, processing = false, mediaStream = null, mediaRecorder = null;
let audioContext = null, analyser = null, monitorFrame = null, currentAudio = null, chunks = [];
let callStartedAt = null, timerId = null, lastScenario = null;

const formatTime = (date = new Date()) => date.toLocaleTimeString("ru-RU", {hour: "2-digit", minute: "2-digit"});
function setCallCopy(title, hint, eyebrow = "ЗВОНОК АКТИВЕН") { $("callTitle").textContent = title; $("callHint").textContent = hint; $("callEyebrow").textContent = eyebrow; }
function setVoiceState(state, label) {
  $("callButton").classList.toggle("active", state !== "idle");
  $("callButton").classList.toggle("processing", state === "processing");
  $("callButton").classList.toggle("speaking", state === "speaking");
  $("callState").classList.toggle("live", state !== "idle");
  $("callState").querySelector("span").textContent = label;
}

function addTranscript(role, text) {
  $("emptyTranscript")?.remove();
  const entry = document.createElement("article"); entry.className = `transcript-entry ${role}`;
  const meta = document.createElement("div"); meta.className = "entry-meta";
  const speaker = document.createElement("span"); speaker.textContent = role === "client" ? "КЛИЕНТ" : role === "assistant" ? "VOICE ROUTER" : "СИСТЕМА";
  const time = document.createElement("time"); time.textContent = formatTime(); meta.append(speaker, time);
  const copy = document.createElement("p"); copy.textContent = text; entry.append(meta, copy); $("messages").appendChild(entry);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function addScenarioSwitch(id) {
  const node = document.createElement("div"); node.className = "scenario-switch";
  const dot = document.createElement("i"), text = document.createElement("span"), strong = document.createElement("b");
  text.append("Переключение на сценарий "); strong.textContent = id; text.append(strong); node.append(dot, text); $("messages").appendChild(node);
}

function renderScenario(data) {
  const scenario = data.trace.scenarios[0]; if (!scenario) return;
  if (lastScenario !== scenario.scenario_id) addScenarioSwitch(scenario.scenario_id); lastScenario = scenario.scenario_id;
  const state = $("scenarioState"); state.querySelector(".scenario-number").textContent = scenario.scenario_id;
  state.querySelector("strong").textContent = `${Math.round(scenario.confidence * 100)}% уверенности`;
  state.querySelector("small").textContent = scenario.reason || "Активный маршрут разговора";
  $("scenarioConfidence").hidden = false; $("confidenceValue").style.width = `${Math.round(scenario.confidence * 100)}%`;
  $("messages").scrollTop = $("messages").scrollHeight;
}

function browserSpeak(text, language) {
  return new Promise(resolve => {
    if (!window.speechSynthesis || !callActive) return resolve();
    const utterance = new SpeechSynthesisUtterance(text); utterance.lang = language === "kk" ? "kk-KZ" : "ru-RU"; utterance.rate = .96;
    utterance.onend = resolve; utterance.onerror = resolve; speechSynthesis.speak(utterance);
  });
}

async function speak(text, language) {
  if (!callActive) return;
  setVoiceState("speaking", "OpenAI озвучивает ответ"); setCallCopy("Отвечаю", "После ответа можете продолжить говорить");
  try {
    const response = await fetch("/api/speech", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text, language})});
    if (!response.ok) throw new Error("OpenAI TTS unavailable");
    const blob = await response.blob(), url = URL.createObjectURL(blob);
    await new Promise(resolve => {
      currentAudio = new Audio(url); currentAudio.onended = resolve; currentAudio.onerror = resolve; currentAudio.play().catch(resolve);
    });
    URL.revokeObjectURL(url); currentAudio = null;
  } catch (_) { await browserSpeak(text, language); }
}

async function routeTranscript(text) {
  if (!text.trim() || processing || !callActive) return;
  processing = true; addTranscript("client", text.trim()); setVoiceState("processing", "Определяю сценарий"); setCallCopy("Обрабатываю", "LLM выбирает нужный сценарий разговора");
  try {
    const response = await fetch("/api/route", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text: text.trim(), session_id: sessionId})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Ошибка маршрутизации");
    sessionId = data.session_id; localStorage.setItem("voice-router-session", sessionId); renderScenario(data); addTranscript("assistant", data.reply); await speak(data.reply, data.trace.language);
  } catch (error) { addTranscript("error", error.message); }
  finally {
    processing = false;
    if (callActive) { setVoiceState("listening", "Слушаю вас"); setCallCopy("Я слушаю", "Говорите свободно — сценарий переключится автоматически"); startRecording(); }
  }
}

async function transcribeRecording(blob) {
  if (!callActive || !blob.size) return;
  processing = true; setVoiceState("processing", "OpenAI распознаёт речь"); setCallCopy("Распознаю", "Преобразую голос в текст");
  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      headers: {"Content-Type": blob.type || "audio/webm", "X-Language-Hint": $("speechLang").value},
      body: blob
    });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Ошибка распознавания речи");
    processing = false; if (data.text) await routeTranscript(data.text); else if (callActive) startRecording();
  } catch (error) { processing = false; addTranscript("error", error.message); if (callActive) startRecording(); }
}

function monitorVoice() {
  if (!analyser || !mediaRecorder || mediaRecorder.state !== "recording") return;
  const values = new Uint8Array(analyser.fftSize), startedAt = performance.now();
  let voiceStarted = false, silenceStartedAt = null;
  const check = () => {
    if (!callActive || !mediaRecorder || mediaRecorder.state !== "recording") return;
    analyser.getByteTimeDomainData(values); let energy = 0;
    for (const value of values) { const normalized = (value - 128) / 128; energy += normalized * normalized; }
    const volume = Math.sqrt(energy / values.length), now = performance.now();
    if (volume > .022) { voiceStarted = true; silenceStartedAt = null; }
    else if (voiceStarted) { silenceStartedAt ??= now; if (now - silenceStartedAt > 1100) { mediaRecorder.stop(); return; } }
    else if (now - startedAt > 12000) { mediaRecorder.stop(); return; }
    monitorFrame = requestAnimationFrame(check);
  };
  check();
}

function startRecording() {
  if (!callActive || processing || !mediaStream || mediaRecorder?.state === "recording") return;
  chunks = []; const preferred = "audio/webm;codecs=opus";
  const options = MediaRecorder.isTypeSupported(preferred) ? {mimeType: preferred} : undefined;
  mediaRecorder = new MediaRecorder(mediaStream, options);
  mediaRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  mediaRecorder.onstop = () => {
    cancelAnimationFrame(monitorFrame); if (!callActive) return;
    const blob = new Blob(chunks, {type: mediaRecorder.mimeType || "audio/webm"});
    if (blob.size > 1000) transcribeRecording(blob); else startRecording();
  };
  mediaRecorder.start(250); monitorVoice();
}

function startTimer() {
  callStartedAt = Date.now(); clearInterval(timerId);
  timerId = setInterval(() => { const elapsed = Math.floor((Date.now() - callStartedAt) / 1000); $("callTimer").textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`; }, 1000);
}

async function startCall() {
  if (callActive) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
    audioContext = new AudioContext(); analyser = audioContext.createAnalyser(); analyser.fftSize = 1024; audioContext.createMediaStreamSource(mediaStream).connect(analyser);
  } catch (_) { addTranscript("error", "Разрешите доступ к микрофону в настройках браузера."); return; }
  callActive = true; $("callButton").setAttribute("aria-pressed", "true"); $("callButton").setAttribute("aria-label", "Завершить звонок"); $("reset").hidden = false;
  startTimer(); setVoiceState("listening", "Слушаю вас"); setCallCopy("Я слушаю", "Говорите свободно — сценарий переключится автоматически"); startRecording();
}

async function endCall({resetSession = false} = {}) {
  callActive = false; processing = false; clearInterval(timerId); cancelAnimationFrame(monitorFrame);
  if (mediaRecorder?.state === "recording") mediaRecorder.stop(); mediaStream?.getTracks().forEach(track => track.stop()); mediaStream = null; analyser = null;
  await audioContext?.close().catch(() => {}); audioContext = null; currentAudio?.pause(); currentAudio = null; speechSynthesis?.cancel();
  $("callButton").setAttribute("aria-pressed", "false"); $("callButton").setAttribute("aria-label", "Начать звонок"); $("reset").hidden = true;
  setVoiceState("idle", "Нажмите, чтобы начать"); setCallCopy("Начать разговор", "Нажмите на кнопку и разрешите доступ к микрофону", "ГОЛОСОВОЙ АССИСТЕНТ ГОТОВ");
  if (resetSession && sessionId) {
    await fetch("/api/reset", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({session_id: sessionId})});
    sessionId = null; lastScenario = null; localStorage.removeItem("voice-router-session");
  }
}

function checkAudioSupport() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.AudioContext) {
    $("callButton").disabled = true; $("callButton").classList.add("unsupported");
    setCallCopy("Браузер не поддерживается", "Откройте страницу в Google Chrome или Microsoft Edge", "МИКРОФОН НЕДОСТУПЕН");
    $("callState").querySelector("span").textContent = "Запись аудио недоступна";
  }
}

fetch("/api/config").then(response => response.json()).then(config => {
  $("status").querySelector("span").textContent = config.llm_enabled ? "OpenAI подключён" : "Демо-режим";
  $("status").classList.toggle("warning", !config.llm_enabled);
}).catch(() => { $("status").querySelector("span").textContent = "Нет соединения"; $("status").classList.add("warning"); });

$("callButton").addEventListener("click", () => callActive ? endCall() : startCall());
$("reset").addEventListener("click", () => endCall({resetSession: true}));
checkAudioSupport();
