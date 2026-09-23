const $ = id => document.getElementById(id);
let sessionId = localStorage.getItem("voice-router-session") || null;
let callActive = false, processing = false, mediaStream = null, mediaRecorder = null;
let audioContext = null, analyser = null, monitorFrame = null, currentAudio = null, chunks = [];
let callStartedAt = null, timerId = null, lastScenario = null;
let utteranceEndedAt = null, lastTraceData = null, lastTimings = {};

const formatTime = (date = new Date()) => date.toLocaleTimeString("ru-RU", {hour: "2-digit", minute: "2-digit"});
const escapeHtml = value => String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const ms = value => Number.isFinite(value) ? `${Math.round(value)} ms` : "—";
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
  if (!data.trace.needs_clarification && !data.trace.closed_scenarios?.includes(scenario.scenario_id)) {
    if (lastScenario !== scenario.scenario_id) addScenarioSwitch(scenario.scenario_id);
    lastScenario = scenario.scenario_id;
  }
  const state = $("scenarioState"); state.querySelector(".scenario-number").textContent = scenario.scenario_id;
  state.querySelector("strong").textContent = data.trace.needs_clarification
    ? "Маршрут требует уточнения"
    : `${Math.round(scenario.confidence * 100)}% · оценка модели`;
  state.querySelector("small").textContent = scenario.reason || "Активный маршрут разговора";
  $("scenarioConfidence").hidden = false; $("confidenceValue").style.width = `${Math.round(scenario.confidence * 100)}%`;
  $("messages").scrollTop = $("messages").scrollHeight;
}

function updateLatency(timings = {}) {
  lastTimings = {...lastTimings, ...timings};
  const trace = lastTraceData?.trace;
  $("sttLatency").textContent = ms(lastTimings.stt);
  $("routerLatency").textContent = ms(trace?.latency_ms?.router);
  $("scenarioLatency").textContent = ms(trace?.latency_ms?.response);
  $("ttsLatency").textContent = ms(lastTimings.tts);
  const pipeline = Number(lastTimings.stt || 0) + Number(trace?.latency_ms?.total || 0) + Number(lastTimings.tts || 0);
  const realTotal = lastTimings.endedAt && lastTimings.audioStartedAt ? lastTimings.audioStartedAt - lastTimings.endedAt : pipeline;
  $("totalLatency").textContent = ms(realTotal);
}

function renderSupervisor(data, timings = {}) {
  lastTraceData = data;
  lastTimings = timings;
  const trace = data.trace;
  $("traceEmpty").hidden = true;
  $("traceContent").hidden = false;
  $("traceTurn").textContent = `#${trace.turn}`;
  const alert = $("confidenceAlert");
  alert.className = `confidence-alert ${trace.confidence_band || "high"}`;
  alert.querySelector("span").textContent = trace.confidence_band === "low"
    ? "Низкая уверенность — требуется переспрос или оператор"
    : trace.confidence_band === "medium"
      ? "Средняя уверенность — робот уточняет запрос"
      : "Высокая уверенность маршрутизации";
  alert.title = "Оценки модели не являются статистически откалиброванными вероятностями.";
  if (!trace.needs_clarification && trace.uncertain_scenarios?.length) {
    alert.querySelector("span").textContent = `Нужно проверить дополнительные намерения: ${trace.uncertain_scenarios.join(", ")}`;
  }
  if (trace.uncertainty_reasons?.includes("close_alternatives")) {
    alert.querySelector("span").textContent = `Близкие альтернативы · разница ${Math.round(trace.ambiguity_gap * 100)} п.п. — нужен переспрос`;
  }
  if (trace.router_meta?.path === "local-fast-path" && !trace.needs_clarification) {
    alert.querySelector("span").textContent = "Fast path — маршрут выбран локально без ожидания LLM";
  }
  $("multiIntentBadge").hidden = !trace.multi_intent;
  $("traceScenarios").innerHTML = trace.scenarios.map(item => `
    <article class="trace-scenario">
      <div class="trace-scenario-head"><span class="trace-scenario-id">${escapeHtml(item.scenario_id)}</span><span class="trace-scenario-name">${escapeHtml(item.name)}</span><span class="trace-scenario-confidence">${Math.round(item.confidence * 100)}%</span></div>
      <div class="trace-bar"><i style="width:${Math.round(item.confidence * 100)}%"></i></div>
      <p>${escapeHtml(item.reason)}</p>
    </article>`).join("");
  $("traceAlternatives").innerHTML = trace.alternatives.length
    ? trace.alternatives.map(item => `<div class="trace-alternative"><span><b>${escapeHtml(item.scenario_id)}</b> · ${escapeHtml(item.name)}<br>${escapeHtml(item.why_rejected || "")}</span><b>${Math.round(item.confidence * 100)}%</b></div>`).join("")
    : "Нет альтернатив";
  const context = [];
  if (trace.is_continuation) context.push('<span class="context-chip">Продолжение сценария</span>');
  for (const id of trace.dialog_state.active_scenarios || []) context.push(`<span class="context-chip">Активен: ${escapeHtml(id)}</span>`);
  for (const id of trace.dialog_state.pending_queue || []) context.push(`<span class="context-chip queue">В очереди: ${escapeHtml(id)}</span>`);
  for (const group of trace.dialog_state.stack || []) {
    for (const id of group) context.push(`<span class="context-chip">Приостановлен: ${escapeHtml(id)}</span>`);
  }
  for (const id of trace.closed_scenarios || []) context.push(`<span class="context-chip">Закрыт: ${escapeHtml(id)}</span>`);
  $("contextState").innerHTML = context.join("") || '<span class="trace-muted">Контекст пуст</span>';
  const slotEntries = Object.entries(trace.slots || {});
  $("traceSlots").innerHTML = slotEntries.length ? slotEntries.map(([key, value]) => `<span class="trace-chip">${escapeHtml(key)}: ${escapeHtml(value)}</span>`).join(" ") : "Не извлечены";
  $("traceActions").innerHTML = trace.actions.length ? trace.actions.map(item => `<span class="trace-chip">${escapeHtml(item.name)} · ${escapeHtml(item.mode)}</span>`).join(" ") : "Нет действий";
  $("traceLanguage").textContent = `Язык: ${trace.language}`;
  const routerPath = trace.router_meta?.path || trace.mode;
  const tier = trace.router_meta?.service_tier || "default";
  $("traceMode").textContent = `${routerPath} · ${tier}`;
  if (data.handoff) {
    alert.className = "confidence-alert low";
    alert.querySelector("span").textContent = "Передача оператору: " + data.handoff_summary;
  }
  updateLatency(timings);
}

function browserSpeak(text, language) {
  return new Promise(resolve => {
    if (!window.speechSynthesis || !callActive) return resolve();
    const utterance = new SpeechSynthesisUtterance(text); utterance.lang = language === "kk" ? "kk-KZ" : "ru-RU"; utterance.rate = .96;
    utterance.onend = resolve; utterance.onerror = resolve; speechSynthesis.speak(utterance);
  });
}

async function speak(text, language) {
  if (!callActive) return {};
  const ttsStartedAt = performance.now();
  let audioStartedAt = null;
  setVoiceState("speaking", "OpenAI озвучивает ответ"); setCallCopy("Отвечаю", "После ответа можете продолжить говорить");
  try {
    const response = await fetch("/api/speech", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text, language})});
    if (!response.ok) throw new Error("OpenAI TTS unavailable");
    const blob = await response.blob(), url = URL.createObjectURL(blob);
    await new Promise(resolve => {
      currentAudio = new Audio(url);
      currentAudio.onplaying = () => { audioStartedAt ??= performance.now(); };
      currentAudio.onended = resolve; currentAudio.onerror = resolve; currentAudio.play().catch(resolve);
    });
    URL.revokeObjectURL(url); currentAudio = null;
  } catch (_) {
    audioStartedAt = performance.now();
    await browserSpeak(text, language);
  }
  return {tts: (audioStartedAt || performance.now()) - ttsStartedAt, audioStartedAt: audioStartedAt || performance.now()};
}

async function routeTranscript(text, timings = {}, allowIdle = false) {
  if (!text.trim() || processing || (!callActive && !allowIdle)) return;
  processing = true;
  $("textInput").disabled = true; $("textSend").disabled = true;
  addTranscript("client", text.trim());
  if (callActive) { setVoiceState("processing", "Определяю сценарий"); setCallCopy("Обрабатываю", "LLM выбирает нужный сценарий разговора"); }
  try {
    const response = await fetch("/api/route", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text: text.trim(), session_id: sessionId})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Ошибка маршрутизации");
    sessionId = data.session_id; localStorage.setItem("voice-router-session", sessionId);
    renderScenario(data); renderSupervisor(data, timings); addTranscript("assistant", data.reply);
    if (callActive) {
      const speechTiming = await speak(data.reply, data.trace.language);
      updateLatency(speechTiming);
    }
  } catch (error) { addTranscript("error", error.message); }
  finally {
    processing = false; $("textInput").disabled = false; $("textSend").disabled = false;
    if (callActive) { setVoiceState("listening", "Слушаю вас"); setCallCopy("Я слушаю", "Говорите свободно — сценарий переключится автоматически"); startRecording(); }
  }
}

async function transcribeRecording(blob, endedAt) {
  if (!callActive || !blob.size) return;
  processing = true; setVoiceState("processing", "OpenAI распознаёт речь"); setCallCopy("Распознаю", "Преобразую голос в текст");
  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      headers: {"Content-Type": blob.type || "audio/webm", "X-Language-Hint": $("speechLang").value},
      body: blob
    });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Ошибка распознавания речи");
    processing = false;
    if (data.text) await routeTranscript(data.text, {stt: data.latency_ms, endedAt});
    else if (callActive) startRecording();
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
    else if (voiceStarted) { silenceStartedAt ??= now; if (now - silenceStartedAt > 1100) { utteranceEndedAt = performance.now(); mediaRecorder.voiceDetected = true; mediaRecorder.stop(); return; } }
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
    if (!mediaRecorder.voiceDetected) { startRecording(); return; }
    const blob = new Blob(chunks, {type: mediaRecorder.mimeType || "audio/webm"});
    if (blob.size > 1000) transcribeRecording(blob, utteranceEndedAt || performance.now()); else startRecording();
  };
  mediaRecorder.start(250); monitorVoice();
}

function pauseRecording() {
  cancelAnimationFrame(monitorFrame);
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.onstop = () => {};
    mediaRecorder.stop();
  }
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
    $("traceContent").hidden = true; $("traceEmpty").hidden = false; $("traceTurn").textContent = "—";
    $("messages").innerHTML = '<div id="emptyTranscript" class="empty-transcript"><span class="empty-icon"><i></i><i></i><i></i><i></i><i></i></span><p>Текст разговора появится здесь после начала звонка</p></div>';
    const scenarioState = $("scenarioState"); scenarioState.querySelector(".scenario-number").textContent = "—";
    scenarioState.querySelector("strong").textContent = "Ожидание звонка"; scenarioState.querySelector("small").textContent = "Маршрут определится автоматически";
    $("scenarioConfidence").hidden = true;
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
$("textFallback").addEventListener("submit", event => {
  event.preventDefault();
  const text = $("textInput").value.trim();
  if (!text || processing) return;
  $("textInput").value = "";
  if (callActive) pauseRecording();
  routeTranscript(text, {stt: 0, endedAt: performance.now()}, true);
});
checkAudioSupport();
