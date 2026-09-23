const $ = (id) => document.getElementById(id);
let sessionId = localStorage.getItem("voice-router-session") || null;
let recognition = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}

function message(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function speak(text, language) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = language === "kk" ? "kk-KZ" : "ru-RU";
  speechSynthesis.speak(utterance);
}

function renderTrace(data) {
  const t = data.trace;
  $("empty").hidden = true;
  $("trace").hidden = false;
  $("language").textContent = t.language;
  $("latency").textContent = `${t.latency_ms.router} ms`;
  $("mode").textContent = t.mode.toUpperCase();
  $("scenarios").innerHTML = t.scenarios.map(s => `
    <article class="scenario">
      <div><b>${escapeHtml(s.scenario_id)}</b><span>${Math.round(s.confidence * 100)}%</span></div>
      <div class="bar"><i style="width:${Math.round(s.confidence * 100)}%"></i></div>
      <p>${escapeHtml(s.reason)}</p>
    </article>`).join("");
  $("alternatives").innerHTML = t.alternatives.length
    ? t.alternatives.map(a => `<div class="alternative"><b>${escapeHtml(a.scenario_id)}</b> · ${Math.round(a.confidence * 100)}% — ${escapeHtml(a.why_rejected)}</div>`).join("")
    : "Нет";
  $("slots").textContent = JSON.stringify(t.slots, null, 2);
  $("actions").innerHTML = t.actions.length
    ? t.actions.map(a => `<span class="chip">${escapeHtml(a.name)} · ${escapeHtml(a.mode)}</span>`).join("")
    : "Нет";
  $("handoff").hidden = !data.handoff;
  $("handoff").textContent = data.handoff ? `Передача оператору: ${data.handoff_summary}` : "";
}

async function send() {
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  message("client", text);
  $("send").disabled = true;
  $("send").textContent = "Думаю…";
  try {
    const response = await fetch("/api/route", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, session_id: sessionId})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Ошибка маршрутизации");
    sessionId = data.session_id;
    localStorage.setItem("voice-router-session", sessionId);
    message("bot", data.reply);
    renderTrace(data);
    speak(data.reply, data.trace.language);
  } catch (error) {
    message("error", error.message);
  } finally {
    $("send").disabled = false;
    $("send").textContent = "Отправить";
    $("input").focus();
  }
}

async function reset() {
  if (sessionId) {
    await fetch("/api/reset", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({session_id: sessionId})});
  }
  sessionId = null;
  localStorage.removeItem("voice-router-session");
  $("messages").innerHTML = '<div class="message bot">Здравствуйте! Опишите, пожалуйста, ваш вопрос.</div>';
  $("trace").hidden = true;
  $("empty").hidden = false;
}

function setupMic() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    $("mic").disabled = true;
    $("mic").title = "Браузер не поддерживает Speech Recognition";
    return;
  }
  recognition = new SpeechRecognition();
  recognition.lang = $("speechLang").value;
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.onstart = () => $("mic").classList.add("listening");
  recognition.onend = () => $("mic").classList.remove("listening");
  recognition.onresult = (event) => {
    $("input").value = Array.from(event.results).map(r => r[0].transcript).join(" ");
    if (event.results[event.results.length - 1].isFinal) send();
  };
  $("mic").onclick = () => {
    recognition.lang = $("speechLang").value;
    recognition.start();
  };
}

fetch("/api/config").then(r => r.json()).then(config => {
  $("status").textContent = config.llm_enabled ? `${config.model} · ${config.scenario_count} сценариев` : "DEMO · добавьте OPENAI_API_KEY";
  $("status").classList.toggle("warning", !config.llm_enabled);
});
$("send").onclick = send;
$("reset").onclick = reset;
$("input").addEventListener("keydown", e => { if (e.ctrlKey && e.key === "Enter") send(); });
setupMic();
