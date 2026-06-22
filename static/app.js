const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const quickButtons = document.querySelectorAll("[data-question]");

function appendMessage(text, role, meta = "", sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);

  if (meta) {
    const small = document.createElement("span");
    small.className = "message-meta";
    small.textContent = meta;
    article.appendChild(small);
  }

  if (sources.length > 0) {
    const sourceLine = document.createElement("div");
    sourceLine.className = "message-sources";
    sourceLine.textContent = `Sources: ${sources.join(", ")}`;
    article.appendChild(sourceLine);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  document.querySelector("#neo4jStatus").textContent = status.neo4j ? "Connected" : "Missing";
  document.querySelector("#groqStatus").textContent = status.groq ? "Ready" : "API key needed";
  document.querySelector("#vectorStatus").textContent = status.vector ? "Ready" : "Run ingest";
  document.querySelector("#modeStatus").textContent = status.mode;
}

async function sendMessage(message) {
  appendMessage(message, "user");
  input.value = "";
  input.focus();

  const pending = appendMessage("Thinking...", "bot");
  const submitButton = form.querySelector("button");
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: "auto",
      }),
    });
    const data = await response.json();
    pending.remove();

    if (!response.ok) {
      appendMessage(data.error || "Something went wrong.", "error");
      return;
    }

    appendMessage(data.answer, "bot", data.mode, data.sources || []);
  } catch (error) {
    pending.remove();
    appendMessage(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (message) {
    sendMessage(message);
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

quickButtons.forEach((button) => {
  button.addEventListener("click", () => {
    sendMessage(button.dataset.question);
  });
});

refreshStatus();
