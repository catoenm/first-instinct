"use strict";

(() => {
  const examples = {
    parcel: {
      title: "What should happen to this parcel?", kind: "choice", typeLabel: "Choose one",
      state: "Expedite a parcel if it is at least two days late. Otherwise, wait.\n\nThis parcel is three days late.",
      hint: "Try changing “three” to “one”.",
      instructions: "Which action follows the supplied policy: expedite the parcel or wait?",
      options: [{ id: "expedite", label: "Expedite", description: "Expedite the parcel" },
                { id: "wait", label: "Wait", description: "Wait without expediting" }],
    },
    support: {
      title: "Which team should handle this request?", kind: "choice", typeLabel: "Choose one",
      state: "Customer message:\n“I was charged twice for my monthly subscription. Please refund the extra payment.”",
      hint: "Try a message about a forgotten password or an app that keeps crashing.",
      instructions: "Which support team should handle this customer's request?",
      options: [{ id: "billing", label: "Billing", description: "Charges, invoices, payments, and refunds" },
                { id: "account", label: "Account access", description: "Passwords, sign-in, and account recovery" },
                { id: "technical", label: "Technical support", description: "Software bugs, crashes, and broken features" }],
    },
    access: {
      title: "Can Robin publish this release?", kind: "binary", typeLabel: "Yes or no",
      state: "Only active maintainers can publish a release.\n\nRobin is an active contributor, but is not a maintainer.",
      hint: "Try making Robin an active maintainer.",
      instructions: "Under the supplied rule, is Robin allowed to publish this release?",
      options: [{ id: "yes", label: "Yes" }, { id: "no", label: "No" }],
    },
    severity: {
      title: "How severe is this incident?", kind: "score", typeLabel: "Ordered scale",
      state: "Low: no users are blocked.\nMedium: some users are blocked, but a workaround exists.\nHigh: all users are blocked, with no workaround.\n\nAll users are unable to sign in. No workaround is available.",
      hint: "Try an incident where some users are blocked but a workaround exists.",
      instructions: "Using the supplied severity rules, choose the incident's severity level.",
      options: [{ id: "0", label: "Low", description: "Low: no users are blocked" },
                { id: "1", label: "Medium", description: "Medium: some users are blocked, but a workaround exists" },
                { id: "2", label: "High", description: "High: all users are blocked, with no workaround" }],
    },
  };
  const form = document.getElementById("demo");
  const input = document.getElementById("example");
  const button = document.getElementById("run");
  const reset = document.getElementById("reset");
  const runLabel = document.getElementById("run-label");
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  const modelNote = document.getElementById("model-note");
  const connection = document.getElementById("connection");
  const probabilityNote = document.getElementById("probability-note");
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const drafts = new Map();
  let selected = "parcel";
  let busy = false;

  function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle("error", error);
  }

  function renderOptions(probabilities = null, choice = null) {
    result.replaceChildren();
    for (const option of examples[selected].options) {
      const row = document.createElement("div");
      row.className = "answer";
      row.classList.toggle("selected", choice === option.id);
      const label = document.createElement("span");
      label.className = "answer-label";
      label.textContent = option.label;
      const value = document.createElement("span");
      value.textContent = probabilities ? percent(probabilities[option.id]) : "—";
      const progress = document.createElement("progress");
      progress.max = 1;
      progress.value = probabilities ? probabilities[option.id] : 0;
      progress.setAttribute("aria-hidden", "true");
      row.append(label, value, progress);
      result.append(row);
    }
    probabilityNote.textContent = probabilities
      ? "Probabilities over these answers, not guarantees."
      : "Run to see the model’s probabilities for these answers.";
  }

  function selectExample(key) {
    if (busy || !Object.hasOwn(examples, key)) return;
    drafts.set(selected, input.value);
    selected = key;
    const example = examples[key];
    input.value = drafts.get(key) ?? example.state;
    input.rows = key === "severity" ? 6 : 4;
    document.getElementById("question").textContent = example.title;
    document.getElementById("question-type").textContent = example.typeLabel;
    document.getElementById("hint").textContent = example.hint;
    document.getElementById("example-panel").setAttribute("aria-labelledby", `tab-${key}`);
    for (const tab of tabs) {
      const active = tab.dataset.example === key;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    renderOptions();
    message("");
  }

  for (const tab of tabs) {
    tab.addEventListener("click", () => selectExample(tab.dataset.example));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      const target = event.key === "ArrowRight" ? (index + 1) % tabs.length
        : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length
        : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
      if (target === null || busy) return;
      event.preventDefault();
      selectExample(tabs[target].dataset.example);
      tabs[target].focus();
    });
  }

  async function request(path, settings = {}) {
    let response;
    try {
      response = await fetch(path, { cache: "no-store", credentials: "same-origin", ...settings });
    } catch {
      throw new Error("Could not reach the model. Run again to retry.");
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong. Try again.");
    return data;
  }

  function describeModel(data) {
    const checkpoint = data.checkpoint;
    if (!checkpoint) throw new Error("The model’s training status is unavailable.");
    let description;
    if (checkpoint.kind === "preview") description = checkpoint.label;
    else if (checkpoint.kind === "reinforcement") description = `Reinforcement checkpoint / update ${checkpoint.step}`;
    else if (checkpoint.step === 0) description = "Supervised start / no reinforcement update selected";
    else description = `Supervised checkpoint / step ${checkpoint.step}`;
    modelNote.textContent = `${data.model.replace(/^Qwen\//, "")} · ${description}`;
    connection.textContent = "Model ready";
    connection.dataset.ready = "true";
  }

  async function connect() {
    const data = await request("/api/status");
    if (!data.ready || !data.csrf_token) throw new Error("The model is still loading. Try again shortly.");
    describeModel(data);
    return data;
  }

  function percent(value) {
    if (value > 0 && value < 0.001) return "<0.1%";
    if (value < 1 && value > 0.999) return ">99.9%";
    return `${(value * 100).toFixed(1)}%`;
  }

  input.addEventListener("input", () => { renderOptions(); message(""); });
  reset.addEventListener("click", () => {
    input.value = examples[selected].state;
    renderOptions();
    message("");
    input.focus();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const state = input.value.trim();
    if (!state) { message("Add some context first.", true); input.focus(); return; }
    const example = examples[selected];
    const question = { type: example.kind, instructions: example.instructions };
    if (example.kind === "choice") question.criteria = Object.fromEntries(example.options.map(option => [option.id, option.description]));
    if (example.kind === "score") question.criteria = example.options.map(option => option.description);
    busy = true;
    for (const control of [button, input, reset, ...tabs]) control.disabled = true;
    form.setAttribute("aria-busy", "true");
    runLabel.textContent = "Scoring…";
    renderOptions();
    message("Scoring the supplied answers…");
    try {
      const model = await connect();
      const response = await request("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": model.csrf_token },
        body: JSON.stringify({ state, questions: { decision: question } }),
      });
      const answer = response.answers?.decision;
      if (!answer || answer.type !== example.kind || example.options.some(({ id }) => {
        const value = answer.probabilities?.[id];
        return typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1;
      })) throw new Error("The model returned an incomplete answer. Try again.");
      const total = example.options.reduce((sum, option) => sum + answer.probabilities[option.id], 0);
      if (Math.abs(total - 1) > .001) throw new Error("The model returned invalid probabilities. Try again.");
      const best = example.options.reduce((a, b) => answer.probabilities[a.id] >= answer.probabilities[b.id] ? a : b);
      describeModel(response);
      renderOptions(answer.probabilities, best.id);
      const seconds = Number.isFinite(answer.milliseconds) ? ` / ${(answer.milliseconds / 1000).toFixed(1)}s` : "";
      message(`${best.label}${seconds}`);
    } catch (error) {
      message(error.message || "Could not run the example. Try again.", true);
    } finally {
      busy = false;
      for (const control of [button, input, reset, ...tabs]) control.disabled = false;
      form.setAttribute("aria-busy", "false");
      runLabel.textContent = "Run decision";
    }
  });

  renderOptions();
  connect().then(() => message("")).catch((error) => {
    message(error.message, true);
    modelNote.textContent = "The model is currently unavailable.";
    connection.textContent = "Offline";
    connection.dataset.ready = "false";
  }).finally(() => { button.disabled = false; });
})();
