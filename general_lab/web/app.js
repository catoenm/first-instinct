"use strict";

(() => {
  const form = document.getElementById("demo");
  const input = document.getElementById("example");
  const button = document.getElementById("run");
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  const modelNote = document.getElementById("model-note");
  const options = { expedite: "Expedite the parcel", wait: "Wait without expediting" };
  let busy = false;

  function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle("error", error);
  }

  async function request(path, settings = {}) {
    let response;
    try {
      response = await fetch(path, { cache: "no-store", credentials: "same-origin", ...settings });
    } catch {
      throw new Error("Could not reach the model. Click Run to try again.");
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong. Try again.");
    return data;
  }

  function describeModel(data) {
    const checkpoint = data.checkpoint;
    if (!checkpoint) throw new Error("The model’s training status is unavailable.");
    let description;
    if (checkpoint.kind === "preview") {
      description = `${checkpoint.label}. Reinforcement-learning results are still pending.`;
    } else if (checkpoint.kind === "reinforcement") {
      description = `This demo uses a reinforcement-trained checkpoint at update ${checkpoint.step}.`;
    } else if (checkpoint.step === 0) {
      description = "This demo uses the supervised starting checkpoint, selected before any reinforcement updates.";
    } else {
      description = "This demo uses the supervised checkpoint.";
    }
    modelNote.textContent = `${data.model.replace(/^Qwen\//, "")} · ${description}`;
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

  input.addEventListener("input", () => {
    result.hidden = true;
    message("");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const state = input.value.trim();
    if (!state) { message("Add an example first.", true); input.focus(); return; }
    busy = true;
    button.disabled = true;
    input.disabled = true;
    button.textContent = "Running…";
    result.hidden = true;
    message("");
    try {
      // Refresh the token and model identity in case the resident model changed.
      const model = await connect();
      const response = await request("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": model.csrf_token },
        body: JSON.stringify({ state, questions: { decision: {
          type: "choice",
          instructions: "Which action follows the supplied policy: expedite the parcel or wait?",
          criteria: options,
        } } }),
      });
      const answer = response.answers?.decision;
      if (!answer || !Object.hasOwn(options, answer.choice) || Object.keys(options).some((key) => {
        const value = answer.probabilities?.[key];
        return typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1;
      })) throw new Error("The model returned an incomplete answer. Try again.");
      describeModel(response);
      for (const key of Object.keys(options)) {
        document.getElementById(`${key}-probability`).textContent = percent(answer.probabilities[key]);
        document.getElementById(`${key}-label`).parentElement.classList.toggle("selected", key === answer.choice);
      }
      result.hidden = false;
      const seconds = Number.isFinite(answer.milliseconds) ? ` · ${(answer.milliseconds / 1000).toFixed(1)}s` : "";
      message(`${answer.choice === "expedite" ? "Expedite" : "Wait"}${seconds}`);
    } catch (error) {
      message(error.message || "Could not run the example. Try again.", true);
    } finally {
      busy = false;
      input.disabled = false;
      button.disabled = false;
      button.textContent = "Run";
    }
  });

  connect().then(() => message("")).catch((error) => {
    message(error.message, true);
    modelNote.textContent = "The local model is currently unavailable.";
  }).finally(() => { button.disabled = false; });
})();
