"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = {
    form: byId("decision-form"), state: byId("state"), stateCount: byId("state-count"),
    questions: byId("question-list"), addQuestion: byId("add-question"),
    questionCount: byId("question-count"), runSummary: byId("run-summary"),
    runButton: byId("run-button"), runLabel: byId("run-label"),
    loadExample: byId("load-example"), exampleMessage: byId("example-message"),
    error: byId("form-error"), connectionText: byId("connection-text"),
    connectionDot: byId("connection-dot"), retry: byId("retry-connection"),
    modelName: byId("model-name"), checkpoint: byId("checkpoint-label"),
    runtime: byId("runtime-label"), modelDetails: byId("model-details-list"),
    preview: byId("preview-notice"), inputLimit: byId("input-limit"),
    resultState: byId("result-state"), emptyResults: byId("empty-results"),
    resultMetadata: byId("result-metadata"), resultCount: byId("result-count"),
    answers: byId("answer-list"), stale: byId("stale-notice"),
    submitted: byId("submitted-input"), submittedJson: byId("submitted-json"),
    probabilityNote: byId("probability-note"),
  };
  const typeNames = { choice: "Choice", binary: "Binary", score: "Ordered score" };
  const controls = new Map();
  let limits = { max_questions: 32, max_options: 36, max_tokens: 1536 };
  let status = null;
  let busy = false;
  let exampleLoading = false;
  let editVersion = 0;
  let nextKey = 0;
  let lastRun = null;
  let draft = [newQuestion("question_1")];

  function element(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = String(text);
    return result;
  }

  function button(text, className, action, label) {
    const result = element("button", className, text);
    result.type = "button";
    if (label) result.setAttribute("aria-label", label);
    result.addEventListener("click", action);
    return result;
  }

  function field(labelText, input) {
    const wrapper = element("div", "field");
    const label = element("label", "field-label", labelText);
    label.htmlFor = input.id;
    wrapper.append(label, input);
    return wrapper;
  }

  function textInput(value, id, className, onInput, multiline = false) {
    const input = element(multiline ? "textarea" : "input", className);
    input.id = id;
    input.value = value;
    if (multiline) input.rows = 1;
    else input.type = "text";
    input.addEventListener("input", () => {
      onInput(input.value);
      markEdited();
    });
    return input;
  }

  function newQuestion(id, definition = {}) {
    return {
      key: ++nextKey,
      id,
      type: definition.type || "choice",
      instructions: definition.instructions || "",
      options: definition.type === "choice" && definition.criteria
        ? Object.entries(definition.criteria).map(([optionId, description]) => ({ id: optionId, description }))
        : [{ id: "option_1", description: "" }, { id: "option_2", description: "" }],
      levels: definition.type === "score" && Array.isArray(definition.criteria)
        ? [...definition.criteria] : ["", "", ""],
    };
  }

  function draftSignature() {
    return JSON.stringify({ state: ui.state.value, questions: draft.map(({ key, ...question }) => question) });
  }

  function markEdited() {
    editVersion += 1;
    clearError();
    ui.stateCount.textContent = `${ui.state.value.length.toLocaleString()} characters`;
    updateStaleNotice();
  }

  function updateStaleNotice() {
    ui.stale.hidden = !lastRun || lastRun.signature === draftSignature();
  }

  function clearError() {
    ui.error.hidden = true;
    ui.error.textContent = "";
    ui.form.querySelectorAll('[aria-invalid="true"]').forEach((input) => input.removeAttribute("aria-invalid"));
  }

  function showError(message, input) {
    ui.error.textContent = message;
    ui.error.hidden = false;
    if (input) {
      input.setAttribute("aria-invalid", "true");
      input.focus();
    } else {
      ui.error.focus();
    }
  }

  function updateCounts() {
    const count = draft.length;
    ui.questionCount.textContent = String(count);
    ui.runSummary.textContent = `${count} independent question${count === 1 ? "" : "s"}`;
    ui.addQuestion.disabled = count >= limits.max_questions;
    ui.addQuestion.title = count >= limits.max_questions ? `Maximum ${limits.max_questions} questions` : "";
  }

  function renderQuestions() {
    controls.clear();
    ui.questions.replaceChildren();
    draft.forEach((question, index) => {
      const questionControls = { optionIds: [], optionDescriptions: [], levels: [] };
      controls.set(question.key, questionControls);
      const card = element("article", "question-card");
      const header = element("div", "question-card-header");
      const heading = element("h3", "question-card-title", `Question ${String(index + 1).padStart(2, "0")}`);
      heading.id = `question-heading-${question.key}`;
      card.setAttribute("aria-labelledby", heading.id);
      const remove = button("Remove", "remove-question", () => {
        draft = draft.filter((entry) => entry.key !== question.key);
        markEdited();
        renderQuestions();
        const nextQuestion = draft[Math.min(index, draft.length - 1)];
        if (nextQuestion) controls.get(nextQuestion.key).id.focus();
      }, `Remove question ${index + 1}`);
      remove.disabled = draft.length === 1;
      remove.title = draft.length === 1 ? "Keep at least one question" : "";
      header.append(heading, element("span", "question-type-badge", typeNames[question.type]), remove);
      const body = element("div", "question-card-body");
      const fields = element("div", "question-fields");
      const idInput = textInput(question.id, `question-id-${question.key}`, "question-id", (value) => { question.id = value; });
      idInput.placeholder = "question_id";
      idInput.spellcheck = false;
      questionControls.id = idInput;
      const type = element("select", "question-type");
      type.id = `question-type-${question.key}`;
      Object.entries(typeNames).forEach(([value, label]) => {
        const option = element("option", "", label);
        option.value = value;
        type.append(option);
      });
      type.value = question.type;
      type.addEventListener("change", () => {
        question.type = type.value;
        markEdited();
        renderQuestions();
        controls.get(question.key).type.focus();
      });
      questionControls.type = type;
      fields.append(field("Question ID", idInput), field("Answer type", type));
      const instructions = textInput(question.instructions, `question-instructions-${question.key}`, "question-instructions", (value) => { question.instructions = value; }, true);
      instructions.rows = 2;
      instructions.placeholder = question.type === "binary"
        ? "State the proposition to assess using the supplied context…"
        : "What should the model decide from the supplied context?";
      questionControls.instructions = instructions;
      body.append(fields, field(question.type === "binary" ? "Proposition or yes/no question" : "Question instructions", instructions));
      if (question.type === "choice") renderChoices(question, body, questionControls);
      else if (question.type === "score") renderLevels(question, body, questionControls);
      else renderBinary(body);
      card.append(header, body);
      ui.questions.append(card);
    });
    updateCounts();
  }

  function definitionsHeading(title, count) {
    const heading = element("div", "definitions-heading");
    heading.append(element("span", "field-label", title), element("span", "definitions-count", `${count} / ${limits.max_options}`));
    return heading;
  }

  function renderChoices(question, body, questionControls) {
    body.append(definitionsHeading("Possible answers", question.options.length));
    const columnLabels = element("div", "option-column-labels");
    columnLabels.setAttribute("aria-hidden", "true");
    columnLabels.append(element("span", "", "Label / ID"), element("span", "", "What this answer means"));
    body.append(columnLabels);
    question.options.forEach((option, index) => {
      const row = element("div", "option-row");
      const id = textInput(option.id, `option-id-${question.key}-${index}`, "option-id", (value) => { option.id = value; });
      id.setAttribute("aria-label", `Option ${index + 1} label for ${question.id || "this question"}`);
      id.placeholder = "label";
      id.spellcheck = false;
      const description = textInput(option.description, `option-description-${question.key}-${index}`, "option-description", (value) => { option.description = value; }, true);
      description.setAttribute("aria-label", `Option ${index + 1} description for ${question.id || "this question"}`);
      description.placeholder = "Describe this answer…";
      const remove = button("×", "remove-option", () => {
        question.options.splice(index, 1);
        markEdited();
        renderQuestions();
        controls.get(question.key).optionDescriptions[Math.min(index, question.options.length - 1)].focus();
      }, `Remove option ${index + 1}`);
      remove.disabled = question.options.length <= 2;
      remove.title = remove.disabled ? "At least two answers are required" : "Remove answer";
      questionControls.optionIds.push(id);
      questionControls.optionDescriptions.push(description);
      row.append(id, description, remove);
      body.append(row);
    });
    const add = button("＋ Add answer", "add-definition", () => {
      let number = question.options.length + 1;
      const ids = new Set(question.options.map((option) => option.id.trim()));
      while (ids.has(`option_${number}`)) number += 1;
      question.options.push({ id: `option_${number}`, description: "" });
      markEdited();
      renderQuestions();
      controls.get(question.key).optionDescriptions[question.options.length - 1].focus();
    });
    add.disabled = question.options.length >= limits.max_options;
    body.append(add);
  }

  function renderLevels(question, body, questionControls) {
    body.append(definitionsHeading("Ordered levels", question.levels.length));
    body.append(element("p", "field-help level-help", "Positions run from 0 upward. Define what each level means; use the arrows to change the order."));
    question.levels.forEach((level, index) => {
      const row = element("div", "level-row");
      const position = element("span", "level-index", index);
      position.setAttribute("aria-hidden", "true");
      const description = textInput(level, `level-description-${question.key}-${index}`, "option-description", (value) => { question.levels[index] = value; }, true);
      description.setAttribute("aria-label", `Level ${index} description for ${question.id || "this question"}`);
      description.placeholder = `Describe level ${index}…`;
      const movers = element("div", "level-movers");
      [-1, 1].forEach((direction) => {
        const move = button(direction < 0 ? "↑" : "↓", "", () => {
          const destination = index + direction;
          [question.levels[index], question.levels[destination]] = [question.levels[destination], question.levels[index]];
          markEdited();
          renderQuestions();
          controls.get(question.key).levels[destination].focus();
        }, `Move level ${index} ${direction < 0 ? "up" : "down"}`);
        move.disabled = index + direction < 0 || index + direction >= question.levels.length;
        movers.append(move);
      });
      const remove = button("×", "remove-option", () => {
        question.levels.splice(index, 1);
        markEdited();
        renderQuestions();
        controls.get(question.key).levels[Math.min(index, question.levels.length - 1)].focus();
      }, `Remove level ${index}`);
      remove.disabled = question.levels.length <= 2;
      remove.title = remove.disabled ? "At least two levels are required" : "Remove level";
      questionControls.levels.push(description);
      row.append(position, description, movers, remove);
      body.append(row);
    });
    const add = button("＋ Add level", "add-definition", () => {
      question.levels.push("");
      markEdited();
      renderQuestions();
      controls.get(question.key).levels[question.levels.length - 1].focus();
    });
    add.disabled = question.levels.length >= limits.max_options;
    body.append(add);
  }

  function renderBinary(body) {
    const info = element("div", "binary-info");
    const labels = element("div", "binary-labels");
    labels.append(element("span", "", "yes"), element("span", "", "no"));
    info.append(labels, element("p", "", "The model assigns probability to the proposition being true (yes) or false (no), using the supplied state."));
    body.append(info);
  }

  class InputError extends Error {
    constructor(message, input) {
      super(message);
      this.input = input;
    }
  }

  function collectPayload() {
    const state = ui.state.value.trim();
    if (!state) throw new InputError("Add the shared state before running your questions.", ui.state);
    if (draft.length < 1 || draft.length > limits.max_questions) {
      throw new InputError(`Use between 1 and ${limits.max_questions} questions.`);
    }
    const questions = Object.create(null);
    for (let index = 0; index < draft.length; index += 1) {
      const question = draft[index];
      const inputs = controls.get(question.key);
      const id = question.id.trim();
      if (!id) throw new InputError(`Give question ${index + 1} an ID.`, inputs.id);
      if (Object.hasOwn(questions, id)) throw new InputError(`Question ID “${id}” is used twice. Give each question a unique ID.`, inputs.id);
      const instructions = question.instructions.trim();
      if (!instructions) throw new InputError(`Add instructions for “${id}”.`, inputs.instructions);
      const definition = { type: question.type, instructions };
      if (question.type === "choice") {
        if (question.options.length < 2 || question.options.length > limits.max_options) {
          throw new InputError(`“${id}” needs 2–${limits.max_options} possible answers.`, inputs.id);
        }
        const criteria = Object.create(null);
        question.options.forEach((option, optionIndex) => {
          const optionId = option.id.trim();
          if (!optionId) throw new InputError(`Add a label for answer ${optionIndex + 1} in “${id}”.`, inputs.optionIds[optionIndex]);
          if (Object.hasOwn(criteria, optionId)) throw new InputError(`Answer label “${optionId}” is used twice in “${id}”.`, inputs.optionIds[optionIndex]);
          const description = option.description.trim();
          if (!description) throw new InputError(`Describe answer “${optionId}” in “${id}”.`, inputs.optionDescriptions[optionIndex]);
          criteria[optionId] = description;
        });
        definition.criteria = criteria;
      } else if (question.type === "score") {
        if (question.levels.length < 2 || question.levels.length > limits.max_options) {
          throw new InputError(`“${id}” needs 2–${limits.max_options} ordered levels.`, inputs.id);
        }
        definition.criteria = question.levels.map((level, levelIndex) => {
          const description = level.trim();
          if (!description) throw new InputError(`Describe level ${levelIndex} in “${id}”.`, inputs.levels[levelIndex]);
          return description;
        });
      }
      questions[id] = definition;
    }
    return { state, questions };
  }

  async function request(path, options = {}) {
    let response;
    try { response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options }); }
    catch {
      const error = new Error("Could not reach the local model server. Retry the connection when the server is ready.");
      error.connectionLost = true;
      throw error;
    }
    let body;
    try { body = await response.json(); }
    catch { throw new Error("The server returned an unreadable response. Check the local server and try again."); }
    if (!response.ok) {
      const error = new Error(typeof body.error === "string" ? body.error : `The request failed (${response.status}). Try again.`);
      error.httpStatus = response.status;
      throw error;
    }
    return body;
  }

  function checkpointText(checkpoint) {
    if (!checkpoint) return "Checkpoint unavailable";
    return checkpoint.label || [checkpoint.kind, checkpoint.step == null ? "" : `step ${checkpoint.step}`].filter(Boolean).join(" · ") || "Checkpoint unavailable";
  }

  function isPreview(checkpoint) {
    return checkpoint && (checkpoint.kind === "preview" || (checkpoint.kind === "reinforcement" && checkpoint.step === 0));
  }

  function labeledCheckpoint(checkpoint) {
    const label = checkpointText(checkpoint);
    return isPreview(checkpoint) && !/^training preview\b/i.test(label) ? `Training preview · ${label}` : label;
  }

  function addDetail(label, value) {
    ui.modelDetails.append(element("dt", "", label), element("dd", "", value == null || value === "" ? "Unavailable" : value));
  }

  function renderStatus() {
    ui.modelName.textContent = status.model || "Model unavailable";
    ui.checkpoint.textContent = checkpointText(status.checkpoint);
    ui.checkpoint.classList.toggle("preview", Boolean(isPreview(status.checkpoint)));
    ui.runtime.textContent = status.device || "Unavailable";
    ui.modelDetails.replaceChildren();
    addDetail("Model", status.model);
    addDetail("Revision", status.model_revision);
    addDetail("Checkpoint", checkpointText(status.checkpoint));
    addDetail("Run status", status.checkpoint?.run_status);
    addDetail("Checkpoint kind", status.checkpoint?.kind);
    addDetail("Training step", status.checkpoint?.step);
    addDetail("Device", status.device);
    ui.preview.hidden = !isPreview(status.checkpoint);
    ui.preview.textContent = isPreview(status.checkpoint)
      ? `${labeledCheckpoint(status.checkpoint)}. This checkpoint is experimental; it does not establish performance or calibration on your questions.` : "";
    ui.inputLimit.textContent = `${limits.max_tokens.toLocaleString()} tokens maximum per question, including state and answer definitions. No truncation.`;
  }

  function setConnection(text, kind = "") {
    ui.connectionText.textContent = text;
    ui.connectionDot.className = `connection-dot ${kind}`.trim();
  }

  async function connect() {
    ui.retry.hidden = true;
    ui.runButton.disabled = true;
    setConnection("Connecting to model…");
    try {
      const response = await request("/api/status");
      if (!response.ready || typeof response.csrf_token !== "string" || !response.csrf_token) {
        throw new Error(response.error || "Model is not ready");
      }
      status = response;
      const previousLimits = JSON.stringify(limits);
      if (response.limits) {
        for (const key of Object.keys(limits)) {
          if (Number.isInteger(response.limits[key]) && response.limits[key] > 0) limits[key] = response.limits[key];
        }
      }
      renderStatus();
      if (JSON.stringify(limits) !== previousLimits) renderQuestions();
      else updateCounts();
      ui.runButton.disabled = busy;
      setConnection(busy ? "Running questions…" : "Model ready", busy ? "busy" : "ready");
      if (!busy) ui.resultState.textContent = lastRun ? "Connected. Previous results remain below." : "";
    } catch {
      status = null;
      ui.runButton.disabled = true;
      ui.retry.hidden = false;
      setConnection("Server unavailable", "error");
      ui.modelName.textContent = "Waiting for local model server";
      ui.resultState.textContent = "The model may still be loading. Retry the connection when the local server is ready.";
    }
  }

  function validateExample(example) {
    if (!example || typeof example !== "object" || !Object.hasOwn(example, "state") || !example.questions || typeof example.questions !== "object" || Array.isArray(example.questions)) {
      throw new Error("The example has an invalid format.");
    }
    const entries = Object.entries(example.questions);
    if (entries.length < 1 || entries.length > limits.max_questions) throw new Error("The example has an invalid question count.");
    for (const [, question] of entries) {
      if (!question || !Object.hasOwn(typeNames, question.type) || typeof question.instructions !== "string") throw new Error("The example has an invalid question.");
      if (question.type === "choice" && (!question.criteria || typeof question.criteria !== "object" || Array.isArray(question.criteria) || Object.values(question.criteria).some((value) => typeof value !== "string"))) throw new Error("The example has invalid answer definitions.");
      if (question.type === "score" && (!Array.isArray(question.criteria) || question.criteria.some((value) => typeof value !== "string"))) throw new Error("The example has invalid ordered levels.");
    }
    return entries;
  }

  async function loadExample(initial = false) {
    if (exampleLoading) return;
    exampleLoading = true;
    ui.loadExample.disabled = true;
    ui.loadExample.textContent = "Loading…";
    const versionAtStart = editVersion;
    try {
      const example = await request("/api/example");
      const entries = validateExample(example);
      // Do not overwrite work entered while the example request was in flight.
      if (editVersion !== versionAtStart) {
        ui.exampleMessage.textContent = "Kept your edits. Load the example again to replace the input.";
        return;
      }
      ui.state.value = typeof example.state === "string" ? example.state : JSON.stringify(example.state, null, 2);
      draft = entries.map(([id, definition]) => newQuestion(id, definition));
      markEdited();
      renderQuestions();
      ui.exampleMessage.textContent = "Example loaded. Edit the state and questions to make it yours.";
      if (!initial) ui.state.focus();
    } catch (error) {
      ui.exampleMessage.textContent = initial ? "Start with your own context or load an example." : "Could not load the example. Your input is unchanged.";
      if (!initial) showError(error.message || "Could not load the example.");
    } finally {
      exampleLoading = false;
      ui.loadExample.disabled = false;
      ui.loadExample.textContent = "Load example ↙";
    }
  }

  function setBusy(value) {
    busy = value;
    ui.runButton.disabled = value || !status;
    ui.runButton.setAttribute("aria-busy", String(value));
    ui.runButton.querySelector(".button-spinner").hidden = !value;
    ui.runButton.querySelector(".run-arrow").hidden = value;
    ui.runLabel.textContent = value ? "Running…" : "Run questions";
    ui.answers.setAttribute("aria-busy", String(value));
    if (status) setConnection(value ? "Running questions…" : "Model ready", value ? "busy" : "ready");
    ui.resultState.classList.toggle("loading", value);
  }

  function validProbability(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  }

  function validateResponse(response, payload) {
    if (!response || !response.answers || typeof response.answers !== "object") throw new Error("The model returned an incomplete answer. Try again.");
    for (const [id, question] of Object.entries(payload.questions)) {
      if (!Object.hasOwn(response.answers, id)) throw new Error(`The model did not return an answer for “${id}”.`);
      const answer = response.answers[id];
      const keys = question.type === "binary" ? ["yes", "no"] : Object.keys(question.criteria);
      if (!answer || answer.type !== question.type || !answer.probabilities || keys.some((key) => !Object.hasOwn(answer.probabilities, key) || !validProbability(answer.probabilities[key]))) {
        throw new Error(`The model returned an invalid probability distribution for “${id}”.`);
      }
      if (question.type === "choice" && !keys.includes(answer.choice)) throw new Error(`The model returned an unknown answer for “${id}”.`);
      if (question.type === "binary" && !validProbability(answer.probability_yes)) throw new Error(`The model returned an invalid yes probability for “${id}”.`);
      if (question.type === "score" && (typeof answer.score !== "number" || !Number.isFinite(answer.score) || answer.score < 0 || answer.score > keys.length - 1 || !Number.isInteger(answer.selected_level) || !keys.includes(String(answer.selected_level)))) {
        throw new Error(`The model returned an invalid ordered score for “${id}”.`);
      }
    }
  }

  function percent(value) {
    if (value > 0 && value < 0.001) return "<0.1%";
    if (value < 1 && value > 0.999) return ">99.9%";
    return `${(value * 100).toFixed(1)}%`;
  }

  function distributionRow(id, description, probability, selected) {
    const row = element("div", selected ? "distribution-row selected" : "distribution-row");
    const label = element("div", "distribution-label");
    const name = element("span", "distribution-name");
    name.append(element("strong", "", id));
    if (selected) name.append(document.createTextNode(" · highest probability"));
    label.append(name, element("span", "distribution-probability", percent(probability)));
    const bar = element("div", "distribution-bar");
    bar.setAttribute("aria-hidden", "true");
    const fill = element("span", "distribution-fill");
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    bar.append(fill);
    row.append(label, bar);
    if (description) row.append(element("span", "distribution-detail", description));
    return row;
  }

  function renderAnswer(id, question, answer) {
    const card = element("article", "answer-card");
    const heading = element("div", "answer-card-header");
    heading.append(element("h3", "answer-id", id), element("span", "answer-kind", typeNames[question.type]));
    card.append(heading, element("p", "answer-instructions", question.instructions));
    const summary = element("div", "answer-summary");
    let selected;
    let entries;
    if (question.type === "choice") {
      selected = answer.choice;
      summary.append(element("span", "summary-label", "Selected answer"), element("div", "answer-main-value", selected), element("p", "answer-description", question.criteria[selected]));
      entries = Object.entries(question.criteria);
    } else if (question.type === "binary") {
      selected = answer.probabilities.yes >= answer.probabilities.no ? "yes" : "no";
      summary.append(element("span", "summary-label", "Probability of yes"), element("div", "answer-main-value numeric", percent(answer.probability_yes)), element("p", "answer-description", "Model probability assigned to the yes answer."));
      entries = [["yes", "The proposition is true"], ["no", "The proposition is false"]];
    } else {
      selected = String(answer.selected_level);
      const value = element("div", "answer-main-value numeric", answer.score.toFixed(2));
      value.append(element("span", "value-suffix", `/ ${question.criteria.length - 1}`));
      summary.append(element("span", "summary-label", "Expected level index"), value,
        element("p", "answer-description", `Most likely: level ${selected} · ${question.criteria[answer.selected_level]}`),
        element("p", "score-explanation", "Probability-weighted average of the level positions, starting at 0."));
      entries = question.criteria.map((description, index) => [String(index), description]);
    }
    const distribution = element("div", "distribution");
    distribution.setAttribute("aria-label", "Answer probabilities");
    entries.forEach(([key, description]) => {
      distribution.append(distributionRow(question.type === "score" ? `Level ${key}` : key, description, answer.probabilities[key], key === selected));
    });
    card.append(summary, distribution);
    if (typeof answer.milliseconds === "number" && Number.isFinite(answer.milliseconds) && answer.milliseconds >= 0) {
      const duration = answer.milliseconds >= 1000 ? `${(answer.milliseconds / 1000).toFixed(2)} s` : `${Math.round(answer.milliseconds)} ms`;
      card.append(element("p", "answer-timing", `${duration} model inference`));
    }
    return card;
  }

  function renderResults(run) {
    const { response, payload, serverStatus, completedAt } = run;
    ui.answers.replaceChildren();
    Object.entries(payload.questions).forEach(([id, question]) => ui.answers.append(renderAnswer(id, question, response.answers[id])));
    const count = Object.keys(payload.questions).length;
    ui.resultCount.textContent = String(count);
    ui.resultCount.hidden = false;
    ui.emptyResults.hidden = true;
    ui.resultMetadata.replaceChildren();
    const model = response.model || serverStatus.model;
    const checkpoint = response.checkpoint || serverStatus.checkpoint;
    ui.resultMetadata.append(
      element("span", "", `${count} answer${count === 1 ? "" : "s"} · ${completedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`),
      element("span", "", model || "Model unavailable"),
      element("span", "result-checkpoint", labeledCheckpoint(checkpoint)),
    );
    ui.resultMetadata.hidden = false;
    ui.submittedJson.textContent = JSON.stringify(payload, null, 2);
    ui.submitted.hidden = false;
    if (typeof response.probability_note === "string" && response.probability_note) ui.probabilityNote.textContent = response.probability_note;
    updateStaleNotice();
  }

  async function runQuestions(event) {
    event.preventDefault();
    if (busy || !status) return;
    clearError();
    let payload;
    try { payload = collectPayload(); }
    catch (error) { showError(error.message, error.input); return; }
    const signature = draftSignature();
    const serverStatus = status;
    setBusy(true);
    ui.resultState.textContent = lastRun
      ? "Running the submitted questions. Previous results remain below until this run finishes."
      : "Running the submitted questions. Each is processed independently; larger inputs can take longer.";
    try {
      const response = await request("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": serverStatus.csrf_token },
        body: JSON.stringify(payload),
      });
      validateResponse(response, payload);
      lastRun = { response, payload, signature, serverStatus, completedAt: new Date() };
      renderResults(lastRun);
      ui.resultState.textContent = "Run complete. Answers are shown below.";
    } catch (error) {
      showError(error.message || "Could not run the model. Check the server and try again.");
      ui.resultState.textContent = lastRun ? "This run failed. The previous results are still shown below." : "No results yet. Resolve the error and run again.";
      if (error.connectionLost || error.httpStatus === 403) {
        status = null;
        ui.retry.hidden = false;
        setConnection("Connection needs retry", "error");
      }
    } finally {
      setBusy(false);
      updateStaleNotice();
    }
  }

  ui.state.addEventListener("input", () => {
    ui.exampleMessage.textContent = "The same state is supplied to every question.";
    markEdited();
  });
  ui.addQuestion.addEventListener("click", () => {
    if (draft.length >= limits.max_questions) return;
    let number = draft.length + 1;
    const ids = new Set(draft.map((question) => question.id.trim()));
    while (ids.has(`question_${number}`)) number += 1;
    const question = newQuestion(`question_${number}`);
    draft.push(question);
    markEdited();
    renderQuestions();
    controls.get(question.key).instructions.focus();
  });
  ui.loadExample.addEventListener("click", () => loadExample());
  ui.retry.addEventListener("click", connect);
  ui.form.addEventListener("submit", runQuestions);
  renderQuestions();
  // Both endpoints are read-only and independent. No model request runs on load.
  void Promise.allSettled([connect(), loadExample(true)]);
})();
