import { DIRECTIONS, newGame, actions, advance, question } from "/snake.mjs";
const $ = id => document.getElementById(id);
const canvas = $("board"), context = canvas.getContext("2d");
let game = newGame(), running = false, busy = false, generation = 0, token = null, controller = null;
let best = 0;
function draw() {
  const unit = canvas.width / game.size;
  context.fillStyle = "#e2e6cc"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#c7cdb3";
  for (let y = 0; y < game.size; y++) for (let x = 0; x < game.size; x++) {
    context.fillRect(x * unit + unit / 2 - 1, y * unit + unit / 2 - 1, 2, 2);
  }
  if (game.food) {
    context.fillStyle = "#ad442b";
    context.fillRect(game.food.x * unit + 8, game.food.y * unit + 8, unit - 16, unit - 16);
  }
  for (let i = game.snake.length - 1; i >= 0; i--) {
    const cell = game.snake[i]; context.fillStyle = i ? "#667452" : "#293f30";
    context.fillRect(cell.x * unit + 2, cell.y * unit + 2, unit - 4, unit - 4);
  }
  const head = game.snake[0], d = DIRECTIONS[game.direction];
  context.fillStyle = "#e2e6cc";
  for (const side of [-1, 1]) {
    const x = (head.x + .5) * unit + d.x * 9 + (d.x ? 0 : side * 7);
    const y = (head.y + .5) * unit + d.y * 9 + (d.y ? 0 : side * 7);
    context.fillRect(x - 2, y - 2, 4, 4);
  }
  $("score").textContent = String(game.score).padStart(2, "0");
  $("moves").textContent = String(game.moves).padStart(3, "0");
  $("best").textContent = String(best).padStart(2, "0");
  canvas.setAttribute("aria-label", `Snake board. Score ${game.score}, move ${game.moves}. Head at ${head.x}, ${head.y}. ${game.over ? "Game finished." : `Food at ${game.food.x}, ${game.food.y}.`}`);
  $("start").textContent = running ? "Pause" : game.over ? "Play again" : game.moves ? "Resume" : "Start model";
  $("step").disabled = busy || running || game.over;
  $("overlay").hidden = !game.over;
  $("end-title").textContent = game.won ? "Board cleared." : "Game over.";
  $("end-note").textContent = `${game.score} food · ${game.moves} moves${game.collision ? ` · ${game.collision} collision` : ""}`;
}
function message(text, error = false) { $("status").textContent = text; $("status").classList.toggle("error", error); }
function disconnected() {
  token = null;
  $("connection").textContent = "Offline"; $("connection").dataset.ready = "false";
  $("model-note").textContent = "Model offline · reconnect to play";
}
function probabilities(values = null, chosen = null, menu = actions(game)) {
  $("choices").replaceChildren();
  for (const action of menu) {
    const row = document.createElement("div"), label = document.createElement("span"), value = document.createElement("span"), bar = document.createElement("progress");
    row.className = "answer"; row.classList.toggle("selected", chosen === action);
    label.textContent = `${DIRECTIONS[action].arrow} ${DIRECTIONS[action].label}`;
    value.textContent = values ? `${(values[action] * 100).toFixed(1)}%` : "—";
    bar.max = 1; bar.value = values ? values[action] : 0; bar.setAttribute("aria-hidden", "true");
    row.append(label, value, bar); $("choices").append(row);
  }
}
async function request(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", credentials: "same-origin", ...options });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The model is unavailable.");
  return body;
}
async function connect(signal) {
  const result = await request("/api/status", { signal });
  if (!result.ready || !result.csrf_token) throw new Error("The model is still loading.");
  token = result.csrf_token;
  $("connection").textContent = "Model ready"; $("connection").dataset.ready = "true";
  $("model-note").textContent = `${result.model.replace(/^Qwen\//, "")} · ${result.checkpoint.label}`;
}
function reset() {
  running = false; generation++; controller?.abort(); game = newGame();
  $("latency").textContent = "—"; $("input-preview").textContent = JSON.stringify(question(game), null, 2);
  probabilities(); draw(); message("The model chooses. The game executes.");
}
async function tick() {
  if (busy || game.over) return;
  const epoch = generation, offered = actions(game), payload = question(game);
  busy = true; controller = new AbortController();
  const timeout = setTimeout(() => controller?.abort(), 60000);
  draw(); message(game.moves ? "Choosing the next move…" : "Choosing the first move…");
  try {
    if (!token) await connect(controller.signal);
    const result = await request("/api/answer", { method: "POST", signal: controller.signal,
      headers: { "Content-Type": "application/json", "X-CSRF-Token": token }, body: JSON.stringify(payload) });
    if (epoch !== generation) return;
    const answer = result.answers?.move;
    if (!answer || answer.type !== "choice" || !offered.includes(answer.choice) || offered.some(key => !Number.isFinite(answer.probabilities?.[key]) || answer.probabilities[key] < 0 || answer.probabilities[key] > 1)
      || Math.abs(offered.reduce((sum, key) => sum + answer.probabilities[key], 0) - 1) > .001) throw new Error("The model returned an invalid move.");
    // Execute exactly the returned choice. No heuristic fallback or collision veto.
    game = advance(game, answer.choice); best = Math.max(best, game.score);
    $("input-preview").textContent = JSON.stringify(payload, null, 2);
    probabilities(answer.probabilities, answer.choice, offered);
    $("latency").textContent = Number.isFinite(answer.milliseconds) ? `${Math.round(answer.milliseconds)} ms` : "—";
    message(game.over ? (game.won ? "Every cell filled." : `The model hit its ${game.collision === "body" ? "body" : "wall"}.`) : `${DIRECTIONS[answer.choice].arrow} ${DIRECTIONS[answer.choice].label}`);
    if (game.over) running = false;
  } catch (error) {
    if (epoch === generation) {
      running = false; disconnected();
      message(error.name === "AbortError" ? "Paused. Press Resume to try again." : error.message || "Could not reach the model.", true);
    }
  } finally {
    clearTimeout(timeout); busy = false; controller = null; draw();
    if (running) setTimeout(() => { if (running) tick(); }, 220);
  }
}
$("start").addEventListener("click", () => {
  if (running) { running = false; draw(); return; }
  if (game.over) reset();
  running = true; draw(); if (!busy) tick();
});
$("step").addEventListener("click", tick);
$("reset").addEventListener("click", reset);
document.addEventListener("visibilitychange", () => { if (document.hidden) { running = false; draw(); } });
reset();
connect().catch(() => { disconnected(); message("Connect to the model, then press Start.", true); });
