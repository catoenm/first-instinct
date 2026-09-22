// Deterministic game rules. This module never chooses a move for the model.
export const DIRECTIONS = {
  up: { x: 0, y: -1, label: "Up", arrow: "↑" },
  right: { x: 1, y: 0, label: "Right", arrow: "→" },
  down: { x: 0, y: 1, label: "Down", arrow: "↓" },
  left: { x: -1, y: 0, label: "Left", arrow: "←" },
};
const opposite = { up: "down", down: "up", left: "right", right: "left" };
const same = (a, b) => a.x === b.x && a.y === b.y;
export function placeFood(game, random = Math.random) {
  const empty = [];
  for (let y = 0; y < game.size; y++) for (let x = 0; x < game.size; x++) {
    if (!game.snake.some(cell => same(cell, { x, y }))) empty.push({ x, y });
  }
  if (!empty.length) return null;
  const draw = random();
  if (!(draw >= 0 && draw < 1)) throw new Error("Invalid food random draw");
  return empty[Math.floor(draw * empty.length)];
}
export function newGame(size = 10, random = Math.random) {
  if (!Number.isInteger(size) || size < 5 || size > 16) throw new Error("Invalid board size");
  const middle = Math.floor(size / 2);
  const game = { size, snake: [{ x: middle, y: middle }, { x: middle, y: middle + 1 }, { x: middle, y: middle + 2 }],
    direction: "up", food: null, score: 0, moves: 0, over: false, won: false, history: [] };
  game.food = placeFood(game, random);
  return game;
}
export function actions(game) {
  return game.over ? [] : Object.keys(DIRECTIONS).filter(key => key !== opposite[game.direction]);
}
export function preview(game, action) {
  if (!actions(game).includes(action)) throw new Error("Move was not offered");
  const delta = DIRECTIONS[action];
  const next = { x: game.snake[0].x + delta.x, y: game.snake[0].y + delta.y };
  const eats = game.food !== null && same(next, game.food);
  // The tail vacates its cell on a non-growing move.
  const occupied = eats ? game.snake : game.snake.slice(0, -1);
  const collision = next.x < 0 || next.y < 0 || next.x >= game.size || next.y >= game.size ? "wall"
    : occupied.some(cell => same(cell, next)) ? "body" : null;
  return { next, eats, collision, distance: game.food ? Math.abs(next.x - game.food.x) + Math.abs(next.y - game.food.y) : 0 };
}
export function advance(game, action, random = Math.random) {
  const predicted = preview(game, action);
  const next = { ...game, moves: game.moves + 1, direction: action,
    history: [...game.history, { action, head: { ...game.snake[0] } }].slice(-6) };
  if (predicted.collision) return { ...next, over: true, collision: predicted.collision };
  next.snake = [predicted.next, ...game.snake];
  if (predicted.eats) { next.score++; next.food = placeFood(next, random); }
  else next.snake.pop();
  if (next.food === null) { next.over = true; next.won = true; }
  return next;
}
export function question(game) {
  if (game.over) throw new Error("Finished games have no next action");
  const board = Array.from({ length: game.size }, () => Array(game.size).fill("."));
  if (game.food) board[game.food.y][game.food.x] = "*";
  for (const cell of game.snake) board[cell.y][cell.x] = "o";
  board[game.snake[0].y][game.snake[0].x] = "H";
  const criteria = Object.fromEntries(actions(game).map(action => {
    const p = preview(game, action);
    return [action, `Move ${action} to (${p.next.x},${p.next.y}). Immediate collision: ${p.collision ?? "none"}. Eats food: ${p.eats ? "yes" : "no"}. Manhattan distance to food after moving: ${p.distance}.`];
  }));
  return {
    state: { game: "Snake", goal: "Eat food and survive as long as possible. Avoid walls, your body, and loops.",
      coordinates: "x increases right; y increases down. Coordinates start at zero. Walls surround the board.",
      board: board.map(row => row.join("")).join("\n"), legend: "H=head, o=body, *=food, .=empty",
      head: game.snake[0], body_head_to_tail: game.snake, food: game.food, current_direction: game.direction,
      recent_moves: game.history, rules: "Each move advances one cell. Eating grows the snake. Otherwise the tail moves away. Reversing direction is disallowed. Collision ends the game.",
      observation_note: "Option descriptions report geometry computed by the game. They are observations, not model forecasts." },
    questions: { move: { type: "choice", instructions: "Choose the next direction. Prefer a safe route toward food while leaving room to escape. Never choose an immediate collision if a safe move exists. Avoid repeating a loop. Return the best supplied move.", criteria } },
  };
}
