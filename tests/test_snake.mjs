import test from "node:test";
import assert from "node:assert/strict";
import { newGame, actions, advance, preview, question, placeFood } from "../general_lab/web/snake.mjs";
const fixture = () => ({ size: 5, snake: [{ x: 2, y: 2 }, { x: 2, y: 3 }, { x: 2, y: 4 }], direction: "up", food: { x: 0, y: 0 }, score: 0, moves: 0, over: false, won: false, history: [] });
test("menus forbid reverse turns but retain losing model choices", () => {
  const game = fixture(); game.snake = [{ x: 2, y: 0 }, { x: 2, y: 1 }, { x: 2, y: 2 }];
  assert.deepEqual(actions(game), ["up", "right", "left"]);
  assert.equal(preview(game, "up").collision, "wall");
  const result = advance(game, "up");
  assert.equal(result.over, true); assert.equal(result.collision, "wall");
  assert.equal(result.direction, "up"); assert.deepEqual(actions(result), []);
  assert.throws(() => advance(game, "down"));
});
test("a normal move preserves length, grows history, and leaves its input intact", () => {
  const game = fixture(), before = JSON.stringify(game), result = advance(game, "right");
  assert.deepEqual(result.snake, [{ x: 3, y: 2 }, { x: 2, y: 2 }, { x: 2, y: 3 }]);
  assert.equal(result.moves, 1); assert.equal(result.score, 0);
  assert.equal(JSON.stringify(game), before); assert.equal(result.history[0].action, "right");
});
test("food grows the snake and is only placed in an empty cell", () => {
  const game = fixture(); game.food = { x: 2, y: 1 };
  const result = advance(game, "up", () => 0);
  assert.equal(result.snake.length, 4); assert.equal(result.score, 1);
  assert.deepEqual(result.food, { x: 0, y: 0 });
  assert.equal(result.snake.some(p => p.x === result.food.x && p.y === result.food.y), false);
});
test("the moving tail is free but another body segment is not", () => {
  const game = fixture(); game.snake = [{ x: 2, y: 2 }, { x: 2, y: 3 }, { x: 1, y: 3 }, { x: 1, y: 2 }];
  assert.equal(preview(game, "left").collision, null);
  assert.equal(advance(game, "left").over, false);
  game.snake.push({ x: 0, y: 2 });
  assert.equal(advance(game, "left").collision, "body");
});
test("filling the board terminates without requesting an impossible food location", () => {
  const game = { ...fixture(), size: 2, snake: [{ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }], food: { x: 1, y: 0 } };
  const result = advance(game, "right", () => { throw new Error("No draw expected"); });
  assert.equal(result.won, true); assert.equal(result.food, null);
  assert.equal(placeFood(result), null); assert.throws(() => question(result));
});
test("the prompt offers exactly the executable moves and truthful immediate observations", () => {
  const game = fixture(), payload = question(game);
  assert.deepEqual(Object.keys(payload.questions.move.criteria), actions(game));
  assert.equal(payload.state.board.split("\n")[2], "..H..");
  assert.match(payload.questions.move.criteria.up, /\(2,1\).*collision: none/);
  assert.match(payload.state.observation_note, /observations, not model forecasts/);
  const initial = newGame(10, () => .5);
  assert.equal(initial.snake.length, 3); assert.equal(initial.over, false);
});
