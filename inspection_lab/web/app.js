const $ = id => document.getElementById(id);
let session = null;
const names = {initial:'Initial check', examples:'Upstream examples', probes:'Perturbed inputs', copy:'Exact copy of the initial check'};
const percent = p => (p * 100).toFixed(1);
async function request(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed');
  return value;
}
function busy(value) {
  document.querySelectorAll('button,select').forEach(el => el.disabled = value);
  if (!session) $('finish').disabled = true;
}
function render(result) {
  $('workspace').hidden = false;
  const forecast = result.history.at(-1);
  $('probability').textContent = percent(forecast.probability);
  $('meter').value = forecast.probability;
  $('raw').textContent = `Raw model: ${percent(forecast.raw_probability)}%. Validation-adjusted forecast shown above. Report on the environment's grid: ${percent(forecast.report_probability)}%.`;
  $('history').replaceChildren();
  result.history.forEach((f,i) => {
    const line = document.createElement('div'); line.className = 'history-line';
    const title = document.createElement('span'); title.textContent = i ? `After inspection ${i}` : 'Before inspection';
    const value = document.createElement('span'); value.textContent = `${percent(f.probability)}%`;
    line.append(title,value); $('history').append(line);
  });
  $('cost').textContent = `Inspection cost so far: ${result.total_cost.toFixed(2)} reward units.`;
  $('result').hidden = !result.finished;
  if (result.finished) {
    session = null;
    $('outcome').textContent = result.outcome ? 'The complete suite passed.' : 'At least one check failed.';
    $('reward').textContent = `Reported ${percent(result.reported_probability)}% chance of passing. Final reward: ${result.reward.toFixed(3)}.`;
    $('inspections').replaceChildren(); $('finish').disabled = true; $('policy-panel').hidden = true;
    $('status').textContent = 'Outcome revealed. Start another episode to explore a different program.';
    return;
  }
  session = result.session;
  $('policy-panel').hidden = !result.recommendation;
  if (result.recommendation) {
    const choice = result.recommendation.choice;
    $('recommendation').textContent = `Small inspector recommends: ${choice === 'finish' ? 'commit the forecast' : names[choice].toLowerCase()}.`;
  }
  const observation = result.observation;
  $('function').textContent = observation.function;
  $('contract').textContent = observation.contract;
  $('code').textContent = observation.code;
  $('evidence').replaceChildren();
  result.evidence_display.forEach(item => {
    const section = document.createElement('section'); section.className = 'evidence-group';
    const title = document.createElement('h3'); title.textContent = names[item.source];
    section.append(title);
    item.checks.forEach(check => {
      const block = document.createElement('div'); block.className = 'check';
      const verdict = document.createElement('strong'); verdict.textContent = check.passed ? 'PASS' : 'FAIL'; verdict.className = check.passed ? 'pass' : 'fail';
      const call = document.createElement('code'); call.textContent = check.call;
      const values = document.createElement('pre'); values.textContent = `Expected: ${check.expected}\nReturned: ${check.actual}`;
      block.append(verdict,call,values); section.append(block);
    });
    $('evidence').append(section);
  });
  $('inspections').replaceChildren();
  observation.inspections.forEach(item => {
    const button = document.createElement('button');
    button.textContent = `${names[item.action]} · ${item.cost.toFixed(2)}`;
    button.title = item.description;
    button.addEventListener('click',() => action(item.action));
    $('inspections').append(button);
  });
  $('status').textContent = `${forecast.input_tokens.toLocaleString()} input tokens · ${(forecast.milliseconds/1000).toFixed(2)} s for this local request · ${observation.purchases_remaining} inspections remaining`;
}
async function action(name) {
  busy(true); $('status').textContent = name === 'finish' ? 'Opening the sealed outcome…' : 'Reading the new evidence and forecasting…';
  try { render(await request('/api/step',{session,action:name})); }
  catch (error) { $('status').textContent = error.message; }
  finally { busy(false); }
}
$('start').addEventListener('click',async () => {
  busy(true); $('status').textContent = 'Reading the program and its first check…';
  try { render(await request('/api/start',{id:$('candidate').value})); }
  catch (error) { $('status').textContent = error.message; }
  finally { busy(false); }
});
$('finish').addEventListener('click',() => action('finish'));
$('policy').addEventListener('click',() => action('policy'));
request('/api/catalog').then(data => {
  data.cases.forEach(item => { const option = document.createElement('option'); option.value = item.id; option.textContent = item.function === 'solution' ? `${item.function} — ${item.summary}` : item.function; $('candidate').append(option); });
  $('status').textContent = `${data.model} · ${data.cases.length} held-out functions · ready locally`;
}).catch(error => { $('status').textContent = error.message; });
