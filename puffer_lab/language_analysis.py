"""Read-only follow-through: exact tokens and matched previously saved controls."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import tempfile

from scale_lab.common import encode, file_hash
from .native import NativeEpisode, compile_core, library
from .contract import digest
from .text_baseline import ROOT, analyze, read, require, rows, summarize, tokenizer, write
from .text_render import cases


def check_tokens(records, tok):
    lengths = []
    for row in records:
        actual = encode(tok, row['input'], 1536)
        require(actual == row['input_ids'], f'Saved tokenizer output differs at index {row["index"]}')
        lengths.append(len(actual))
    return {'questions': len(lengths), 'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths),
            'all_token_ids_reproduced': True}


def existing_numeric_controls():
    """Select the exact matching cases from the existing study; do not query weights."""
    records, sources = [], {}
    with tempfile.TemporaryDirectory() as temporary:
        lib = library(compile_core(Path(temporary) / 'core.so'))
        for seed in (41, 73):
            folder = ROOT / f'results/puffer-reservation-v1/learning/seed-{seed}'
            available = []
            for cohort in ('familiar', 'combined'):
                path = folder / f'selected-{cohort}.json'
                sources[str(path.relative_to(ROOT))] = file_hash(path)
                available.extend(read(path)['rows'])
            sources[str((folder / 'weights.json').relative_to(ROOT))] = file_hash(folder / 'weights.json')
            for case in cases():
                matches = [r for r in available if r['world'] == case['world'] and
                           all(r['profile'][k] == case['profile'][k] for k in ('costs', 'horizon', 'prior'))]
                require(len(matches) == 1, 'Require one exactly matching previously evaluated numeric case')
                recorded = matches[0]
                with NativeEpisode(lib, case['world'], case['profile']) as episode:
                    total = sum(episode.step(action) for action in recorded['actions'])
                    require(total == recorded['return'] and episode.state()['outcome'] == recorded['outcome'],
                            'Previously saved numeric control no longer replays')
                    records.append({**case, 'variant': f'numeric-seed-{seed}', 'actions': recorded['actions'],
                                    'return': total, 'success': recorded['success'], 'state': episode.state()})
    return {'source_sha256': sources, 'summary': summarize(records),
            'scope': 'Exact profile/world subset of previously published diagnostics; no new model inference. Numeric and textual policies receive different representations.'}


def public_mistakes(folder):
    """Post-hoc descriptive checks of observed facts, not an optimal-action oracle."""
    counts = defaultdict(Counter)
    for request, transition in zip(rows(Path(folder) / 'requests.jsonl'), rows(Path(folder) / 'transitions.jsonl')):
        observation = request['public_observation']
        state = observation['request']
        action = transition['action']
        complete = bool(state['header'] and state['A'] == state['B'] == 1)
        any_request = bool(state['header'] or state['A'] or state['B'])
        flags = {
            'finish_incomplete': action == 'finish' and not complete,
            'finish_partial_with_scoped_undo_available': action == 'finish' and any_request and not complete,
            'reserve_existing_request': action in ('atomic', 'sequential') and bool(state['header']),
            'undo_empty_request': action == 'undo' and not any_request,
            'inspect_already_known_stock': action == 'inspect_stock' and min(observation['stock'].values()) >= 0,
            'inspect_already_known_account': action == 'inspect_account' and observation['account'] >= 0,
            'create_known_existing_account': action == 'create_account' and observation['account'] == 1,
            'replenish_known_positive_stock': action in ('replenish_a', 'replenish_b') and
                                             observation['stock'][action[-1].upper()] > 0,
            'continue_after_known_completion': action != 'finish' and complete,
        }
        counts[request['variant']]['decisions'] += 1
        counts[request['variant']].update({name: int(value) for name, value in flags.items()})
    return {'counts': {name: dict(values) for name, values in counts.items()},
            'scope': 'Post-hoc public-fact diagnostics; categories may overlap. Finishing incomplete can be appropriate when recovery is impossible; it is not automatically an error.'}


def unique_contexts(folder):
    observed = defaultdict(dict)
    duplicate_count = changed_choices = 0
    durations = []
    for request, response, transition in zip(rows(Path(folder) / 'requests.jsonl'), rows(Path(folder) / 'responses.jsonl'),
                                           rows(Path(folder) / 'transitions.jsonl')):
        key = digest({'observation': request['public_observation'], 'history': request['history']})
        variant = request['variant']
        durations.append(response['milliseconds'])
        if variant in observed[key]:
            duplicate_count += 1
            changed_choices += observed[key][variant]['action'] != transition['action']
        else:
            observed[key][variant] = {'action': transition['action'], 'initial': not request['history']}
    comparisons = {}
    for variant in ('reversed', 'reworded'):
        paired = [group for group in observed.values() if 'original' in group and variant in group]
        initial = [group for group in paired if group['original']['initial']]
        comparisons[variant] = {'unique_matched_histories': len(paired), 'unique_initial_histories': len(initial),
                                'unique_initial_action_agreement': sum(g['original']['action'] == g[variant]['action'] for g in initial) / len(initial),
                                'unique_matched_action_agreement': sum(g['original']['action'] == g[variant]['action'] for g in paired) / len(paired)}
    return {'comparisons': comparisons, 'duplicate_presentations': duplicate_count,
            'duplicate_action_changes': changed_choices, 'median_model_milliseconds': statistics.median(durations),
            'scope': 'Post-hoc deduplication by full public history, taking the earliest identical presentation. Histories within episodes remain correlated.'}


def report(language_folder, forecast_folder=None):
    language_folder = Path(language_folder)
    result = {'language': analyze(language_folder), 'numeric_controls': existing_numeric_controls(),
              'public_fact_diagnostics': public_mistakes(language_folder), 'unique_public_contexts': unique_contexts(language_folder)}
    tok = tokenizer()
    result['language_tokens'] = check_tokens(rows(language_folder / 'requests.jsonl'), tok)
    if forecast_folder:
        from .forecast_probe import analyze as forecast_analysis
        forecast_folder = Path(forecast_folder)
        result['forecast'] = forecast_analysis(forecast_folder)
        result['forecast_tokens'] = check_tokens(read(forecast_folder / 'questions.json'), tok)
    result['analysis_source_sha256'] = file_hash(__file__)
    result['training'] = False
    result['model_updated'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', type=Path, required=True)
    parser.add_argument('--forecast', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, report(args.language, args.forecast))
    print(args.output)
