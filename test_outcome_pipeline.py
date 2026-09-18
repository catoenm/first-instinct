"""CPU fixtures only: no models, paid APIs, provider operations, or GPU work."""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_outcome_experiment as pipeline


def create(root, name, text='fixture'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def arguments(workspace):
    args = pipeline.parser().parse_args([
        '--project', str(workspace / 'project'), '--data', str(workspace / 'input/prepared'),
        '--raw-data', str(workspace / 'input/raw'), '--adapter', str(workspace / 'input/adapter'),
        '--freeze', str(workspace / 'project/results/outcome-v2/freeze.json'),
        '--output', str(workspace / 'output'), '--artifact', str(workspace / 'artifact.tar.gz'),
        '--deadline-file', str(workspace / 'private/deadline.json'), '--python', sys.executable])
    return args


def fixtures(args):
    for folder, name in ((args.project, 'general_lab/outcome_train.py'),
                         (args.project, 'results/outcome-v2/freeze.json'),
                         (args.data, 'manifest.json'), (args.raw_data, 'manifest.json'),
                         (args.adapter, 'adapter_config.json'), (args.adapter, 'adapter_model.safetensors')):
        create(folder, name, '{}')


class PlanningTests(unittest.TestCase):
    def test_fixed_recipe_two_seeds_and_full_bound(self):
        args = arguments(Path('/fixture'))
        now = 100000
        plan = pipeline.experiment_plan(args, now + 8 * 3600, now)
        self.assertTrue(plan['full_protocol_fits'])
        self.assertEqual(plan['workers_deadline'], now + 8 * 3600 - 45 * 60)
        self.assertEqual(plan['required_worker_seconds'], 3 * (7200 + 120 + 15))
        self.assertFalse(pipeline.experiment_plan(args, now + 6 * 3600, now)['full_protocol_fits'])
        self.assertEqual([(w['gpu'], w['seed']) for w in plan['workers']], [(0, 77), (1, 83)])
        for worker in plan['workers']:
            self.assertEqual([stage['arm'] for stage in worker['stages']], ['outcome', 'reward', 'hybrid'])
            for stage in worker['stages']:
                command = stage['command']
                self.assertEqual(command[:3], [sys.executable, '-m', 'general_lab.outcome_train'])
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(flags['--freeze'], str(args.freeze))
                self.assertEqual(flags['--seed'], str(worker['seed']))
                self.assertEqual(flags['--max-hours'], '2')
                self.assertEqual(flags['--learning-rate'], '3e-6')
                self.assertEqual(flags['--output'], str(args.output / 'runs' / stage['name']))

    def test_explicit_recipe_matches_trainer_defaults_without_importing_model_libraries(self):
        tree = ast.parse(Path('general_lab/outcome_train.py').read_text())
        arguments_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'arguments')
        defaults = {}
        for node in ast.walk(arguments_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    values = {key.arg: key.value for key in node.keywords}
                    if 'default' in values:
                        defaults[node.args[0].value] = ast.literal_eval(values['default'])
        recipe = dict(zip(pipeline.PROTOCOL_ARGS[::2], pipeline.PROTOCOL_ARGS[1::2]))
        self.assertEqual(set(recipe), set(defaults))
        for flag, value in recipe.items():
            self.assertEqual(type(defaults[flag])(value), defaults[flag], flag)

    def test_plan_reads_only_and_checks_bad_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deadline = create(root, 'deadline.json', json.dumps({'deadline': time.time() + 8 * 3600, 'provider_id': 'PROVIDER_SECRET_MARKER'}))
            before = set(root.rglob('*'))
            output = io.StringIO()
            with redirect_stdout(output), patch.object(pipeline.subprocess, 'Popen') as popen:
                self.assertEqual(pipeline.main(['--plan', '--project', str(root / 'project'),
                    '--output', str(root / 'output'), '--deadline-file', str(deadline)]), 0)
            popen.assert_not_called()
            self.assertEqual(set(root.rglob('*')), before)
            self.assertNotIn('PROVIDER_SECRET_MARKER', output.getvalue())
            self.assertEqual(len(json.loads(output.getvalue())['workers']), 2)
            for bad in (True, None, '1000', float('nan'), float('inf'), -1):
                deadline.write_text(json.dumps({'deadline': bad}))
                with self.assertRaises(ValueError):
                    pipeline.read_deadline(deadline)

    def test_environment_isolated_offline_and_secret_free(self):
        source = {'PATH': '/bin', 'HOME': '/root', 'LD_LIBRARY_PATH': '/cuda', 'LANG': 'C.UTF-8',
                  'OPENAI_API_KEY': 'secret', 'HF_TOKEN': 'secret', 'UNUSUAL_SERVICE_CREDENTIAL': 'secret',
                  'AWS_ACCESS_KEY_ID': 'secret', 'SSH_AUTH_SOCK': '/secret', 'LC_SECRET_TOKEN': 'secret',
                  'RANK': '4', 'LOCAL_RANK': '2', 'WORLD_SIZE': '8', 'MASTER_ADDR': 'host',
                  'TORCHELASTIC_RUN_ID': 'abc', 'OMPI_COMM_WORLD_RANK': '4', 'PMI_RANK': '4',
                  'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': '/untrusted', 'HF_ENDPOINT': 'https://bad'}
        env = pipeline.child_environment(1, source)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['HF_HOME'], '/opt/hf-cache')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/cuda')
        self.assertFalse(set(source) - {'PATH', 'HOME', 'LD_LIBRARY_PATH', 'LANG', 'CUDA_VISIBLE_DEVICES'} & set(env))


class ExecutionTests(unittest.TestCase):
    def test_permission_denied_probe_means_alive_until_confirmed_absent(self):
        with patch.object(pipeline.sys, 'platform', 'darwin'), patch.object(
                pipeline.os, 'killpg', side_effect=[PermissionError(1, 'Operation not permitted'), ProcessLookupError()]):
            self.assertTrue(pipeline.group_alive(1234))
            self.assertFalse(pipeline.group_alive(1234))

    def test_refuses_short_stage_without_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); args.output.mkdir()
            runner = pipeline.Pipeline(args, time.time() + 3600)
            with patch.object(pipeline.subprocess, 'Popen') as popen:
                self.assertFalse(runner.run_stage(runner.plan['workers'][0]['stages'][0], 0))
            popen.assert_not_called()
            self.assertTrue(runner.abort.is_set())
            self.assertEqual(json.loads((args.output / 'receipts/outcome-s77-exit.json').read_text())['status'], 'not_started')

    @unittest.skipUnless(os.name == 'posix', 'Process-group fixture requires POSIX')
    def test_child_failure_cancels_other_group_and_archives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); fixtures(args)
            marker = Path(directory) / 'other-worker-started'
            long_code = ('import os,signal,time; from pathlib import Path; '
                         'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                         f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)')
            fail_code = ('import sys,time; from pathlib import Path\n'
                         f'p=Path({str(marker)!r}); end=time.monotonic()+5\n'
                         'while not p.exists() and time.monotonic()<end: time.sleep(.02)\n'
                         'sys.exit(7)\n')
            def command(unused_args, arm, seed):
                return [sys.executable, '-c', fail_code if seed == 77 else long_code]
            with patch.object(pipeline, 'stage_command', command), patch.object(pipeline, 'TERM_GRACE_SECONDS', .15), \
                    patch.object(pipeline, 'KILL_GRACE_SECONDS', 2):
                runner = pipeline.Pipeline(args, time.time() + 8 * 3600)
                started = time.monotonic()
                result = runner.run(install_signals=False)
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['phase'], 'artifact_ready', json.dumps(result))
            self.assertEqual(set(result['stages']), {'outcome-s77', 'outcome-s83'})
            self.assertEqual(result['stages']['outcome-s77']['returncode'], 7)
            self.assertEqual(result['stages']['outcome-s83']['status'], 'cancelled', json.dumps(result))
            self.assertFalse(pipeline.group_alive(int(marker.read_text())))
            ready = json.loads((args.output / 'artifact-ready.json').read_text())
            self.assertEqual(ready['pipeline_status'], 'failed')
            with tarfile.open(ready['path']) as archive:
                state = json.load(archive.extractfile('pipeline.json'))
                self.assertEqual(state['status'], 'failed')
                self.assertTrue(state['errors'])
                self.assertIn('receipts/outcome-s83-exit.json', archive.getnames())

    def test_preflight_failure_still_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); args.project.mkdir()
            runner = pipeline.Pipeline(args, time.time() + 8 * 3600)
            with patch.object(pipeline.subprocess, 'Popen') as popen:
                result = runner.run(install_signals=False)
            popen.assert_not_called()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['phase'], 'artifact_ready')
            self.assertIn('Missing frozen inputs', result['errors'][0]['message'])

    def test_success_runs_all_six_stages_and_writes_ready_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); fixtures(args)
            def command(unused_args, arm, seed):
                return [sys.executable, '-c', f'print({arm!r}, {seed})']
            with patch.object(pipeline, 'stage_command', command), patch.object(pipeline, 'completed_run', return_value=True):
                result = pipeline.Pipeline(args, time.time() + 8 * 3600).run(install_signals=False)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['stages']), 6)
            for seed in (77, 83):
                stages = [result['stages'][f'{arm}-s{seed}'] for arm in pipeline.ARMS]
                self.assertTrue(all(a['ended'] <= b['started'] for a, b in zip(stages, stages[1:])))
                self.assertTrue(all(stage['returncode'] == 0 for stage in stages))
            ready = json.loads((args.output / 'artifact-ready.json').read_text())
            self.assertEqual(ready['pipeline_status'], 'complete')
            self.assertEqual(ready['sha256'], pipeline.sha256(args.artifact))


class ArchiveTests(unittest.TestCase):
    def test_allowlist_regular_files_and_every_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); fixtures(args)
            allowed = ['runs/outcome-s77/run.json', 'runs/outcome-s77/best/adapter_model.safetensors',
                       'runs/outcome-s77/best/adapter_config.json', 'runs/outcome-s77/tokenizer/tokenizer.json',
                       'runs/outcome-s77/tokenizer/special_tokens_map.json', 'runs/outcome-s77/tokenizer/added_tokens.json',
                       'runs/outcome-s77/tokenizer/tokenizer.model', 'runs/outcome-s77/training.jsonl',
                       'logs/outcome-s77.log', 'receipts/runtime.json', 'pipeline.json']
            forbidden = ['runs/outcome-s77/model.safetensors', 'runs/outcome-s77/optimizer.pt',
                         'runs/outcome-s77/foundation/config.json', 'runs/outcome-s77/data/train.jsonl',
                         'runs/outcome-s77/credentials.json', 'runs/outcome-s77/hf-token.json',
                         'runs/outcome-s77/.env', 'runs/outcome-s77/link.json']
            for name in allowed + forbidden[:-1]:
                create(args.output, name)
            private = create(Path(directory), 'private/deadline.json', '{"provider_id":"SECRET"}')
            (args.output / forbidden[-1]).symlink_to(private)
            (args.output / 'runs/outcome-s77/linked').symlink_to(private.parent, target_is_directory=True)
            create(args.project, 'docs/outcome-v2-protocol.md')
            create(args.project, 'scripts/run_outcome_experiment.py')
            create(args.project, 'provenance/dependency-versions.json')
            create(args.project, 'results/outcome-v2/regeneration.json')
            create(args.data, 'train.jsonl', 'RAW PRIVATE INPUT')
            create(args.raw_data, 'test-audit.jsonl', 'RAW PRIVATE INPUT')
            receipt = pipeline.make_archive(args, time.time() + 30)
            self.assertEqual(receipt['sha256'], pipeline.sha256(args.artifact))
            self.assertEqual(receipt['bytes'], args.artifact.stat().st_size)
            with tarfile.open(args.artifact) as archive:
                names = set(archive.getnames())
                self.assertTrue(set(allowed) <= names)
                self.assertFalse(set(forbidden) & names)
                self.assertTrue({'inputs/prepared-manifest.json', 'inputs/raw-manifest.json', 'protocol/freeze.json',
                                 'source/scripts/run_outcome_experiment.py', 'provenance/dependency-versions.json'} <= names)
                self.assertIn('source/results/outcome-v2/regeneration.json', names)
                self.assertTrue(all(item.isfile() for item in archive.getmembers()))
                hashes = json.load(archive.extractfile('artifact-hashes.json'))
                self.assertEqual(set(hashes), names - {'artifact-hashes.json'})
                for name, expected in hashes.items():
                    content = archive.extractfile(name).read()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected)
                    self.assertNotIn(b'RAW PRIVATE INPUT', content)
                    self.assertNotIn(b'SECRET', content)
            with self.assertRaises(FileExistsError):
                pipeline.make_archive(args, time.time() + 30)

    def test_expired_archive_fails_without_ready_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory)); args.output.mkdir()
            create(args.output, 'pipeline.json', '{}')
            with self.assertRaises(TimeoutError):
                pipeline.make_archive(args, time.time() - 1)
            self.assertFalse(args.artifact.exists())
            self.assertFalse((args.output / 'artifact-ready.json').exists())


if __name__ == '__main__':
    unittest.main()
