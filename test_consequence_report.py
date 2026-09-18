import json
from pathlib import Path
import tempfile
import unittest

from puffer_lab.consequence_report import matched_predictions, replay_diagnostics
from puffer_lab.forecast_probe import selected, question
from puffer_lab.native import NativeEpisode, compile_core, library
from puffer_lab.text_render import cases, render
from scale_lab.common import write_rows


class ConsequenceReportTests(unittest.TestCase):
    def test_predictions_reject_wrong_ids_and_nonfinite_probabilities(self):
        rows = [dict(id='one', option_ids=['yes','no'])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.jsonl'
            for prediction in [dict(id='other', probabilities=dict(yes=.2,no=.8)),
                               dict(id='one', probabilities=dict(yes=float('nan'),no=.8)),
                               dict(id='one', probabilities=dict(yes=.2,no=.4))]:
                path.write_text(json.dumps(prediction)+'\n')
                with self.assertRaises(ValueError):
                    matched_predictions(rows,path)
            path.write_text(json.dumps(dict(id='one', probabilities=dict(no=.8,yes=.2)))+'\n')
            self.assertEqual(matched_predictions(rows,path),[[.2,.8]])

    def test_executed_trajectory_and_forecast_replay_detects_state_corruption(self):
        # Real deterministic environment execution, with a deliberately simple
        # synthetic policy. No model or network is required.
        case = next(c for c in cases() if c['world']==2)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); lib=library(compile_core(folder/'core.so'))
            transitions=[];history=[];total=0.
            with NativeEpisode(lib,case['world'],case['profile']) as env:
                while not env.state()['done']:
                    item=render(env.public(),history,'original')
                    action='sequential'
                    p={o['id']:float(o['id']=='m3') for o in item['options']}
                    reward=env.step(action);total+=reward
                    transitions.append(dict(case=case['case'],variant='original',input=item,
                                            probabilities=p,action=action,reward=reward,state=env.state(),public=env.public()))
                    history.append(dict(action=action,result=env.public()['last_result']))
                episode=dict(**case,variant='original',actions=history,success=int(env.state()['outcome']==1))
                episode['return']=total
            write_rows(folder/'transitions.jsonl',transitions)
            write_rows(folder/'episodes.jsonl',[episode])
            forecasts=[]
            for target in selected():
                q=target['success_probability'][0]/target['success_probability'][1]
                for order in (False,True):
                    forecasts.append(dict(input=question(target,order),q=q,reversed=order,
                                          probabilities=dict(success=q,failure=1-q)))
            write_rows(folder/'forecasts.jsonl',forecasts)
            spec=dict(cases=[case],variants=['original'])
            report=replay_diagnostics(folder,spec)
            self.assertEqual(report['replayed_transitions'],len(transitions))
            self.assertEqual(report['forecast']['False/all']['mse'],0)
            transitions[0]['state']['a']+=1
            write_rows(folder/'transitions.jsonl',transitions)
            with self.assertRaises(ValueError):replay_diagnostics(folder,spec)


if __name__=='__main__':unittest.main()
