"""Host-only semantic/boundary checks; no containers, model or paid services."""

import copy
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

from tool_lab.generate import VERIFIER, make_task
from tool_lab.protocol import choose, digest, proposal_request, selector_payload, validate_proposal


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        namespace={'__name__':'verifier_test'};exec(VERIFIER,namespace)
        self.check=namespace['check']

    def fixture(self,family):
        p=self.root/family;make_task(p,family,101)
        return p/'environment/data',json.loads((p/'tests/expected.json').read_text())

    def test_noop_fails_all_families(self):
        for family in ('config','sqlite','report'):
            root,spec=self.fixture(family)
            self.assertFalse(self.check(root,spec))

    def test_scoped_config_and_protected_mutation(self):
        root,spec=self.fixture('config')
        p=root/'services.json';p.write_text(json.dumps(spec['files']['services.json']['value']))
        self.assertTrue(self.check(root,spec))
        (root/'reference.txt').write_text('damaged')
        self.assertFalse(self.check(root,spec))

    def test_json_type_and_duplicate_keys(self):
        root,spec=self.fixture('config')
        d=copy.deepcopy(spec['files']['services.json']['value']);d['services']['east']['enabled']=1
        p=root/'services.json';p.write_text(json.dumps(d))
        self.assertFalse(self.check(root,spec))
        p.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):self.check(root,spec)

    def test_sqlite_rows_schema_and_audit_are_protected(self):
        root,spec=self.fixture('sqlite');c=sqlite3.connect(root/'orders.db')
        c.execute("UPDATE orders SET status='ready' WHERE id IN (1,4)");c.commit()
        self.assertTrue(self.check(root,spec))
        c.execute("UPDATE orders SET status='ready' WHERE id=3");c.commit()
        self.assertFalse(self.check(root,spec));c.close()

    def test_report_requires_preserved_source(self):
        root,spec=self.fixture('report')
        (root/'report.json').write_text(json.dumps(spec['files']['report.json']['value']))
        self.assertTrue(self.check(root,spec))
        (root/'sales.csv').write_text('changed')
        self.assertFalse(self.check(root,spec))

    def test_symlink_and_reward_spoof_fail(self):
        root,spec=self.fixture('config')
        (root/'services.json').write_text(json.dumps(spec['files']['services.json']['value']))
        (root/'reward.txt').write_text('1')
        self.assertFalse(self.check(root,spec));(root/'reward.txt').unlink()
        (root/'reference.txt').unlink();(root/'reference.txt').symlink_to('/etc/passwd')
        self.assertFalse(self.check(root,spec))


class BoundaryTests(unittest.TestCase):
    def test_later_observations_cannot_change_an_earlier_request(self):
        history=[{'command':'cat data/a','stdout':'original'}]
        request=proposal_request('Inspect',history,1,6)
        original_hash=digest(request)
        history[0]['stdout']='changed'
        history.append({'command':'write data/a','stdout':'future'})
        self.assertEqual(request['history'],[{'command':'cat data/a','stdout':'original'}])
        self.assertEqual(digest(request),original_hash)

    def request(self):return {'instruction':'Inspect the file','history':[],'steps_remaining':3}
    def proposal(self):return {'request_sha256':digest(self.request()),'proposer':'test',
                              'candidates':[{'description':'Read','command':'cat data/a'},
                                            {'description':'List','command':'ls data'}]}
    def test_stale_and_duplicate_proposals_rejected(self):
        proposal=self.proposal();validate_proposal(proposal,self.request())
        with self.assertRaises(ValueError):validate_proposal(proposal,{**self.request(),'history':['changed']})
        proposal['candidates'][1]=proposal['candidates'][0]
        with self.assertRaises(ValueError):validate_proposal(proposal,self.request())

    def test_exact_commands_remain_visible_after_permutation(self):
        payload,mapping=selector_payload(self.request(),self.proposal()['candidates'],4)
        for key,item in mapping.items():
            if item['command']:self.assertIn(item['command'],payload['questions']['action']['criteria'][key])
        self.assertIn('finish',mapping)

    def test_sampler_and_invalid_probabilities(self):
        self.assertEqual(choose({'a':1.,'b':0.},['a','b'],'sample',random.Random(1)),'a')
        for p in ({'a':float('nan'),'b':0.},{'a':.5,'b':.2},{'a':1.}):
            with self.assertRaises(ValueError):choose(p,['a','b'],'argmax',random.Random(1))


if __name__=='__main__':unittest.main()
