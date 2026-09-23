import unittest

from release_lab.startup_watchdog import watch, validate


class StartupWatchdogTests(unittest.TestCase):
    name = 'first-instinct-decision-supervision-v1-test0001'

    def run_watch(self, responses, fail_stops=0):
        clock = [0.]; calls = []; remaining = [fail_stops]
        def call(method, path):
            calls.append((method, path))
            if method == 'POST':
                if remaining[0]: remaining[0] -= 1; raise OSError('temporary')
                return {}
            value = responses.pop(0) if len(responses)>1 else responses[0]
            if isinstance(value, Exception): raise value
            return value
        result = watch(self.name, 30, call=call, now=lambda: clock[0], sleep=lambda s: clock.__setitem__(0,clock[0]+s))
        return result, calls, clock[0]

    def test_no_rental_expires_without_mutations(self):
        result,calls,_ = self.run_watch([[dict(id='old',name='other',desiredStatus='EXITED')]])
        self.assertEqual(result['status'],'expired_without_rental')
        self.assertTrue(all(method=='GET' for method,_ in calls))

    def test_only_exact_new_rental_stopped(self):
        result,calls,elapsed = self.run_watch([[dict(id='new123',name=self.name,desiredStatus='RUNNING'),
            dict(id='old123',name='first-instinct-paired-capacity-v1',desiredStatus='EXITED')]])
        self.assertEqual(result,dict(status='stop_requested',pod_id='new123'))
        self.assertEqual([p for m,p in calls if m=='POST'],['/pods/new123/stop'])
        self.assertEqual(elapsed,30)

    def test_outage_after_identity_is_known_still_stops(self):
        result,calls,_ = self.run_watch([[dict(id='new123',name=self.name,desiredStatus='RUNNING')],OSError('temporary')],2)
        self.assertEqual(result['status'],'stop_requested')
        self.assertEqual(sum(m=='POST' for m,_ in calls),3)

    def test_removed_rental_is_not_resumed(self):
        result,calls,_ = self.run_watch([[dict(id='new123',name=self.name,desiredStatus='RUNNING')],[]])
        self.assertEqual(result['status'],'already_absent')
        self.assertTrue(all(m=='GET' for m,_ in calls))

    def test_invalid_scope_or_duration_rejected(self):
        for name,seconds in [('first-instinct-paired-capacity-v1',30),(self.name,1201),(self.name,0)]:
            with self.assertRaises(ValueError): validate(name,seconds)


if __name__ == '__main__': unittest.main()
