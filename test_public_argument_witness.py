import copy
import unittest

from tool_lab.public_argument_witness import discover,verify,date_witness


class PublicWitnessTests(unittest.TestCase):
    def test_currency_span_future_source_and_wrong_value(self):
        history=[dict(api='show_note',response={'content':'Alice => $1,020.50; Bob => $19'})]
        witness=discover('',history,None,'amount',19)
        self.assertTrue(verify(witness,'',history,None,'amount',19))
        for change in ({'source_call':1},{'span':[0,3]},{'decimal':'20'}):
            with self.assertRaises(ValueError):verify(dict(witness,**change),'',history,None,'amount',19)
        with self.assertRaises(ValueError):verify(witness,'',history,None,'amount',20)
        self.assertIsNone(discover('',history,None,'amount',1))
        self.assertIsNone(discover('',history,None,'amount',True))
        self.assertTrue(verify(discover('',history,None,'amount',1020.5),'',history,None,'amount',1020.5))

    def test_secret_and_partial_number_do_not_supply_evidence(self):
        for event in (dict(api='login',response={'note':'$19'}),
                      dict(api='show_note',response={'password':'$19'}),
                      dict(api='show_note',response={'content':'$192 $19.999'})):
            self.assertIsNone(discover('',[event],None,'amount',19))

    def test_calendar_month_year_leap_and_inclusive_window(self):
        cases=[('Monday, January 01, 2024','yesterday','2023-12-31'),
               ('Friday, March 01, 2024','yesterday','2024-02-29'),
               ('Wednesday, March 01, 2023','yesterday','2023-02-28'),
               ('Friday, June 02, 2023','last 10 days (including today)','2023-05-24')]
        for current,goal,value in cases:
            clock={'date':current};witness=date_witness(goal,clock,value)
            self.assertTrue(verify(witness,goal,[],clock,'min_created_at',value))
            with self.assertRaises(ValueError):verify(dict(witness,day_offset=-100),goal,[],clock,'min_created_at',value)
        self.assertIsNone(date_witness('recently',{'date':cases[0][0]},'2023-12-31'))
        self.assertIsNone(date_witness('yesterday',None,'2023-12-31'))
        with self.assertRaises(ValueError):date_witness('yesterday',{'date':'Tuesday, January 01, 2024'},'2023-12-31')


if __name__=='__main__':unittest.main()
