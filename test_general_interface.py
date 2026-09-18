import copy
import unittest

from general_lab.interface import requests, answer
from scale_lab.common import messages


class FakePredictor:
    def validate(self, item):
        return [1]

    def predict(self, item):
        options = item["options"]
        return {"probabilities": {o["id"]: 1 / len(options) for o in options},
                "choice": options[0]["id"], "milliseconds": 0}


class InterfaceTests(unittest.TestCase):
    def test_questions_are_independent_and_ids_not_prompt_features(self):
        payload = {"state": {"ready": True}, "questions": {"a": {"type": "binary", "instructions": "Is ready true?"}}}
        before = messages(requests(payload)[0][2])
        changed = copy.deepcopy(payload)
        changed["questions"]["b"] = {"type": "choice", "instructions": "Which value?", "criteria": {"x": "True", "y": "False"}}
        self.assertEqual(before, messages(requests(changed)[0][2]))
        changed["questions"]["secret_name"] = changed["questions"].pop("a")
        self.assertEqual(before, messages(requests(changed)[1][2]))

    def test_dynamic_options_and_ordered_expectation(self):
        payload = {"state": "Five open issues.", "questions": {
            "level": {"type": "score", "instructions": "Use the supplied issue count thresholds.", "criteria": ["Zero", "One to five", "More than five"]},
            "check": {"type": "binary", "instructions": "Are there five issues?"},
            "route": {"type": "choice", "instructions": "Where?", "criteria": {"new_label_1": "North", "new_label_2": "South"}}}}
        values = answer(payload, FakePredictor())["answers"]
        self.assertEqual(values["level"]["score"], 1.)
        self.assertEqual(values["check"]["probability_yes"], .5)
        self.assertEqual(set(values["route"]["probabilities"]), {"new_label_1", "new_label_2"})

    def test_rejects_incomplete_question(self):
        with self.assertRaises(ValueError):
            requests({"state": "x", "questions": {"a": {"type": "choice", "instructions": "Which?", "criteria": {"one": "Only one"}}}})


if __name__ == "__main__":
    unittest.main()
