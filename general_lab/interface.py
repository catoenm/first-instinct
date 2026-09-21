"""User-defined choice, binary and ordered questions over a shared supplied state.

Each question is encoded independently. Adding a question cannot alter another
question's prompt. This implementation repeats state processing; it does not
claim Jev's shared-state latency or its calibration guarantees.
"""

import argparse
import json
from pathlib import Path

from scale_lab.common import validate_input


def requests(payload):
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("Provide state and questions")
    state = payload["state"]
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False, allow_nan=False, sort_keys=True)
    questions = payload.get("questions")
    if not isinstance(state, str) or not state.strip() or not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise ValueError("Provide nonempty state and 1–32 named questions")
    result = []
    for identity, question in questions.items():
        if not isinstance(identity, str) or not identity or not isinstance(question, dict):
            raise ValueError("Every question needs a nonempty identifier and definition")
        kind = question.get("type", "choice")
        instructions = question.get("instructions")
        if kind == "binary":
            options = [{"id": "yes", "description": "Yes, the proposition is true"},
                       {"id": "no", "description": "No, the proposition is false"}]
        elif kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict):
                raise ValueError("Choice criteria must map identifiers to descriptions")
            options = [{"id": key, "description": value} for key, value in criteria.items()]
        elif kind == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list):
                raise ValueError("Score criteria must be an ordered list of level descriptions")
            options = [{"id": str(i), "description": value} for i, value in enumerate(criteria)]
        else:
            raise ValueError("Question type must be choice, binary, or score")
        item = {"state": state, "question": instructions, "options": options}
        validate_input(item)
        result.append((identity, kind, item))
    return result


def answer(payload, predictor):
    prepared = requests(payload)
    # Validate all lengths before starting model work, so a malformed later
    # question cannot create a misleading partial successful response.
    for _, _, item in prepared:
        predictor.validate(item)
    answers = {}
    for identity, kind, item in prepared:
        prediction = predictor.predict(item)
        probabilities = prediction["probabilities"]
        value = {"type": kind, "probabilities": probabilities, "milliseconds": prediction["milliseconds"]}
        if kind == "binary":
            value["probability_yes"] = probabilities["yes"]
        elif kind == "score":
            value["score"] = sum(int(key) * probability for key, probability in probabilities.items())
            value["legend"] = {o["id"]: o["description"] for o in item["options"]}
            value["selected_level"] = int(prediction["choice"])
        else:
            value["choice"] = prediction["choice"]
        answers[identity] = value
    return {"answers": answers,
            "probability_note": "Model probabilities conditional on the supplied state and choices; calibration on arbitrary new questions is not established.",
            "execution": "Independent prompts; shared-state computation is not implemented."}


def main():
    # The shared typed interface also serves non-PyTorch backends.
    from scale_lab.infer import Predictor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=1536)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    requests(payload)
    predictor = Predictor(args.model, args.run, args.max_tokens, args.device)
    print(json.dumps(answer(payload, predictor), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
