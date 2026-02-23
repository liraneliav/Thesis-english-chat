import numpy as np
from transformers import pipeline
from typing import Sequence, Tuple

_TOXICITY_PIPE = pipeline(
    task="text-classification",
    model="textdetox/glot500-toxicity-classifier",
    return_all_scores=True,
    device=-1,
    truncation=True
)

_LABEL_MAP = {
    "LABEL_0": "non-toxic",
    "LABEL_1": "toxic"
}

def measuring_toxicity(message: str) -> dict[str, float]:
    """
    Run the glot500 toxicity classifier and return a dict like:
    {"non-toxic": 0.83, "toxic": 0.17}
    """
    outputs = _TOXICITY_PIPE(str(message))

    records = outputs[0] if isinstance(outputs, list) and len(outputs) and isinstance(outputs[0], list) else outputs

    results: dict[str, float] = {}
    for rec in records:
        # normalize label
        lbl = rec["label"]
        if isinstance(lbl, np.ndarray):
            lbl = lbl.item()
        lbl = _LABEL_MAP.get(lbl, str(lbl))

        # normalize score
        sc = rec["score"]
        if isinstance(sc, np.ndarray):
            sc = sc.item()
        sc = float(sc)

        results[lbl] = sc

    results.setdefault("non-toxic", 0.0)
    results.setdefault("toxic", 0.0)
    return results
