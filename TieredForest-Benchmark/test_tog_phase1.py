#!/usr/bin/env python3
"""
Phase-1 smoke checks for ToG data loader and ToG-style EM evaluator.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader_tog import ToGDatasetLoader
from src.tog_eval import evaluate_prediction


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tog_data_dir = os.path.join(base_dir, "..", "ToG", "data")
    tog_data_dir = os.path.abspath(tog_data_dir)

    loader = ToGDatasetLoader(tog_data_dir)

    webqsp = loader.load("webqsp", split="test", limit=5)
    cwq_test = loader.load("cwq", split="test", limit=5)

    assert len(webqsp) == 5, "webqsp test split should be loadable"
    assert len(cwq_test) == 5, "cwq test split should be loadable"
    assert webqsp[0]["question"], "normalized question should not be empty"
    assert isinstance(webqsp[0]["answers"], list), "answers should be list[str]"

    assert evaluate_prediction("{Baruch Spinoza}", ["Giambattista Vico", "Baruch Spinoza"])
    assert not evaluate_prediction("Unknown", ["Baruch Spinoza"])

    print("Phase-1 checks passed.")


if __name__ == "__main__":
    main()
