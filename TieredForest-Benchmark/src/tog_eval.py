from typing import Any, Dict, List, Optional


def clean_results(text: str) -> str:
    """
    Extract content between the first '{' and '}'.
    Returns 'NULL' when the pair is not found.
    """
    if "{" not in text or "}" not in text:
        return "NULL"
    start = text.find("{") + 1
    end = text.find("}")
    if end <= start:
        return "NULL"
    return text[start:end]


def normalize_prediction(prediction: Any) -> str:
    """
    Keep behavior close to ToG eval:
    - If prediction contains "{...}", evaluate on inner content.
    - Otherwise evaluate on raw text.
    """
    text = "" if prediction is None else str(prediction).strip()
    extracted = clean_results(text)
    return text if extracted == "NULL" else extracted


def exact_match(prediction: str, answers: List[str]) -> bool:
    """
    ToG-compatible EM check:
    - lowercase
    - remove spaces
    - allow exact / substring containment both ways
    """
    pred = prediction.strip().replace(" ", "").lower()
    if not pred:
        return False

    for answer in answers:
        ans = str(answer).strip().replace(" ", "").lower()
        if not ans:
            continue
        if pred == ans or pred in ans or ans in pred:
            return True
    return False


def evaluate_prediction(prediction: Any, answers: List[str]) -> bool:
    normalized = normalize_prediction(prediction)
    return exact_match(normalized, answers)


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a batch of records.

    Required record keys:
    - prediction
    - answers (list[str])
    """
    right = 0
    error = 0

    for record in records:
        prediction = record.get("prediction", "")
        answers = record.get("answers", [])
        if evaluate_prediction(prediction, answers):
            right += 1
        else:
            error += 1

    total = len(records)
    exact_match_score = float(right / total) if total else 0.0
    return {
        "Exact Match": exact_match_score,
        "Right Samples": right,
        "Error Samples": error,
        "Total Samples": total,
    }


def build_eval_record(
    sample_id: str,
    dataset: str,
    question: str,
    prediction: str,
    answers: List[str],
    split: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "dataset": dataset,
        "split": split,
        "question": question,
        "prediction": prediction,
        "answers": answers,
    }
