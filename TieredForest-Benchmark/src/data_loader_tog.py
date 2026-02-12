import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


DATASET_FILES = {
    "cwq": "cwq.json",
    "webqsp": "WebQSP.json",
    "grailqa": "graliqa.json",
    "simpleqa": "SimpleQA.json",
    "qald": "qald_10-en.json",
    "webquestions": "WebQuestions.json",
    "trex": "T-REX.json",
    "zeroshotre": "Zero_Shot_RE.json",
    "creak": "creak.json",
}


QUESTION_FIELDS = {
    "cwq": "question",
    "webqsp": "RawQuestion",
    "grailqa": "question",
    "simpleqa": "question",
    "qald": "question",
    "webquestions": "question",
    "trex": "input",
    "zeroshotre": "input",
    "creak": "sentence",
}


class ToGDatasetLoader:
    """
    Load ToG datasets from ToG/data/* and normalize each sample into a common schema.

    Normalized sample fields:
    - id: stable string id
    - dataset: dataset name
    - split: train/dev/test
    - question: input question string
    - answers: list[str]
    - topic_entity: dict[mid, name]
    - qid_topic_entity: dict[qid, name]
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(
        self,
        dataset: str,
        split: str = "test",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        dataset_key = dataset.lower()
        if dataset_key not in DATASET_FILES:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. Choose from: {sorted(DATASET_FILES.keys())}"
            )

        path = os.path.join(self.data_dir, DATASET_FILES[dataset_key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        normalized = [
            self._normalize_sample(dataset_key, sample, idx)
            for idx, sample in enumerate(raw_samples)
        ]

        normalized = self._filter_split(normalized, split.lower())

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(normalized)

        if limit is not None:
            normalized = normalized[:limit]

        return normalized

    def _normalize_sample(
        self, dataset: str, sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        sample_id = self._extract_id(dataset, sample, idx)
        inferred_split = self._infer_split(dataset, sample, sample_id)
        question = self._extract_question(dataset, sample)
        answers = self._extract_answers(dataset, sample)
        topic_entity, qid_topic_entity = self._extract_topic_entities(dataset, sample)

        return {
            "id": sample_id,
            "dataset": dataset,
            "split": inferred_split,
            "question": question,
            "answers": answers,
            "topic_entity": topic_entity,
            "qid_topic_entity": qid_topic_entity,
        }

    def _extract_id(self, dataset: str, sample: Dict[str, Any], idx: int) -> str:
        candidate_keys = [
            "ID",
            "QuestionId",
            "qid",
            "ex_id",
            "url",
        ]
        for key in candidate_keys:
            value = sample.get(key)
            if value is not None:
                return str(value)
        return f"{dataset}-{idx}"

    def _infer_split(self, dataset: str, sample: Dict[str, Any], sample_id: str) -> str:
        split_field = sample.get("split")
        if split_field:
            split = str(split_field).strip().lower()
            if split.startswith("train"):
                return "train"
            if split.startswith("dev") or split.startswith("val"):
                return "dev"
            if split.startswith("test"):
                return "test"

        sid = sample_id.lower()
        if dataset in {"cwq", "webqsp"}:
            if "webqtest" in sid or "test" in sid:
                return "test"
            if "webqtrn" in sid or "train" in sid or "trn" in sid:
                return "train"
            if "dev" in sid or "valid" in sid:
                return "dev"

        # Most ToG distributed files are test-oriented in this repo.
        return "test"

    def _extract_question(self, dataset: str, sample: Dict[str, Any]) -> str:
        field = QUESTION_FIELDS[dataset]
        value = sample.get(field)
        if value is None:
            return ""
        return str(value).strip()

    def _extract_answers(self, dataset: str, sample: Dict[str, Any]) -> List[str]:
        answers: List[str] = []

        if dataset == "cwq":
            answers_field = sample.get("answers")
            if isinstance(answers_field, list):
                for answer in answers_field:
                    if isinstance(answer, dict):
                        aliases = answer.get("aliases", [])
                        if isinstance(aliases, list):
                            answers.extend(str(a) for a in aliases)
                        if answer.get("answer") is not None:
                            answers.append(str(answer["answer"]))
                    elif answer is not None:
                        answers.append(str(answer))
            else:
                answer_field = sample.get("answer")
                if isinstance(answer_field, list):
                    for answer in answer_field:
                        if isinstance(answer, dict):
                            if answer.get("entity_name") is not None:
                                answers.append(str(answer["entity_name"]))
                            elif answer.get("answer_argument") is not None:
                                answers.append(str(answer["answer_argument"]))
                        elif answer is not None:
                            answers.append(str(answer))
                elif answer_field is not None:
                    answers.append(str(answer_field))

        elif dataset == "webqsp":
            parses = sample.get("Parses", [])
            for parse in parses:
                for answer in parse.get("Answers", []):
                    entity_name = answer.get("EntityName")
                    answer_arg = answer.get("AnswerArgument")
                    if entity_name:
                        answers.append(str(entity_name))
                    elif answer_arg:
                        answers.append(str(answer_arg))

        elif dataset == "grailqa":
            for answer in sample.get("answer", []):
                if isinstance(answer, dict):
                    if answer.get("entity_name") is not None:
                        answers.append(str(answer["entity_name"]))
                    elif answer.get("answer_argument") is not None:
                        answers.append(str(answer["answer_argument"]))
                elif answer is not None:
                    answers.append(str(answer))

        elif dataset == "simpleqa":
            if sample.get("answer") is not None:
                answers.append(str(sample["answer"]))

        elif dataset == "qald":
            answer_field = sample.get("answer", {})
            if isinstance(answer_field, dict):
                answers.extend(str(v) for v in answer_field.values())
            elif answer_field is not None:
                answers.append(str(answer_field))

        elif dataset == "webquestions":
            raw_answers = sample.get("answers", [])
            if isinstance(raw_answers, list):
                answers.extend(str(a) for a in raw_answers if a is not None)
            elif raw_answers is not None:
                answers.append(str(raw_answers))

        elif dataset in {"trex", "zeroshotre"}:
            if sample.get("answer") is not None:
                answers.append(str(sample["answer"]))
            aliases = sample.get("alias", [])
            if isinstance(aliases, list):
                answers.extend(str(a) for a in aliases if a is not None)

        elif dataset == "creak":
            label = sample.get("label")
            if label is not None:
                answers.append(str(label))

        return self._unique_non_empty(answers)

    def _extract_topic_entities(
        self, dataset: str, sample: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        topic_entity = self._coerce_entity_map(sample.get("topic_entity", {}))
        qid_topic_entity = self._coerce_entity_map(sample.get("qid_topic_entity", {}))

        if dataset == "webqsp" and not topic_entity:
            parses = sample.get("Parses", [])
            for parse in parses:
                mid = parse.get("TopicEntityMid")
                name = parse.get("TopicEntityName")
                if mid and name:
                    topic_entity[str(mid)] = str(name)

        if dataset in {"trex", "zeroshotre"} and not topic_entity:
            topic_entity = self._coerce_entity_map(sample.get("topic_entity_ids", {}))

        return topic_entity, qid_topic_entity

    def _coerce_entity_map(self, value: Any) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, str] = {}
        for k, v in value.items():
            if k is None or v is None:
                continue
            out[str(k)] = str(v)
        return out

    def _filter_split(
        self, samples: List[Dict[str, Any]], split: str
    ) -> List[Dict[str, Any]]:
        if split == "all":
            return samples
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of {'train', 'dev', 'test', 'all'}")
        return [sample for sample in samples if sample["split"] == split]

    def _unique_non_empty(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            norm = text.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out.append(text)
        return out
