import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .base_backend import BaseKGBackend
from .. import config


class WikidataBackend(BaseKGBackend):
    """
    Wikidata backend via ToG's distributed XML-RPC query client.

    This backend expects a running simple-wikidata-db server cluster.
    """

    name = "wikidata"

    def __init__(
        self,
        server_urls: Optional[Sequence[str]] = None,
        server_urls_file: Optional[str] = None,
        max_relation_scan: int = 8,
    ):
        self.max_relation_scan = max_relation_scan
        self._qid_cache: Dict[str, Optional[str]] = {}
        self._pid_cache: Dict[str, Optional[str]] = {}
        self._relation_cache: Dict[str, List[str]] = {}

        urls = self._resolve_urls(server_urls, server_urls_file)
        self.client = self._build_client(urls)

    def _resolve_urls(
        self,
        server_urls: Optional[Sequence[str]],
        server_urls_file: Optional[str],
    ) -> List[str]:
        if server_urls:
            return [u.strip() for u in server_urls if u and u.strip()]

        path = server_urls_file or config.WIKIDATA_SERVER_URLS_FILE
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            if urls:
                return urls

        if config.WIKIDATA_SERVER_URLS:
            return [u.strip() for u in config.WIKIDATA_SERVER_URLS if u.strip()]

        raise RuntimeError(
            "No Wikidata server URLs found. Configure [kg] wikidata_server_urls "
            "or wikidata_server_urls_file in config.ini."
        )

    def _build_client(self, urls: List[str]):
        tog_client_dir = os.path.join(config.ROOT_DIR, "ToG", "ToG")
        if tog_client_dir not in sys.path:
            sys.path.insert(0, tog_client_dir)

        try:
            from client import MultiServerWikidataQueryClient  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ToG Wikidata client from ToG/ToG/client.py"
            ) from exc

        return MultiServerWikidataQueryClient(urls)

    def _query_all(self, method: str, *args):
        try:
            result = self.client.query_all(method, *args)
            if result == "Not Found!":
                return None
            return result
        except Exception:
            return None

    def _is_qid(self, value: str) -> bool:
        return bool(re.fullmatch(r"Q\d+", value.strip(), re.IGNORECASE))

    def _is_pid(self, value: str) -> bool:
        return bool(re.fullmatch(r"P\d+", value.strip(), re.IGNORECASE))

    def _label_to_qid(self, label: str) -> Optional[str]:
        key = label.strip().lower()
        if key in self._qid_cache:
            return self._qid_cache[key]

        result = self._query_all("label2qid", label)
        qid = None
        if isinstance(result, set):
            qid = next(iter(result), None)
        elif isinstance(result, (list, tuple)):
            qid = result[0] if result else None
        elif isinstance(result, str):
            qid = result

        if qid:
            qid = str(qid).upper()
        self._qid_cache[key] = qid
        return qid

    def _label_to_pid(self, label_or_pid: str) -> Optional[str]:
        raw = label_or_pid.strip()
        if self._is_pid(raw):
            return raw.upper()

        key = raw.lower()
        if key in self._pid_cache:
            return self._pid_cache[key]

        result = self._query_all("label2pid", raw)
        pid = None
        if isinstance(result, set):
            pid = next(iter(result), None)
        elif isinstance(result, (list, tuple)):
            pid = result[0] if result else None
        elif isinstance(result, str):
            pid = result

        if pid:
            pid = str(pid).upper()
        self._pid_cache[key] = pid
        return pid

    def _relation_name(self, rel_obj: Dict[str, str]) -> Optional[str]:
        # ToG server may return {"pid": "P31", "label": "instance of"}.
        label = rel_obj.get("label")
        pid = rel_obj.get("pid")
        if label and label != "N/A":
            return label
        if pid:
            return pid
        return None

    def find_entity(self, entity_name: str) -> Optional[str]:
        raw = entity_name.strip()
        if not raw:
            return None

        if self._is_qid(raw):
            return raw.upper()

        # try exact, title-case and lower-case fallback
        for candidate in (raw, raw.title(), raw.lower()):
            qid = self._label_to_qid(candidate)
            if qid:
                return qid
        return None

    def get_entity_relations(self, entity: str) -> List[str]:
        qid = self.find_entity(entity)
        if not qid:
            return []

        if qid in self._relation_cache:
            return self._relation_cache[qid]

        rels = self._query_all("get_all_relations_of_an_entity", qid)
        if not isinstance(rels, dict):
            self._relation_cache[qid] = []
            return []

        names: List[str] = []
        seen: Set[str] = set()

        for rel_obj in rels.get("head", []):
            if isinstance(rel_obj, dict):
                name = self._relation_name(rel_obj)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)

        for rel_obj in rels.get("tail", []):
            if isinstance(rel_obj, dict):
                name = self._relation_name(rel_obj)
                if name:
                    inv = f"inverse_{name}"
                    if inv not in seen:
                        seen.add(inv)
                        names.append(inv)

        self._relation_cache[qid] = names
        return names

    def query_relation(self, entity: str, relation: str) -> List[str]:
        qid = self.find_entity(entity)
        if not qid:
            return []

        is_inverse = relation.startswith("inverse_")
        rel_name = relation[len("inverse_") :] if is_inverse else relation
        pid = self._label_to_pid(rel_name)
        if not pid:
            return []

        values = self._query_all("get_tail_entities_given_head_and_relation", qid, pid)
        if not isinstance(values, dict):
            if is_inverse:
                return []
            scalar_values = self._query_all("get_tail_values_given_head_and_relation", qid, pid)
            if isinstance(scalar_values, set):
                return sorted(str(v) for v in scalar_values if v is not None)
            if isinstance(scalar_values, (list, tuple)):
                return [str(v) for v in scalar_values if v is not None]
            return []

        key = "head" if is_inverse else "tail"
        items = values.get(key, [])
        out: List[str] = []
        for item in items:
            if isinstance(item, dict):
                label = item.get("label")
                qid_item = item.get("qid")
                out.append(label if (label and label != "N/A") else str(qid_item or ""))
            elif item is not None:
                out.append(str(item))

        return [v for v in out if v]

    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str]]:
        direction = direction.lower()
        if direction not in {"out", "in", "both"}:
            direction = "out"

        relations: List[str]
        explicit_relation = relation is not None and relation.strip() != ""
        if explicit_relation:
            base_rel = relation.strip()
            if direction == "in":
                relations = [base_rel if base_rel.startswith("inverse_") else f"inverse_{base_rel}"]
            elif direction == "both":
                core = base_rel[len("inverse_") :] if base_rel.startswith("inverse_") else base_rel
                relations = [core, f"inverse_{core}"]
            else:
                relations = [base_rel[len("inverse_") :] if base_rel.startswith("inverse_") else base_rel]
        else:
            relations = self.get_entity_relations(entity)
            if direction == "out":
                relations = [r for r in relations if not r.startswith("inverse_")]
            elif direction == "in":
                relations = [r for r in relations if r.startswith("inverse_")]
            relations = relations[: self.max_relation_scan]

        neighbors: List[Tuple[str, str]] = []
        for rel in relations:
            if not explicit_relation:
                if direction == "out" and rel.startswith("inverse_"):
                    continue
                if direction == "in" and not rel.startswith("inverse_"):
                    continue

            for candidate in self.query_relation(entity, rel):
                neighbors.append((candidate, rel))

        return neighbors

    def query_simple(self, question: str) -> Optional[str]:
        """
        Lightweight symbolic QA fallback for Tier-1 on Wikidata backend.
        """
        q = question.strip()
        if not q:
            return None

        templates = [
            (re.compile(r"^who directed (.+?)[\?]?$", re.IGNORECASE), ["director", "P57"], "out"),
            (
                re.compile(r"^who (?:is|was) the director of (.+?)[\?]?$", re.IGNORECASE),
                ["director", "P57"],
                "out",
            ),
            (
                re.compile(r"^what (?:films|movies) did (.+?) (?:act in|star in|appear in)[\?]?$", re.IGNORECASE),
                ["cast member", "performer", "P161", "P175"],
                "in",
            ),
            (
                re.compile(r"^who (?:acted in|starred in|was in) (.+?)[\?]?$", re.IGNORECASE),
                ["cast member", "performer", "P161", "P175"],
                "out",
            ),
            (re.compile(r"^what (?:is the )?genre (?:of |is )?(.+?)[\?]?$", re.IGNORECASE), ["genre", "P136"], "out"),
            (
                re.compile(r"^when was (.+?) released[\?]?$", re.IGNORECASE),
                ["publication date", "date of publication", "inception", "P577"],
                "out",
            ),
        ]

        for pattern, relations, direction in templates:
            match = pattern.match(q)
            if not match:
                continue

            entity_text = match.group(1).strip()
            entity = self.find_entity(entity_text)
            if not entity:
                continue

            for rel in relations:
                neighbors = self.get_neighbors(entity, relation=rel, direction=direction)
                if neighbors:
                    answers = [n[0] for n in neighbors[:3] if n and n[0]]
                    if answers:
                        return ", ".join(answers)

        return None
