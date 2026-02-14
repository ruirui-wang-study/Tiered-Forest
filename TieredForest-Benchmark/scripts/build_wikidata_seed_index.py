#!/usr/bin/env python3
"""
Build a targeted Wikidata index for ToG/Tiered-Forest from WebQSP seed entities.

This script avoids full-dump preprocessing by querying Wikidata SPARQL directly and
generating the same index artifacts expected by ToG/Wikidata/simple_wikidata_db server:
  - labels/labels_0.jsonl
  - plabels/plabels_0.jsonl
  - indices/relation_entities_chunk_1.pickle
  - indices/tail_entities_chunk_1.pickle
  - indices/tail_values_chunk_1.pickle
  - indices/external_ids_chunk_1.pickle
  - indices/mid_to_qid_chunk_1.pickle
"""

import argparse
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)
ENTITY_URI_PREFIX = "http://www.wikidata.org/entity/"

# Relation set aligned with current Tier-1 rules.
PID_TO_LABEL = {
    "P57": "director",
    "P161": "cast member",
    "P175": "performer",
    "P136": "genre",
    "P50": "author",
    "P58": "screenwriter",
    "P577": "publication date",
    "P364": "original language of film or TV show",
}

# PIDs where reverse lookup helps Tier-1 "in" queries the most.
INVERSE_PRIORITY_PIDS = {"P57", "P161", "P175", "P50", "P58"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted Wikidata index from WebQSP seeds")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=os.path.join("ToG", "data", "WebQSP.json"),
        help="Path to WebQSP.json containing qid_topic_entity",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Question split to use for seed collection",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of samples to use for seed extraction (0 means all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("ToG", "Wikidata", "local_service_data"),
        help="Output local_service_data directory",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://query.wikidata.org/sparql",
        help="Wikidata SPARQL endpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=60,
        help="QID batch size for SPARQL VALUES queries",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=45,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max_rows_per_query",
        type=int,
        default=12000,
        help="SPARQL LIMIT per query to avoid huge payloads",
    )
    return parser.parse_args()


def iter_chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def split_matches(sample_id: str, split: str) -> bool:
    if split == "all":
        return True
    sid = sample_id.lower()
    if split == "test":
        return "webqtest" in sid or "test" in sid
    if split == "train":
        return "webqtrn" in sid or "train" in sid or "trn" in sid
    return False


def load_seed_qids(dataset_file: str, split: str, limit: int) -> List[str]:
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seeds: List[str] = []
    seen: Set[str] = set()
    kept = 0
    for sample in data:
        sample_id = str(sample.get("QuestionId", ""))
        if not split_matches(sample_id, split):
            continue

        qid_map = sample.get("qid_topic_entity", {})
        if isinstance(qid_map, dict):
            for qid in qid_map.keys():
                q = str(qid).strip().upper()
                if QID_RE.fullmatch(q) and q not in seen:
                    seen.add(q)
                    seeds.append(q)

        kept += 1
        if limit > 0 and kept >= limit:
            break

    return seeds


def to_wd_values(qids: Sequence[str]) -> str:
    return " ".join(f"wd:{q}" for q in qids)


def extract_qid(uri_or_qid: str) -> Optional[str]:
    text = str(uri_or_qid).strip()
    if text.startswith(ENTITY_URI_PREFIX):
        text = text[len(ENTITY_URI_PREFIX) :]
    text = text.upper()
    if QID_RE.fullmatch(text):
        return text
    return None


def run_sparql(
    session: requests.Session,
    endpoint: str,
    query: str,
    timeout_s: int,
    retries: int = 3,
    sleep_s: float = 1.2,
) -> List[Dict[str, Dict[str, str]]]:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "TieredForest-Benchmark/targeted-index-builder",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                obj = resp.json()
                return obj.get("results", {}).get("bindings", [])
            # Retry on throttling/server errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(sleep_s * attempt)
                continue
            return []
        except Exception:
            time.sleep(sleep_s * attempt)
    return []


def ensure_dirs(base_dir: str) -> None:
    Path(base_dir, "labels").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "plabels").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "indices").mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    seeds = load_seed_qids(args.dataset_file, args.split, args.limit)
    if not seeds:
        raise RuntimeError("No seed QIDs found from dataset.")

    print(f"Seed QIDs: {len(seeds)}")
    ensure_dirs(args.output_dir)

    # Core index structures
    relation_entities: DefaultDict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: {"head": [], "tail": []}
    )
    tail_entities: DefaultDict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: {"head": [], "tail": []}
    )
    tail_values: DefaultDict[str, List[str]] = defaultdict(list)
    external_ids: DefaultDict[str, List[str]] = defaultdict(list)
    mid_to_qid: DefaultDict[str, List[str]] = defaultdict(list)

    qid_to_label: Dict[str, str] = {}
    relation_seen: Set[Tuple[str, str, str]] = set()  # (qid, side, pid)
    entity_link_seen: Set[Tuple[str, str, str, str]] = set()  # (key, side, qid, label)
    value_seen: Set[Tuple[str, str]] = set()  # (key, value)

    session = requests.Session()

    # 1) Fetch labels for seed entities
    for chunk in iter_chunks(seeds, args.batch_size):
        query = f"""
SELECT ?q ?qLabel WHERE {{
  VALUES ?q {{ {to_wd_values(chunk)} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
        rows = run_sparql(session, args.endpoint, query, args.timeout_s)
        for row in rows:
            q = extract_qid(row.get("q", {}).get("value", ""))
            label = row.get("qLabel", {}).get("value", "")
            if q and label and q not in qid_to_label:
                qid_to_label[q] = label

    # 2) Fetch entity links and scalar values for target relations
    pids = list(PID_TO_LABEL.keys())
    for pid in pids:
        print(f"Fetching PID {pid} ...")
        for chunk in iter_chunks(seeds, args.batch_size):
            values = to_wd_values(chunk)

            # Outgoing entity-valued triples
            query_out_entities = f"""
SELECT ?h ?hLabel ?t ?tLabel WHERE {{
  VALUES ?h {{ {values} }}
  ?h wdt:{pid} ?t .
  FILTER(STRSTARTS(STR(?t), "{ENTITY_URI_PREFIX}"))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {args.max_rows_per_query}
"""
            rows = run_sparql(session, args.endpoint, query_out_entities, args.timeout_s)
            for row in rows:
                h = extract_qid(row.get("h", {}).get("value", ""))
                t = extract_qid(row.get("t", {}).get("value", ""))
                h_label = row.get("hLabel", {}).get("value", "")
                t_label = row.get("tLabel", {}).get("value", "")
                if not h or not t:
                    continue
                if h_label and h not in qid_to_label:
                    qid_to_label[h] = h_label
                if t_label and t not in qid_to_label:
                    qid_to_label[t] = t_label

                rel_key_head = (h, "head", pid)
                if rel_key_head not in relation_seen:
                    relation_entities[h]["head"].append({"pid": pid, "label": PID_TO_LABEL[pid]})
                    relation_seen.add(rel_key_head)
                rel_key_tail = (t, "tail", pid)
                if rel_key_tail not in relation_seen:
                    relation_entities[t]["tail"].append({"pid": pid, "label": PID_TO_LABEL[pid]})
                    relation_seen.add(rel_key_tail)

                key_h = f"{h}@{pid}"
                item_t = {"qid": t, "label": qid_to_label.get(t, t)}
                link_key_h = (key_h, "tail", item_t["qid"], item_t["label"])
                if link_key_h not in entity_link_seen:
                    tail_entities[key_h]["tail"].append(item_t)
                    entity_link_seen.add(link_key_h)

                key_t = f"{t}@{pid}"
                item_h = {"qid": h, "label": qid_to_label.get(h, h)}
                link_key_t = (key_t, "head", item_h["qid"], item_h["label"])
                if link_key_t not in entity_link_seen:
                    tail_entities[key_t]["head"].append(item_h)
                    entity_link_seen.add(link_key_t)

            # Outgoing scalar-valued triples (e.g., P577)
            query_out_values = f"""
SELECT ?h ?hLabel ?v WHERE {{
  VALUES ?h {{ {values} }}
  ?h wdt:{pid} ?v .
  FILTER(!STRSTARTS(STR(?v), "{ENTITY_URI_PREFIX}"))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {args.max_rows_per_query}
"""
            rows = run_sparql(session, args.endpoint, query_out_values, args.timeout_s)
            for row in rows:
                h = extract_qid(row.get("h", {}).get("value", ""))
                if not h:
                    continue
                h_label = row.get("hLabel", {}).get("value", "")
                if h_label and h not in qid_to_label:
                    qid_to_label[h] = h_label

                rel_key_head = (h, "head", pid)
                if rel_key_head not in relation_seen:
                    relation_entities[h]["head"].append({"pid": pid, "label": PID_TO_LABEL[pid]})
                    relation_seen.add(rel_key_head)

                v = row.get("v", {}).get("value", "")
                if v:
                    key_h = f"{h}@{pid}"
                    pair = (key_h, v)
                    if pair not in value_seen:
                        tail_values[key_h].append(v)
                        value_seen.add(pair)

            # Inverse links for relation-centric question patterns.
            if pid in INVERSE_PRIORITY_PIDS:
                query_in_entities = f"""
SELECT ?h ?hLabel ?t ?tLabel WHERE {{
  VALUES ?t {{ {values} }}
  ?h wdt:{pid} ?t .
  FILTER(STRSTARTS(STR(?h), "{ENTITY_URI_PREFIX}"))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {args.max_rows_per_query}
"""
                rows = run_sparql(session, args.endpoint, query_in_entities, args.timeout_s)
                for row in rows:
                    h = extract_qid(row.get("h", {}).get("value", ""))
                    t = extract_qid(row.get("t", {}).get("value", ""))
                    h_label = row.get("hLabel", {}).get("value", "")
                    t_label = row.get("tLabel", {}).get("value", "")
                    if not h or not t:
                        continue
                    if h_label and h not in qid_to_label:
                        qid_to_label[h] = h_label
                    if t_label and t not in qid_to_label:
                        qid_to_label[t] = t_label

                    rel_key_head = (h, "head", pid)
                    if rel_key_head not in relation_seen:
                        relation_entities[h]["head"].append({"pid": pid, "label": PID_TO_LABEL[pid]})
                        relation_seen.add(rel_key_head)
                    rel_key_tail = (t, "tail", pid)
                    if rel_key_tail not in relation_seen:
                        relation_entities[t]["tail"].append({"pid": pid, "label": PID_TO_LABEL[pid]})
                        relation_seen.add(rel_key_tail)

                    key_h = f"{h}@{pid}"
                    item_t = {"qid": t, "label": qid_to_label.get(t, t)}
                    link_key_h = (key_h, "tail", item_t["qid"], item_t["label"])
                    if link_key_h not in entity_link_seen:
                        tail_entities[key_h]["tail"].append(item_t)
                        entity_link_seen.add(link_key_h)

                    key_t = f"{t}@{pid}"
                    item_h = {"qid": h, "label": qid_to_label.get(h, h)}
                    link_key_t = (key_t, "head", item_h["qid"], item_h["label"])
                    if link_key_t not in entity_link_seen:
                        tail_entities[key_t]["head"].append(item_h)
                        entity_link_seen.add(link_key_t)

    # Include seed labels where missing (fallback to QID text)
    for q in seeds:
        qid_to_label.setdefault(q, q)

    labels_rows = [{"qid": qid, "label": label} for qid, label in sorted(qid_to_label.items())]
    plabel_rows = [{"pid": pid, "label": label} for pid, label in PID_TO_LABEL.items()]

    labels_path = os.path.join(args.output_dir, "labels", "labels_0.jsonl")
    plabels_path = os.path.join(args.output_dir, "plabels", "plabels_0.jsonl")
    write_jsonl(labels_path, labels_rows)
    write_jsonl(plabels_path, plabel_rows)

    idx_dir = os.path.join(args.output_dir, "indices")
    with open(os.path.join(idx_dir, "relation_entities_chunk_1.pickle"), "wb") as f:
        pickle.dump(dict(relation_entities), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idx_dir, "tail_entities_chunk_1.pickle"), "wb") as f:
        pickle.dump(dict(tail_entities), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idx_dir, "tail_values_chunk_1.pickle"), "wb") as f:
        pickle.dump(dict(tail_values), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idx_dir, "external_ids_chunk_1.pickle"), "wb") as f:
        pickle.dump(dict(external_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idx_dir, "mid_to_qid_chunk_1.pickle"), "wb") as f:
        pickle.dump(dict(mid_to_qid), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    print(f"labels: {len(labels_rows)}")
    print(f"relation_entities keys: {len(relation_entities)}")
    print(f"tail_entities keys: {len(tail_entities)}")
    print(f"tail_values keys: {len(tail_values)}")
    print(f"written to: {args.output_dir}")


if __name__ == "__main__":
    main()
