import re
from typing import List, Optional, Set, Tuple

from .base_backend import BaseKGBackend
from .. import config


class FreebaseBackend(BaseKGBackend):
    """
    Freebase backend via SPARQL endpoint.
    """

    name = "freebase"

    def __init__(self, sparql_endpoint: Optional[str] = None, max_relation_scan: int = 8):
        self.sparql_endpoint = sparql_endpoint or config.FREEBASE_SPARQL_ENDPOINT
        self.max_relation_scan = max_relation_scan
        self._sparql = self._build_client()

    def _build_client(self):
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "SPARQLWrapper is required for Freebase backend. "
                "Install it or use wikidata/metaqa backend."
            ) from exc

        client = SPARQLWrapper(self.sparql_endpoint)
        client.setReturnFormat(JSON)
        return client

    def _run(self, query: str):
        try:
            self._sparql.setQuery(query)
            return self._sparql.query().convert().get("results", {}).get("bindings", [])
        except Exception:
            return []

    def _entity_uri_to_mid(self, uri: str) -> str:
        prefix = "http://rdf.freebase.com/ns/"
        return uri.replace(prefix, "")

    def _is_mid(self, value: str) -> bool:
        return bool(re.fullmatch(r"(m|g)\.[A-Za-z0-9_]+", value.strip()))

    def _escape_literal(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def find_entity(self, entity_name: str) -> Optional[str]:
        raw = entity_name.strip()
        if not raw:
            return None
        if self._is_mid(raw):
            return raw

        escaped = self._escape_literal(raw)
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
WHERE {{
  {{
    ?entity ns:type.object.name "{escaped}"@en .
  }}
  UNION
  {{
    ?entity ns:type.object.name "{escaped}" .
  }}
}}
LIMIT 1
"""
        rows = self._run(query)
        if not rows:
            return None
        uri = rows[0].get("entity", {}).get("value")
        if not uri:
            return None
        return self._entity_uri_to_mid(uri)

    def get_entity_relations(self, entity: str) -> List[str]:
        mid = self.find_entity(entity)
        if not mid:
            return []

        out_query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
  ns:{mid} ?relation ?x .
}}
"""
        in_query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
  ?x ?relation ns:{mid} .
}}
"""

        relations: List[str] = []
        seen: Set[str] = set()

        for row in self._run(out_query):
            uri = row.get("relation", {}).get("value", "")
            rel = self._entity_uri_to_mid(uri)
            if rel and rel not in seen:
                seen.add(rel)
                relations.append(rel)

        for row in self._run(in_query):
            uri = row.get("relation", {}).get("value", "")
            rel = self._entity_uri_to_mid(uri)
            inv = f"inverse_{rel}" if rel else ""
            if inv and inv not in seen:
                seen.add(inv)
                relations.append(inv)

        return relations

    def query_relation(self, entity: str, relation: str) -> List[str]:
        mid = self.find_entity(entity)
        if not mid:
            return []

        is_inverse = relation.startswith("inverse_")
        rel = relation[len("inverse_") :] if is_inverse else relation

        if is_inverse:
            query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?neighbor
WHERE {{
  ?neighbor ns:{rel} ns:{mid} .
}}
LIMIT 100
"""
        else:
            query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?neighbor
WHERE {{
  ns:{mid} ns:{rel} ?neighbor .
}}
LIMIT 100
"""

        rows = self._run(query)
        out: List[str] = []
        for row in rows:
            val = row.get("neighbor", {}).get("value")
            if not val:
                continue
            out.append(self._entity_uri_to_mid(val))
        return out

    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str]]:
        direction = direction.lower()
        if direction not in {"out", "in", "both"}:
            direction = "out"

        relations = [relation] if relation else self.get_entity_relations(entity)
        if not relation:
            if direction == "out":
                relations = [r for r in relations if not r.startswith("inverse_")]
            elif direction == "in":
                relations = [r for r in relations if r.startswith("inverse_")]
            relations = relations[: self.max_relation_scan]

        neighbors: List[Tuple[str, str]] = []
        for rel in relations:
            if direction == "out" and rel.startswith("inverse_"):
                continue
            if direction == "in" and not rel.startswith("inverse_"):
                continue
            for candidate in self.query_relation(entity, rel):
                neighbors.append((candidate, rel))
        return neighbors
