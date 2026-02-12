from typing import List, Optional, Tuple

from .base_backend import BaseKGBackend
from ..graph_engine import MetaQAGraphEngine


class MetaQABackend(BaseKGBackend):
    """
    Adapter that exposes existing MetaQAGraphEngine via the unified backend API.
    """

    name = "metaqa"

    def __init__(self, graph_engine: MetaQAGraphEngine):
        self.graph = graph_engine

    def find_entity(self, entity_name: str) -> Optional[str]:
        return self.graph.find_entity(entity_name)

    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str]]:
        return self.graph.get_neighbors(entity, relation=relation, direction=direction)

    def get_entity_relations(self, entity: str) -> List[str]:
        return self.graph.get_entity_relations(entity)

    def query_relation(self, entity: str, relation: str) -> List[str]:
        return self.graph.query_relation(entity, relation)

    def query_simple(self, question: str) -> Optional[str]:
        return self.graph.query_simple(question)
