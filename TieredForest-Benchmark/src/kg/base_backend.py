from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseKGBackend(ABC):
    """
    Pluggable KG backend interface used by Tiered-Forest / ToG agents.
    """

    name = "base"

    @abstractmethod
    def find_entity(self, entity_name: str) -> Optional[str]:
        """Resolve a surface name/id to a backend-specific canonical entity id."""
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str]]:
        """
        Return neighbors as (neighbor_label_or_id, relation_name).
        direction in {'out', 'in', 'both'}.
        """
        raise NotImplementedError

    @abstractmethod
    def get_entity_relations(self, entity: str) -> List[str]:
        """Return relation names connected with entity."""
        raise NotImplementedError

    @abstractmethod
    def query_relation(self, entity: str, relation: str) -> List[str]:
        """Query neighbors reachable by a specific relation."""
        raise NotImplementedError

    def query_simple(self, question: str) -> Optional[str]:
        """
        Optional fast-path QA for symbolic tier.
        Backends can override. Default is no-op.
        """
        return None
