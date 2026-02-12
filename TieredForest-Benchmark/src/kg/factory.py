from typing import Optional

from ..graph_engine import MetaQAGraphEngine
from .base_backend import BaseKGBackend
from .freebase_backend import FreebaseBackend
from .metaqa_backend import MetaQABackend
from .wikidata_backend import WikidataBackend


def create_kg_backend(
    backend_type: str,
    graph_engine: Optional[MetaQAGraphEngine] = None,
    **kwargs,
) -> BaseKGBackend:
    """
    Build a KG backend by type:
    - metaqa
    - wikidata
    - freebase
    """
    key = backend_type.strip().lower()
    if key == "metaqa":
        if graph_engine is None:
            raise ValueError("graph_engine is required for metaqa backend")
        return MetaQABackend(graph_engine)
    if key == "wikidata":
        return WikidataBackend(**kwargs)
    if key == "freebase":
        return FreebaseBackend(**kwargs)
    raise ValueError(f"Unsupported backend_type: {backend_type}")
