"""
Auto-discovery module for extracting table relationships from SQL queries and metadata.

This module provides tools to automatically discover table relationships without
requiring manual configuration, using:
- SQL query analysis (ETL, reporting, dynamic queries)
- PK/FK metadata from database catalogs
- Pattern-based inference

Zero-intervention usage:
    from bank_synth.discovery import auto_discover

    graph = auto_discover(
        schema="CORE",
        queries="./etl_queries/",
    )
"""

from bank_synth.discovery.query_parser import QueryParser, ParsedQuery, JoinInfo
from bank_synth.discovery.relationship_inferrer import RelationshipInferrer
from bank_synth.discovery.auto_discovery import AutoDiscoveryPipeline, auto_discover

__all__ = [
    "QueryParser",
    "ParsedQuery",
    "JoinInfo",
    "RelationshipInferrer",
    "AutoDiscoveryPipeline",
    "auto_discover",
]
