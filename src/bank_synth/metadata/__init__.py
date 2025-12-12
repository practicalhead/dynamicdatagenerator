"""
Metadata introspection module for Oracle and Hive databases.

Provides unified interface to extract table metadata, column definitions,
and relationships from database catalogs.
"""

from bank_synth.metadata.resolver import MetadataResolver
from bank_synth.metadata.oracle import OracleMetadataExtractor
from bank_synth.metadata.hive import HiveMetadataExtractor

__all__ = [
    "MetadataResolver",
    "OracleMetadataExtractor",
    "HiveMetadataExtractor",
]
