"""
Bank Synth - Synthetic Data Generator for Banking Relational Models

A model-trained framework that generates realistic, relationally consistent
synthetic test data for Oracle and Hive databases.

Features:
- Zero-intervention relationship discovery from SQL queries
- Auto-discovery from ETL/reporting/dynamic queries
- Output to Hive tables in ORC format
- Privacy-aware synthetic data generation
"""

__version__ = "0.2.0"
__author__ = "DDG Team"

from bank_synth.models import (
    ColumnMetadata,
    TableMetadata,
    Relationship,
    RelationshipGraph,
    ModelPack,
    GenerationConfig,
)

# Import auto-discovery module
from bank_synth.discovery import (
    AutoDiscoveryPipeline,
    QueryParser,
    RelationshipInferrer,
    auto_discover,
)

# Import output module
from bank_synth.output import (
    OutputWriter,
    HiveOrcWriter,
    HiveParquetWriter,
)

__all__ = [
    # Core models
    "ColumnMetadata",
    "TableMetadata",
    "Relationship",
    "RelationshipGraph",
    "ModelPack",
    "GenerationConfig",
    # Discovery
    "AutoDiscoveryPipeline",
    "QueryParser",
    "RelationshipInferrer",
    "auto_discover",
    # Output
    "OutputWriter",
    "HiveOrcWriter",
    "HiveParquetWriter",
]
