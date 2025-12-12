"""
Bank Synth - Synthetic Data Generator for Banking Relational Models

A model-trained framework that generates realistic, relationally consistent
synthetic test data for Oracle and Hive databases.
"""

__version__ = "0.1.0"
__author__ = "DDG Team"

from bank_synth.models import (
    ColumnMetadata,
    TableMetadata,
    Relationship,
    RelationshipGraph,
    ModelPack,
    GenerationConfig,
)

__all__ = [
    "ColumnMetadata",
    "TableMetadata",
    "Relationship",
    "RelationshipGraph",
    "ModelPack",
    "GenerationConfig",
]
