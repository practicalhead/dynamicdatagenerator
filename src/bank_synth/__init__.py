"""
Bank Synth - Synthetic Data Generator for Banking Relational Models

A model-trained framework that generates realistic, relationally consistent
synthetic test data for Oracle and Hive databases.

Features:
- Zero-intervention relationship discovery from SQL queries
- Auto-discovery from ETL/reporting/dynamic queries
- PySpark-based parallel data generation for massive datasets
- Output to Hive tables in ORC format
- Privacy-aware synthetic data generation
"""

__version__ = "0.3.0"
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


# Spark module (lazy import to avoid errors if PySpark not installed)
def get_spark_generator():
    """Get SparkGenerator class (requires PySpark)."""
    from bank_synth.spark import SparkGenerator
    return SparkGenerator


def get_spark_trainer():
    """Get SparkTrainer class (requires PySpark)."""
    from bank_synth.spark import SparkTrainer
    return SparkTrainer


def get_spark_session():
    """Get or create SparkSession (requires PySpark)."""
    from bank_synth.spark import get_spark_session as _get_spark_session
    return _get_spark_session()


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
    # Spark (lazy imports)
    "get_spark_generator",
    "get_spark_trainer",
    "get_spark_session",
]
