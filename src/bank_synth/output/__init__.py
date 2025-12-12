"""
Output module for writing generated data to various formats.

Supports:
- Hive tables (ORC format)
- Hive tables (Parquet format)
- Oracle-compatible CSV files
- DDL scripts for loading
"""

from bank_synth.output.writer import OutputWriter
from bank_synth.output.hive_orc import HiveOrcWriter
from bank_synth.output.hive_parquet import HiveParquetWriter

__all__ = [
    "OutputWriter",
    "HiveOrcWriter",
    "HiveParquetWriter",
]
