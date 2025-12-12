"""
Spark module for distributed data generation.

Provides:
- SparkSession management
- Parallel data generation using Spark
- Direct Hive table writes in ORC format
"""

from bank_synth.spark.session import SparkSessionManager, get_spark_session
from bank_synth.spark.generator import SparkGenerator
from bank_synth.spark.trainer import SparkTrainer

__all__ = [
    "SparkSessionManager",
    "get_spark_session",
    "SparkGenerator",
    "SparkTrainer",
]
