"""
Spark-based trainer for large-scale data sampling and statistics.

Uses Spark to sample and analyze large datasets that don't fit in memory,
computing statistics needed for synthetic data generation.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from bank_synth.models import (
    ColumnStats,
    DataType,
    ModelPack,
    RelationshipGraph,
    TableStats,
)
from bank_synth.spark.session import get_spark_session

logger = logging.getLogger(__name__)


class SparkTrainer:
    """
    Distributed trainer using PySpark.

    Computes statistics and builds models from large-scale datasets
    that don't fit in memory, using Spark's distributed computing.

    Key features:
    - Sample large tables using Spark SQL
    - Compute column statistics in parallel
    - Handle billions of rows efficiently
    - Direct integration with Hive tables
    """

    def __init__(
        self,
        relationship_graph: RelationshipGraph,
        privacy_policy: Optional[Dict[str, Dict[str, str]]] = None,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize Spark trainer.

        Args:
            relationship_graph: Graph of table relationships
            privacy_policy: Privacy policy for columns
            spark: Optional SparkSession (creates one if not provided)
        """
        self.relationship_graph = relationship_graph
        self.privacy_policy = privacy_policy or {}
        self.spark = spark or get_spark_session()

        # Sample data storage (Spark DataFrames)
        self._sample_data: Dict[str, DataFrame] = {}

        # Statistics computed from samples
        self._table_stats: Dict[str, TableStats] = {}

        # Trained synthesizers (fallback models when SDV not available)
        self._synthesizers: Dict[str, bytes] = {}

        # Categorical encoders
        self._encoders: Dict[str, Dict[str, Any]] = {}

    def sample_from_hive(
        self,
        database: str,
        tables: List[str],
        sample_fraction: float = 0.01,
        sample_rows: Optional[int] = None,
    ) -> None:
        """
        Sample data from Hive tables.

        Args:
            database: Hive database name
            tables: List of table names to sample
            sample_fraction: Fraction of data to sample (0.01 = 1%)
            sample_rows: Alternative: specific number of rows per table
        """
        logger.info(f"Sampling from Hive database: {database}")

        for table_name in tables:
            table_upper = table_name.upper()
            full_table = f"{database}.{table_name}"

            try:
                # Read from Hive
                df = self.spark.table(full_table)

                # Sample
                if sample_rows:
                    total_rows = df.count()
                    fraction = min(1.0, sample_rows / total_rows)
                    df = df.sample(fraction=fraction, seed=42)
                else:
                    df = df.sample(fraction=sample_fraction, seed=42)

                # Cache the sample
                df = df.cache()
                row_count = df.count()

                self._sample_data[table_upper] = df
                logger.info(f"Sampled {row_count:,} rows from {full_table}")

            except Exception as e:
                logger.error(f"Error sampling {full_table}: {e}")

    def sample_from_oracle(
        self,
        connection_string: str,
        schema: str,
        tables: List[str],
        sample_rows: int = 100000,
    ) -> None:
        """
        Sample data from Oracle using JDBC.

        Args:
            connection_string: Oracle JDBC connection string
            schema: Oracle schema name
            tables: List of table names to sample
            sample_rows: Number of rows to sample per table
        """
        logger.info(f"Sampling from Oracle schema: {schema}")

        jdbc_url = f"jdbc:oracle:thin:@{connection_string}"

        for table_name in tables:
            table_upper = table_name.upper()
            full_table = f"{schema}.{table_name}"

            try:
                # Sample query using Oracle's SAMPLE clause
                query = f"(SELECT * FROM {full_table} SAMPLE({min(100, sample_rows / 100)}) WHERE ROWNUM <= {sample_rows}) sample_table"

                df = (
                    self.spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", query)
                    .option("driver", "oracle.jdbc.driver.OracleDriver")
                    .load()
                )

                # Normalize column names to uppercase
                for col in df.columns:
                    df = df.withColumnRenamed(col, col.upper())

                df = df.cache()
                row_count = df.count()

                self._sample_data[table_upper] = df
                logger.info(f"Sampled {row_count:,} rows from {full_table}")

            except Exception as e:
                logger.error(f"Error sampling {full_table}: {e}")

    def sample_from_parquet(
        self,
        data_dir: Path,
        tables: Optional[List[str]] = None,
        sample_fraction: float = 0.1,
    ) -> None:
        """
        Sample data from Parquet files.

        Args:
            data_dir: Directory containing Parquet files
            tables: Optional list of tables (auto-discovers if not provided)
            sample_fraction: Fraction of data to sample
        """
        data_dir = Path(data_dir)
        logger.info(f"Sampling from Parquet files in: {data_dir}")

        # Find Parquet files
        if tables:
            parquet_files = [(t, data_dir / f"{t.lower()}.parquet") for t in tables]
        else:
            parquet_files = [(f.stem.upper(), f) for f in data_dir.glob("**/*.parquet")]

        for table_name, file_path in parquet_files:
            if not file_path.exists():
                logger.warning(f"Parquet file not found: {file_path}")
                continue

            try:
                df = self.spark.read.parquet(str(file_path))

                if sample_fraction < 1.0:
                    df = df.sample(fraction=sample_fraction, seed=42)

                # Normalize column names to uppercase
                for col in df.columns:
                    df = df.withColumnRenamed(col, col.upper())

                df = df.cache()
                row_count = df.count()

                self._sample_data[table_name] = df
                logger.info(f"Sampled {row_count:,} rows from {file_path}")

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    def compute_statistics(self) -> Dict[str, TableStats]:
        """
        Compute statistics for all sampled tables using Spark.

        Returns:
            Dict of table_name -> TableStats
        """
        logger.info("Computing statistics using Spark")

        for table_name, df in self._sample_data.items():
            logger.info(f"Computing statistics for {table_name}")
            stats = self._compute_table_stats_spark(table_name, df)
            self._table_stats[table_name] = stats

        return self._table_stats

    def _compute_table_stats_spark(self, table_name: str, df: DataFrame) -> TableStats:
        """Compute statistics for a single table using Spark."""
        row_count = df.count()
        column_stats = {}

        for col_name in df.columns:
            col_stats = self._compute_column_stats_spark(df, col_name)
            column_stats[col_name] = col_stats

        return TableStats(
            name=table_name,
            row_count=row_count,
            column_stats=column_stats,
        )

    def _compute_column_stats_spark(self, df: DataFrame, col_name: str) -> ColumnStats:
        """Compute statistics for a single column using Spark."""
        col = F.col(col_name)
        total_count = df.count()

        # Compute null count
        null_count = df.filter(col.isNull()).count()
        null_fraction = null_count / total_count if total_count > 0 else 0

        # Compute distinct count (approximate for large datasets)
        distinct_count = df.select(F.approx_count_distinct(col_name)).collect()[0][0]

        # Determine data type
        spark_type = df.schema[col_name].dataType
        data_type = self._spark_type_to_datatype(spark_type)

        # Compute type-specific statistics
        stats_row = None
        min_val = None
        max_val = None
        mean = None
        std = None
        value_frequencies = None

        if isinstance(spark_type, (IntegerType, LongType, FloatType, DoubleType, DecimalType)):
            # Numeric statistics
            stats_row = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.stddev(col).alias("std"),
            ).collect()[0]

            min_val = float(stats_row["min"]) if stats_row["min"] is not None else None
            max_val = float(stats_row["max"]) if stats_row["max"] is not None else None
            mean = float(stats_row["mean"]) if stats_row["mean"] is not None else None
            std = float(stats_row["std"]) if stats_row["std"] is not None else None

        elif isinstance(spark_type, (DateType, TimestampType)):
            # Date statistics
            stats_row = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
            ).collect()[0]
            min_val = str(stats_row["min"]) if stats_row["min"] else None
            max_val = str(stats_row["max"]) if stats_row["max"] else None

        # Compute value frequencies for low-cardinality columns
        if distinct_count <= 100:
            freq_rows = (
                df.groupBy(col_name)
                .count()
                .withColumn("freq", F.col("count") / F.lit(total_count))
                .collect()
            )
            value_frequencies = {
                row[col_name]: row["freq"]
                for row in freq_rows
                if row[col_name] is not None
            }

        return ColumnStats(
            name=col_name,
            data_type=data_type,
            null_fraction=null_fraction,
            distinct_count=distinct_count,
            total_count=total_count,
            min_value=min_val,
            max_value=max_val,
            mean=mean,
            std=std,
            value_frequencies=value_frequencies,
            is_unique=distinct_count == total_count,
        )

    def _spark_type_to_datatype(self, spark_type) -> DataType:
        """Map Spark SQL type to DataType."""
        if isinstance(spark_type, StringType):
            return DataType.STRING
        elif isinstance(spark_type, IntegerType):
            return DataType.INTEGER
        elif isinstance(spark_type, LongType):
            return DataType.BIGINT
        elif isinstance(spark_type, DecimalType):
            return DataType.DECIMAL
        elif isinstance(spark_type, FloatType):
            return DataType.FLOAT
        elif isinstance(spark_type, DoubleType):
            return DataType.DOUBLE
        elif isinstance(spark_type, DateType):
            return DataType.DATE
        elif isinstance(spark_type, TimestampType):
            return DataType.TIMESTAMP
        else:
            return DataType.UNKNOWN

    def build_fallback_models(self) -> Dict[str, bytes]:
        """
        Build fallback synthesizer models from computed statistics.

        These models are simpler than SDV but work at scale and don't
        require fitting on the full dataset.

        Returns:
            Dict of table_name -> serialized model bytes
        """
        logger.info("Building fallback synthesizer models")

        for table_name, df in self._sample_data.items():
            logger.info(f"Building model for {table_name}")
            model = self._build_fallback_model_spark(table_name, df)
            self._synthesizers[table_name] = pickle.dumps(model)

        return self._synthesizers

    def _build_fallback_model_spark(
        self,
        table_name: str,
        df: DataFrame,
    ) -> Dict[str, Any]:
        """Build a fallback synthesizer model from Spark DataFrame."""
        stats = self._table_stats.get(table_name)
        if not stats:
            return {"_type": "fallback", "columns": {}}

        model = {
            "_type": "fallback",
            "table_name": table_name,
            "columns": {},
        }

        for col_name in df.columns:
            col_stats = stats.column_stats.get(col_name)
            if not col_stats:
                continue

            col_model = {
                "name": col_name,
                "dtype": col_stats.data_type.value,
                "null_fraction": col_stats.null_fraction,
            }

            # Store value distribution for categorical columns
            if col_stats.value_frequencies:
                # Normalize frequencies
                total = sum(col_stats.value_frequencies.values())
                col_model["value_counts"] = {
                    k: v / total for k, v in col_stats.value_frequencies.items()
                }

            # Store numeric distribution
            elif col_stats.mean is not None:
                col_model["mean"] = col_stats.mean
                col_model["std"] = col_stats.std or 1.0
                col_model["min"] = col_stats.min_value
                col_model["max"] = col_stats.max_value

            # Store sample values for string columns
            elif col_stats.data_type == DataType.STRING:
                sample_values = (
                    df.select(col_name)
                    .filter(F.col(col_name).isNotNull())
                    .limit(1000)
                    .distinct()
                    .collect()
                )
                col_model["sample_values"] = [row[0] for row in sample_values]

            model["columns"][col_name] = col_model

        return model

    def train(
        self,
        schema_name: str,
        sample_strategy: str = "percent:1",
    ) -> ModelPack:
        """
        Train models and create ModelPack.

        Args:
            schema_name: Schema/database name
            sample_strategy: Sampling strategy used

        Returns:
            ModelPack with trained models
        """
        logger.info("Training models using Spark")

        # Compute statistics if not already done
        if not self._table_stats:
            self.compute_statistics()

        # Build fallback models
        self.build_fallback_models()

        # Create model pack
        model_pack = ModelPack(
            schema_name=schema_name,
            relationship_graph=self.relationship_graph,
            tables_trained=list(self._sample_data.keys()),
            sample_strategy=sample_strategy,
            training_config={"engine": "spark"},
            table_stats=self._table_stats,
            synthesizers=self._synthesizers,
            encoders=self._encoders,
            privacy_policy=self.privacy_policy,
        )

        return model_pack

    def cleanup(self):
        """Clean up cached Spark DataFrames."""
        for df in self._sample_data.values():
            df.unpersist()
        self._sample_data.clear()
        logger.info("Cleaned up cached Spark DataFrames")
