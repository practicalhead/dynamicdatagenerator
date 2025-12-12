"""
Spark-based parallel data generator.

Generates massive synthetic datasets using Spark's distributed computing,
enabling parallel generation across a cluster for millions/billions of rows.
"""

from __future__ import annotations

import logging
import pickle
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    GenerationConfig,
    ModelPack,
    PrivacyLevel,
    Relationship,
    TableMetadata,
)
from bank_synth.spark.session import get_spark_session, configure_for_large_scale

logger = logging.getLogger(__name__)


class SparkGenerator:
    """
    Distributed data generator using PySpark.

    Generates synthetic data in parallel across Spark executors, enabling
    generation of massive datasets (millions to billions of rows).

    Key features:
    - Parallel row generation across executors
    - Broadcast join for FK constraint application
    - Direct write to Hive tables in ORC format
    - Configurable partitioning for optimal performance
    """

    # Map DataType to Spark SQL types
    TYPE_MAP = {
        DataType.STRING: StringType(),
        DataType.INTEGER: IntegerType(),
        DataType.BIGINT: LongType(),
        DataType.DECIMAL: DecimalType(18, 2),
        DataType.FLOAT: FloatType(),
        DataType.DOUBLE: DoubleType(),
        DataType.DATE: DateType(),
        DataType.TIMESTAMP: TimestampType(),
        DataType.BOOLEAN: BooleanType(),
        DataType.BINARY: BinaryType(),
        DataType.UNKNOWN: StringType(),
    }

    def __init__(
        self,
        model_pack: ModelPack,
        config: GenerationConfig,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize Spark generator.

        Args:
            model_pack: Trained model pack with synthesizers
            config: Generation configuration
            spark: Optional SparkSession (creates one if not provided)
        """
        self.model_pack = model_pack
        self.config = config
        self.spark = spark or get_spark_session()

        # Configure for large-scale generation
        configure_for_large_scale(self.spark, config.target_rows)

        # Set seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)

        # Generated data storage (Spark DataFrames)
        self._generated_data: Dict[str, DataFrame] = {}

        # Broadcast parent keys for FK sampling
        self._broadcast_keys: Dict[str, Any] = {}

    def generate(self) -> Dict[str, DataFrame]:
        """
        Generate synthetic data for target table and dependencies.

        Uses Spark's parallel processing to generate data across executors.

        Returns:
            Dictionary mapping table names to Spark DataFrames
        """
        logger.info(f"Spark generating data for target: {self.config.target_table}")
        logger.info(f"Spark parallelism: {self.spark.sparkContext.defaultParallelism}")

        # Get dependency closure and generation order
        closure, sorted_tables = self.model_pack.relationship_graph.get_dependency_closure(
            self.config.target_table,
            include_children=self.config.include_children,
        )

        logger.info(f"Dependency closure: {len(closure)} tables")
        logger.info(f"Generation order: {sorted_tables}")

        # Calculate row counts for each table
        row_counts = self._calculate_row_counts(sorted_tables)

        # Generate tables in dependency order
        for table_name in sorted_tables:
            table_upper = table_name.upper()
            target_rows = row_counts.get(table_upper, self.config.target_rows)

            logger.info(f"Generating {target_rows:,} rows for {table_name} using Spark")

            df = self._generate_table_spark(table_upper, target_rows)
            self._generated_data[table_upper] = df

            # Broadcast keys for child FK sampling
            self._broadcast_parent_keys(table_upper, df)

        return self._generated_data

    def _calculate_row_counts(self, tables: List[str]) -> Dict[str, int]:
        """Calculate row counts for each table in generation order."""
        row_counts = {}
        target_upper = self.config.target_table.upper()

        for table_name in tables:
            table_upper = table_name.upper()

            if table_upper in self.config.table_row_counts:
                row_counts[table_upper] = self.config.table_row_counts[table_upper]
            elif table_upper == target_upper:
                row_counts[table_upper] = self.config.target_rows
            else:
                table_meta = self.model_pack.relationship_graph.get_table(table_name)
                if table_meta and table_meta.is_reference_table:
                    stats = self.model_pack.table_stats.get(table_upper)
                    if stats:
                        row_counts[table_upper] = min(stats.row_count, 1000)
                    else:
                        row_counts[table_upper] = 100
                else:
                    scaled = int(self.config.target_rows * self.config.parent_scale_factor)
                    row_counts[table_upper] = max(scaled, 10)

        return row_counts

    def _generate_table_spark(self, table_name: str, num_rows: int) -> DataFrame:
        """Generate data for a single table using Spark."""
        table_meta = self.model_pack.relationship_graph.get_table(table_name)

        # Calculate optimal number of partitions
        partitions = self._calculate_partitions(num_rows)

        # Generate base data
        df = self._generate_base_data_spark(table_name, num_rows, table_meta, partitions)

        # Apply FK constraints using broadcast join
        df = self._apply_fk_constraints_spark(table_name, df)

        # Apply privacy transformations
        df = self._apply_privacy_spark(table_name, df, table_meta)

        # Ensure PK uniqueness
        df = self._ensure_pk_uniqueness_spark(table_name, df, table_meta)

        # Cache the result for FK lookups
        df = df.cache()
        df.count()  # Materialize the cache

        return df

    def _calculate_partitions(self, num_rows: int) -> int:
        """Calculate optimal partition count for given row count."""
        # Aim for ~500K-1M rows per partition
        rows_per_partition = 750_000
        partitions = max(1, num_rows // rows_per_partition)
        # Cap at cluster parallelism
        max_partitions = self.spark.sparkContext.defaultParallelism * 2
        return min(partitions, max_partitions)

    def _generate_base_data_spark(
        self,
        table_name: str,
        num_rows: int,
        table_meta: Optional[TableMetadata],
        partitions: int,
    ) -> DataFrame:
        """Generate base data using Spark's parallel execution."""
        if not table_meta or not table_meta.columns:
            # Minimal fallback
            return self.spark.range(0, num_rows, 1, partitions).withColumn(
                "ID", F.col("id") + 1
            ).drop("id")

        # Build schema from metadata
        schema = self._build_spark_schema(table_meta)

        # Create a range DataFrame as the base
        base_df = self.spark.range(0, num_rows, 1, partitions)

        # Generate each column using Spark UDFs and functions
        for col_meta in table_meta.columns:
            col_name = col_meta.name.upper()
            base_df = self._add_column_spark(base_df, col_meta)

        # Drop the base id column
        return base_df.drop("id")

    def _build_spark_schema(self, table_meta: TableMetadata) -> StructType:
        """Build Spark schema from table metadata."""
        fields = []
        for col_meta in table_meta.columns:
            spark_type = self.TYPE_MAP.get(col_meta.data_type, StringType())
            fields.append(StructField(col_meta.name.upper(), spark_type, col_meta.nullable))
        return StructType(fields)

    def _add_column_spark(self, df: DataFrame, col_meta: ColumnMetadata) -> DataFrame:
        """Add a generated column to the DataFrame."""
        col_name = col_meta.name.upper()
        data_type = col_meta.data_type

        # Handle allowed values (categorical)
        if col_meta.allowed_values:
            values = col_meta.allowed_values
            # Use modulo to select from allowed values
            df = df.withColumn(
                col_name,
                F.element_at(
                    F.array(*[F.lit(v) for v in values]),
                    (F.abs(F.hash(F.col("id"))) % len(values)) + 1
                )
            )

        # Handle primary key columns
        elif col_meta.is_primary_key:
            if data_type in (DataType.INTEGER, DataType.BIGINT):
                df = df.withColumn(col_name, F.col("id") + 1)
            else:
                df = df.withColumn(col_name, F.concat(F.lit("PK_"), F.col("id").cast(StringType())))

        # Handle numeric types
        elif data_type in (DataType.INTEGER, DataType.BIGINT):
            if col_meta.value_range:
                min_val, max_val = col_meta.value_range
            else:
                min_val, max_val = 1, 1000000
            range_size = max_val - min_val
            df = df.withColumn(
                col_name,
                (F.abs(F.hash(F.col("id"), F.lit(col_name))) % range_size + min_val).cast(
                    LongType() if data_type == DataType.BIGINT else IntegerType()
                )
            )

        elif data_type in (DataType.DECIMAL, DataType.FLOAT, DataType.DOUBLE):
            if col_meta.value_range:
                min_val, max_val = col_meta.value_range
            else:
                min_val, max_val = 0.0, 10000.0
            range_size = max_val - min_val
            df = df.withColumn(
                col_name,
                (F.rand(self.config.seed) * range_size + min_val).cast(DoubleType())
            )
            if col_meta.scale:
                df = df.withColumn(col_name, F.round(F.col(col_name), col_meta.scale))

        # Handle date types
        elif data_type == DataType.DATE:
            # Generate dates within last 5 years
            days_range = 365 * 5
            df = df.withColumn(
                col_name,
                F.date_sub(
                    F.current_date(),
                    (F.abs(F.hash(F.col("id"), F.lit(col_name))) % days_range).cast(IntegerType())
                )
            )

        elif data_type == DataType.TIMESTAMP:
            # Generate timestamps within last 5 years
            seconds_range = 365 * 5 * 24 * 3600
            df = df.withColumn(
                col_name,
                F.from_unixtime(
                    F.unix_timestamp() - (F.abs(F.hash(F.col("id"), F.lit(col_name))) % seconds_range)
                )
            )

        elif data_type == DataType.BOOLEAN:
            df = df.withColumn(
                col_name,
                (F.abs(F.hash(F.col("id"), F.lit(col_name))) % 2 == 0)
            )

        # Handle string types
        else:
            max_len = col_meta.max_length or 50
            # Generate pseudo-random strings using hash
            df = df.withColumn(
                col_name,
                F.substring(F.md5(F.concat(F.col("id").cast(StringType()), F.lit(col_name))), 1, min(max_len, 32))
            )

        # Apply nulls if column is nullable
        if col_meta.nullable and not col_meta.is_primary_key:
            null_fraction = 0.05
            df = df.withColumn(
                col_name,
                F.when(F.rand(self.config.seed) < null_fraction, F.lit(None)).otherwise(F.col(col_name))
            )

        return df

    def _apply_fk_constraints_spark(self, table_name: str, df: DataFrame) -> DataFrame:
        """Apply foreign key constraints using broadcast join."""
        for rel in self.model_pack.relationship_graph.relationships:
            if rel.child_table.upper() != table_name.upper():
                continue

            parent_upper = rel.parent_table.upper()
            if parent_upper not in self._broadcast_keys:
                logger.warning(f"Parent keys not available for {parent_upper}")
                continue

            # Get broadcast keys
            broadcast_data = self._broadcast_keys[parent_upper]

            for child_col, parent_col in zip(rel.child_columns, rel.parent_columns):
                child_col_upper = child_col.upper()
                parent_col_upper = parent_col.upper()

                if parent_col_upper not in broadcast_data:
                    continue

                parent_values = broadcast_data[parent_col_upper]
                if not parent_values:
                    continue

                # Create a broadcast variable with parent keys
                num_parents = len(parent_values)
                parent_keys_broadcast = self.spark.sparkContext.broadcast(parent_values)

                # Use UDF to sample from parent keys
                @F.udf(returnType=self._get_spark_type_for_values(parent_values))
                def sample_fk(row_id):
                    keys = parent_keys_broadcast.value
                    idx = abs(hash(row_id)) % len(keys)
                    return keys[idx]

                df = df.withColumn(child_col_upper, sample_fk(F.col("id") if "id" in df.columns else F.monotonically_increasing_id()))

                # Handle optional relationships (some nulls)
                if rel.is_optional:
                    df = df.withColumn(
                        child_col_upper,
                        F.when(F.rand(self.config.seed) < 0.05, F.lit(None)).otherwise(F.col(child_col_upper))
                    )

        return df

    def _get_spark_type_for_values(self, values: List[Any]):
        """Infer Spark type from sample values."""
        if not values:
            return StringType()
        sample = values[0]
        if isinstance(sample, int):
            return LongType()
        elif isinstance(sample, float):
            return DoubleType()
        elif isinstance(sample, bool):
            return BooleanType()
        else:
            return StringType()

    def _broadcast_parent_keys(self, table_name: str, df: DataFrame) -> None:
        """Extract and broadcast parent key values for FK sampling."""
        table_meta = self.model_pack.relationship_graph.get_table(table_name)
        if not table_meta:
            return

        self._broadcast_keys[table_name.upper()] = {}

        # Extract PK columns
        for pk_col in table_meta.primary_key:
            pk_col_upper = pk_col.upper()
            if pk_col_upper in df.columns:
                # Collect to driver and broadcast
                values = [row[0] for row in df.select(pk_col_upper).distinct().collect()]
                self._broadcast_keys[table_name.upper()][pk_col_upper] = values

        # Also extract columns that are referenced by FKs
        for rel in self.model_pack.relationship_graph.relationships:
            if rel.parent_table.upper() == table_name.upper():
                for parent_col in rel.parent_columns:
                    parent_col_upper = parent_col.upper()
                    if parent_col_upper in df.columns and parent_col_upper not in self._broadcast_keys.get(table_name.upper(), {}):
                        values = [row[0] for row in df.select(parent_col_upper).distinct().collect()]
                        self._broadcast_keys[table_name.upper()][parent_col_upper] = values

    def _apply_privacy_spark(
        self,
        table_name: str,
        df: DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> DataFrame:
        """Apply privacy transformations using Spark."""
        privacy_policy = self.model_pack.privacy_policy.get(table_name.lower(), {})

        for col_name in df.columns:
            if col_name == "id":
                continue

            col_meta = table_meta.get_column(col_name) if table_meta else None
            col_policy = privacy_policy.get(col_name.lower(), {})

            privacy_level = PrivacyLevel.PUBLIC
            format_pattern = None

            if col_meta:
                privacy_level = col_meta.privacy_level
                format_pattern = col_meta.format_pattern

            if isinstance(col_policy, str):
                privacy_level = PrivacyLevel(col_policy.lower())
            elif isinstance(col_policy, dict):
                if "level" in col_policy:
                    privacy_level = PrivacyLevel(col_policy["level"].lower())
                if "format_pattern" in col_policy:
                    format_pattern = col_policy["format_pattern"]

            if privacy_level == PrivacyLevel.RESTRICTED:
                df = df.drop(col_name)
            elif privacy_level == PrivacyLevel.PII:
                # Generate random data for PII columns
                df = df.withColumn(
                    col_name,
                    F.concat(F.lit("PII_"), F.substring(F.md5(F.col(col_name).cast(StringType())), 1, 10))
                )
            elif privacy_level == PrivacyLevel.SENSITIVE and format_pattern:
                # Apply format pattern
                df = df.withColumn(
                    col_name,
                    self._apply_format_pattern_spark(F.col(col_name), format_pattern)
                )

        return df

    def _apply_format_pattern_spark(self, col, pattern: str):
        """Apply format pattern to column using Spark."""
        # Simplified pattern application using hash
        return F.substring(F.md5(col.cast(StringType())), 1, len(pattern))

    def _ensure_pk_uniqueness_spark(
        self,
        table_name: str,
        df: DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> DataFrame:
        """Ensure primary key columns have unique values."""
        if not table_meta or not table_meta.primary_key:
            return df

        for pk_col in table_meta.primary_key:
            pk_col_upper = pk_col.upper()
            if pk_col_upper in df.columns:
                col_meta = table_meta.get_column(pk_col)
                if col_meta and col_meta.data_type in (DataType.INTEGER, DataType.BIGINT):
                    # Use monotonically increasing ID
                    df = df.withColumn(pk_col_upper, F.monotonically_increasing_id() + 1)
                else:
                    # Use UUID for strings
                    df = df.withColumn(pk_col_upper, F.expr("uuid()"))

        return df

    def write_to_hive(
        self,
        database: Optional[str] = None,
        format: str = "orc",
        mode: str = "overwrite",
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, str]:
        """
        Write generated data directly to Hive tables.

        This is the recommended way to output large-scale generated data,
        as it uses Spark's native Hive integration for optimal performance.

        Args:
            database: Hive database name (uses schema_name if not provided)
            format: Storage format ("orc" or "parquet")
            mode: Write mode ("overwrite", "append", "error", "ignore")
            partition_by: Optional dict of table_name -> partition columns

        Returns:
            Dict of table_name -> Hive table path
        """
        db_name = database or self.model_pack.schema_name or "default"

        # Create database if not exists
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        output_tables = {}

        for table_name, df in self._generated_data.items():
            table_lower = table_name.lower()
            full_table_name = f"{db_name}.{table_lower}"

            # Rename columns to lowercase for Hive compatibility
            for col in df.columns:
                df = df.withColumnRenamed(col, col.lower())

            # Get partition columns
            partition_cols = partition_by.get(table_name) if partition_by else None

            # Write to Hive
            writer = df.write.mode(mode).format(format)

            if partition_cols:
                writer = writer.partitionBy(*partition_cols)

            writer.saveAsTable(full_table_name)

            output_tables[table_name] = full_table_name
            logger.info(f"Wrote {df.count():,} rows to Hive table {full_table_name}")

        return output_tables

    def to_pandas(self) -> Dict[str, "pd.DataFrame"]:
        """
        Convert generated Spark DataFrames to pandas.

        Warning: Only use for small datasets that fit in driver memory.

        Returns:
            Dict of table_name -> pandas DataFrame
        """
        return {name: df.toPandas() for name, df in self._generated_data.items()}

    def get_generated_data(self) -> Dict[str, DataFrame]:
        """Return all generated Spark DataFrames."""
        return self._generated_data
