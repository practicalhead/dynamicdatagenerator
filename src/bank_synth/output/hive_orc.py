"""
Hive ORC Writer - Write generated data to Hive tables in ORC format.

ORC (Optimized Row Columnar) is a highly efficient columnar storage format
for Hive that provides:
- Excellent compression
- Predicate pushdown for fast queries
- Type-specific encodings
- Built-in indexing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from bank_synth.models import DataType, GenerationConfig, ModelPack, TableMetadata

logger = logging.getLogger(__name__)


class HiveOrcWriter:
    """
    Writes generated data to ORC format for Hive tables.

    Features:
    - Writes data in ORC format using PyArrow
    - Generates Hive-compatible DDL for table creation
    - Supports partitioning
    - Handles data type mapping to Hive types
    """

    # Map DataType to Hive type strings
    HIVE_TYPE_MAP = {
        DataType.STRING: "STRING",
        DataType.INTEGER: "INT",
        DataType.BIGINT: "BIGINT",
        DataType.DECIMAL: "DECIMAL(18,2)",
        DataType.FLOAT: "FLOAT",
        DataType.DOUBLE: "DOUBLE",
        DataType.DATE: "DATE",
        DataType.TIMESTAMP: "TIMESTAMP",
        DataType.BOOLEAN: "BOOLEAN",
        DataType.BINARY: "BINARY",
        DataType.UNKNOWN: "STRING",
    }

    def __init__(
        self,
        output_dir: Path,
        model_pack: ModelPack,
        compression: str = "snappy",
        stripe_size: int = 64 * 1024 * 1024,  # 64MB default
    ):
        """
        Initialize the ORC writer.

        Args:
            output_dir: Base output directory
            model_pack: Model pack with table metadata
            compression: ORC compression codec (snappy, zlib, lz4, zstd, none)
            stripe_size: ORC stripe size in bytes
        """
        self.output_dir = Path(output_dir)
        self.model_pack = model_pack
        self.compression = compression
        self.stripe_size = stripe_size

        # Create output directories
        self.orc_dir = self.output_dir / "hive_orc"
        self.ddl_dir = self.output_dir / "ddl"
        self.orc_dir.mkdir(parents=True, exist_ok=True)
        self.ddl_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Path]:
        """
        Write all generated data to ORC files.

        Args:
            data: Dict of table_name -> DataFrame
            partition_by: Optional dict of table_name -> partition columns

        Returns:
            Dict of table_name -> output path
        """
        output_paths = {}

        for table_name, df in data.items():
            partition_cols = partition_by.get(table_name) if partition_by else None
            output_path = self.write_table(table_name, df, partition_cols)
            output_paths[table_name] = output_path

        # Generate DDL
        self._generate_ddl(data, partition_by)

        logger.info(f"Wrote {len(output_paths)} tables to ORC format")
        return output_paths

    def write_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        partition_by: Optional[List[str]] = None,
    ) -> Path:
        """
        Write a single table to ORC format.

        Args:
            table_name: Name of the table
            df: DataFrame to write
            partition_by: Optional list of partition columns

        Returns:
            Path to output file/directory
        """
        import pyarrow as pa
        import pyarrow.orc as orc

        table_dir = self.orc_dir / table_name.lower()
        table_dir.mkdir(parents=True, exist_ok=True)

        # Get table metadata for schema
        table_meta = self.model_pack.relationship_graph.get_table(table_name)

        # Convert DataFrame to Arrow table with proper schema
        arrow_table = self._df_to_arrow(df, table_meta)

        if partition_by:
            # Write partitioned data
            self._write_partitioned(arrow_table, table_dir, partition_by)
            output_path = table_dir
        else:
            # Write single ORC file
            output_path = table_dir / f"{table_name.lower()}.orc"
            orc.write_table(
                arrow_table,
                str(output_path),
                compression=self.compression.upper() if self.compression != "none" else "UNCOMPRESSED",
                stripe_size=self.stripe_size,
            )

        logger.info(f"Wrote {len(df)} rows to {output_path}")
        return output_path

    def _df_to_arrow(
        self,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> "pa.Table":
        """Convert DataFrame to Arrow table with proper schema."""
        import pyarrow as pa

        # Build schema from metadata if available
        if table_meta:
            fields = []
            for col_meta in table_meta.columns:
                if col_meta.name.upper() in [c.upper() for c in df.columns]:
                    arrow_type = self._datatype_to_arrow(col_meta.data_type)
                    fields.append(pa.field(col_meta.name.lower(), arrow_type, nullable=col_meta.nullable))

            # Add any columns not in metadata
            df_cols_upper = {c.upper() for c in df.columns}
            meta_cols_upper = {c.name.upper() for c in table_meta.columns}
            for col in df.columns:
                if col.upper() not in meta_cols_upper:
                    # Infer type from pandas
                    dtype = df[col].dtype
                    arrow_type = self._pandas_dtype_to_arrow(dtype)
                    fields.append(pa.field(col.lower(), arrow_type, nullable=True))

            schema = pa.schema(fields)

            # Rename columns to match schema
            df_renamed = df.copy()
            df_renamed.columns = [c.lower() for c in df_renamed.columns]

            return pa.Table.from_pandas(df_renamed, schema=schema, preserve_index=False)
        else:
            # Convert without explicit schema
            df_renamed = df.copy()
            df_renamed.columns = [c.lower() for c in df_renamed.columns]
            return pa.Table.from_pandas(df_renamed, preserve_index=False)

    def _datatype_to_arrow(self, data_type: DataType) -> "pa.DataType":
        """Map DataType to Arrow type."""
        import pyarrow as pa

        mapping = {
            DataType.STRING: pa.string(),
            DataType.INTEGER: pa.int32(),
            DataType.BIGINT: pa.int64(),
            DataType.DECIMAL: pa.decimal128(18, 2),
            DataType.FLOAT: pa.float32(),
            DataType.DOUBLE: pa.float64(),
            DataType.DATE: pa.date32(),
            DataType.TIMESTAMP: pa.timestamp("us"),
            DataType.BOOLEAN: pa.bool_(),
            DataType.BINARY: pa.binary(),
            DataType.UNKNOWN: pa.string(),
        }
        return mapping.get(data_type, pa.string())

    def _pandas_dtype_to_arrow(self, dtype) -> "pa.DataType":
        """Map pandas dtype to Arrow type."""
        import pyarrow as pa

        dtype_str = str(dtype).lower()

        if "int64" in dtype_str:
            return pa.int64()
        elif "int32" in dtype_str:
            return pa.int32()
        elif "float64" in dtype_str:
            return pa.float64()
        elif "float32" in dtype_str:
            return pa.float32()
        elif "datetime" in dtype_str:
            return pa.timestamp("us")
        elif "bool" in dtype_str:
            return pa.bool_()
        else:
            return pa.string()

    def _write_partitioned(
        self,
        table: "pa.Table",
        output_dir: Path,
        partition_by: List[str],
    ) -> None:
        """Write partitioned ORC data."""
        import pyarrow as pa
        import pyarrow.orc as orc

        # Get unique partition values
        df = table.to_pandas()

        # Group by partition columns
        for partition_values, group_df in df.groupby(partition_by):
            if not isinstance(partition_values, tuple):
                partition_values = (partition_values,)

            # Build partition path
            partition_path = output_dir
            for col, val in zip(partition_by, partition_values):
                partition_path = partition_path / f"{col}={val}"
            partition_path.mkdir(parents=True, exist_ok=True)

            # Remove partition columns from data
            data_cols = [c for c in group_df.columns if c not in partition_by]
            partition_df = group_df[data_cols]

            # Write partition
            partition_table = pa.Table.from_pandas(partition_df, preserve_index=False)
            output_file = partition_path / "data.orc"
            orc.write_table(
                partition_table,
                str(output_file),
                compression=self.compression.upper() if self.compression != "none" else "UNCOMPRESSED",
                stripe_size=self.stripe_size,
            )

    def _generate_ddl(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Path:
        """Generate Hive DDL for creating tables."""
        ddl_lines = []
        ddl_lines.append("-- Hive DDL for loading ORC data")
        ddl_lines.append(f"-- Schema: {self.model_pack.schema_name}")
        ddl_lines.append("")

        for table_name, df in data.items():
            table_meta = self.model_pack.relationship_graph.get_table(table_name)
            partition_cols = partition_by.get(table_name) if partition_by else None

            ddl = self._generate_table_ddl(table_name, df, table_meta, partition_cols)
            ddl_lines.append(ddl)
            ddl_lines.append("")

        # Add load statements
        ddl_lines.append("-- Load data from ORC files")
        for table_name in data.keys():
            table_lower = table_name.lower()
            ddl_lines.append(f"-- LOAD DATA INPATH '/path/to/{table_lower}' INTO TABLE {table_lower};")

        ddl_content = "\n".join(ddl_lines)
        ddl_path = self.ddl_dir / "hive_orc_tables.sql"
        ddl_path.write_text(ddl_content)

        logger.info(f"Generated Hive ORC DDL at {ddl_path}")
        return ddl_path

    def _generate_table_ddl(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
        partition_by: Optional[List[str]] = None,
    ) -> str:
        """Generate DDL for a single table."""
        table_lower = table_name.lower()
        lines = [f"CREATE EXTERNAL TABLE IF NOT EXISTS {table_lower} ("]

        # Build column definitions
        columns = []
        partition_cols_set = set(partition_by) if partition_by else set()

        if table_meta:
            for col_meta in table_meta.columns:
                col_lower = col_meta.name.lower()
                if col_lower not in partition_cols_set:
                    hive_type = self.HIVE_TYPE_MAP.get(col_meta.data_type, "STRING")
                    columns.append(f"    {col_lower} {hive_type}")
        else:
            # Infer from DataFrame
            for col in df.columns:
                col_lower = col.lower()
                if col_lower not in partition_cols_set:
                    dtype = df[col].dtype
                    hive_type = self._pandas_dtype_to_hive(dtype)
                    columns.append(f"    {col_lower} {hive_type}")

        lines.append(",\n".join(columns))
        lines.append(")")

        # Add partition clause
        if partition_by:
            partition_defs = []
            for col in partition_by:
                col_lower = col.lower()
                if table_meta:
                    col_meta = table_meta.get_column(col)
                    if col_meta:
                        hive_type = self.HIVE_TYPE_MAP.get(col_meta.data_type, "STRING")
                    else:
                        hive_type = "STRING"
                else:
                    hive_type = "STRING"
                partition_defs.append(f"{col_lower} {hive_type}")
            lines.append(f"PARTITIONED BY ({', '.join(partition_defs)})")

        # Add storage format
        lines.append("STORED AS ORC")
        lines.append(f"LOCATION '/data/{self.model_pack.schema_name}/{table_lower}'")
        lines.append("TBLPROPERTIES (")
        lines.append(f"    'orc.compress'='{self.compression.upper()}',")
        lines.append("    'transactional'='false'")
        lines.append(");")

        return "\n".join(lines)

    def _pandas_dtype_to_hive(self, dtype) -> str:
        """Map pandas dtype to Hive type string."""
        dtype_str = str(dtype).lower()

        if "int64" in dtype_str:
            return "BIGINT"
        elif "int32" in dtype_str:
            return "INT"
        elif "float" in dtype_str:
            return "DOUBLE"
        elif "datetime" in dtype_str:
            return "TIMESTAMP"
        elif "bool" in dtype_str:
            return "BOOLEAN"
        else:
            return "STRING"

    def write_to_hive(
        self,
        data: Dict[str, pd.DataFrame],
        spark_session: Optional[Any] = None,
        database: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Write directly to Hive metastore using Spark.

        This method writes data directly to Hive tables using SparkSession,
        which is the preferred approach for production environments.

        Args:
            data: Dict of table_name -> DataFrame
            spark_session: Optional SparkSession (creates one if not provided)
            database: Hive database name (uses schema_name if not provided)

        Returns:
            Dict of table_name -> Hive table path
        """
        from pyspark.sql import SparkSession

        # Create or use provided SparkSession
        if spark_session is None:
            spark_session = (
                SparkSession.builder
                .appName("BankSynth_HiveWriter")
                .enableHiveSupport()
                .getOrCreate()
            )

        db_name = database or self.model_pack.schema_name or "default"
        spark_session.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        spark_session.sql(f"USE {db_name}")

        output_tables = {}

        for table_name, df in data.items():
            table_lower = table_name.lower()

            # Convert pandas to Spark DataFrame
            spark_df = spark_session.createDataFrame(df)

            # Write to Hive in ORC format
            (spark_df
             .write
             .mode("overwrite")
             .format("orc")
             .option("compression", self.compression)
             .saveAsTable(f"{db_name}.{table_lower}"))

            output_tables[table_name] = f"{db_name}.{table_lower}"
            logger.info(f"Wrote {len(df)} rows to Hive table {db_name}.{table_lower}")

        return output_tables
