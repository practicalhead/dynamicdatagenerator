"""
Hive Parquet Writer - Write generated data to Hive tables in Parquet format.

Parquet is a columnar storage format that provides:
- Efficient compression and encoding
- Schema evolution support
- Wide tool compatibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from bank_synth.models import DataType, GenerationConfig, ModelPack, TableMetadata

logger = logging.getLogger(__name__)


class HiveParquetWriter:
    """
    Writes generated data to Parquet format for Hive tables.

    Features:
    - Writes data in Parquet format using PyArrow
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
        row_group_size: int = 128 * 1024 * 1024,  # 128MB default
    ):
        """
        Initialize the Parquet writer.

        Args:
            output_dir: Base output directory
            model_pack: Model pack with table metadata
            compression: Parquet compression codec (snappy, gzip, brotli, lz4, zstd, none)
            row_group_size: Parquet row group size in bytes
        """
        self.output_dir = Path(output_dir)
        self.model_pack = model_pack
        self.compression = compression
        self.row_group_size = row_group_size

        # Create output directories
        self.parquet_dir = self.output_dir / "hive_parquet"
        self.ddl_dir = self.output_dir / "ddl"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.ddl_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Path]:
        """
        Write all generated data to Parquet files.

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

        logger.info(f"Wrote {len(output_paths)} tables to Parquet format")
        return output_paths

    def write_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        partition_by: Optional[List[str]] = None,
    ) -> Path:
        """
        Write a single table to Parquet format.

        Args:
            table_name: Name of the table
            df: DataFrame to write
            partition_by: Optional list of partition columns

        Returns:
            Path to output file/directory
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_dir = self.parquet_dir / table_name.lower()
        table_dir.mkdir(parents=True, exist_ok=True)

        # Get table metadata for schema
        table_meta = self.model_pack.relationship_graph.get_table(table_name)

        # Convert DataFrame to Arrow table with proper schema
        arrow_table = self._df_to_arrow(df, table_meta)

        if partition_by:
            # Write partitioned data
            pq.write_to_dataset(
                arrow_table,
                root_path=str(table_dir),
                partition_cols=partition_by,
                compression=self.compression,
            )
            output_path = table_dir
        else:
            # Write single Parquet file
            output_path = table_dir / f"{table_name.lower()}.parquet"
            pq.write_table(
                arrow_table,
                str(output_path),
                compression=self.compression,
                row_group_size=self.row_group_size,
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

        # Rename columns to lowercase
        df_renamed = df.copy()
        df_renamed.columns = [c.lower() for c in df_renamed.columns]

        # Build schema from metadata if available
        if table_meta:
            fields = []
            for col_meta in table_meta.columns:
                if col_meta.name.lower() in df_renamed.columns:
                    arrow_type = self._datatype_to_arrow(col_meta.data_type)
                    fields.append(pa.field(col_meta.name.lower(), arrow_type, nullable=col_meta.nullable))

            # Add any columns not in metadata
            meta_cols_lower = {c.name.lower() for c in table_meta.columns}
            for col in df_renamed.columns:
                if col not in meta_cols_lower:
                    dtype = df_renamed[col].dtype
                    arrow_type = self._pandas_dtype_to_arrow(dtype)
                    fields.append(pa.field(col, arrow_type, nullable=True))

            schema = pa.schema(fields)
            return pa.Table.from_pandas(df_renamed, schema=schema, preserve_index=False)
        else:
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

    def _generate_ddl(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Path:
        """Generate Hive DDL for creating tables."""
        ddl_lines = []
        ddl_lines.append("-- Hive DDL for loading Parquet data")
        ddl_lines.append(f"-- Schema: {self.model_pack.schema_name}")
        ddl_lines.append("")

        for table_name, df in data.items():
            table_meta = self.model_pack.relationship_graph.get_table(table_name)
            partition_cols = partition_by.get(table_name) if partition_by else None

            ddl = self._generate_table_ddl(table_name, df, table_meta, partition_cols)
            ddl_lines.append(ddl)
            ddl_lines.append("")

        ddl_content = "\n".join(ddl_lines)
        ddl_path = self.ddl_dir / "hive_parquet_tables.sql"
        ddl_path.write_text(ddl_content)

        logger.info(f"Generated Hive Parquet DDL at {ddl_path}")
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
        lines.append("STORED AS PARQUET")
        lines.append(f"LOCATION '/data/{self.model_pack.schema_name}/{table_lower}'")
        lines.append("TBLPROPERTIES (")
        lines.append(f"    'parquet.compression'='{self.compression.upper()}'")
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
