"""
Hive metadata extractor using Spark.

Extracts table metadata from Hive metastore via SparkSession.
Note: Hive has limited native FK support, so relationships are
typically provided via relationships.yaml.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    Relationship,
    TableMetadata,
)

logger = logging.getLogger(__name__)


# Hive type mapping
HIVE_TYPE_MAP = {
    "string": DataType.STRING,
    "varchar": DataType.STRING,
    "char": DataType.STRING,
    "int": DataType.INTEGER,
    "integer": DataType.INTEGER,
    "bigint": DataType.BIGINT,
    "smallint": DataType.INTEGER,
    "tinyint": DataType.INTEGER,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "decimal": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "date": DataType.DATE,
    "timestamp": DataType.TIMESTAMP,
    "binary": DataType.BINARY,
}


class HiveMetadataExtractor:
    """
    Extracts metadata from Hive metastore via Spark.

    Supports:
    - Column metadata from DESCRIBE
    - Table properties
    - Limited FK support via table properties (if available)
    """

    def __init__(self, spark_session: Optional[Any] = None, app_name: str = "bank_synth"):
        """
        Initialize extractor with Spark session.

        Args:
            spark_session: Existing SparkSession or None to create one
            app_name: Application name for new SparkSession
        """
        self._spark = spark_session
        self._app_name = app_name
        self._owns_spark = False

    def connect(self) -> None:
        """Create or configure Spark session."""
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = (
                SparkSession.builder
                .appName(self._app_name)
                .enableHiveSupport()
                .getOrCreate()
            )
            self._owns_spark = True
            logger.info("Created new SparkSession with Hive support")

    def disconnect(self) -> None:
        """Stop Spark session if we own it."""
        if self._owns_spark and self._spark:
            self._spark.stop()
            self._spark = None
            self._owns_spark = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @property
    def spark(self):
        """Get the Spark session."""
        if self._spark is None:
            self.connect()
        return self._spark

    def get_table_metadata(self, database: str, table_name: str) -> Optional[TableMetadata]:
        """
        Get metadata for a specific Hive table.

        Args:
            database: Hive database name
            table_name: Table name

        Returns:
            TableMetadata or None if table not found
        """
        try:
            # Set database context
            self.spark.sql(f"USE {database}")

            # Check table exists
            tables = [t.name.lower() for t in self.spark.catalog.listTables(database)]
            if table_name.lower() not in tables:
                logger.warning(f"Table not found: {database}.{table_name}")
                return None

            # Get columns via DESCRIBE
            columns = self._get_columns(database, table_name)

            # Try to get table comment
            comment = self._get_table_comment(database, table_name)

            # Attempt to infer row count
            row_count = self._estimate_row_count(database, table_name)

            # Get PK from table properties if available
            pk_columns = self._get_primary_key_from_properties(database, table_name)

            # Mark PK columns
            for col in columns:
                if col.name.lower() in [pk.lower() for pk in pk_columns]:
                    col.is_primary_key = True

            return TableMetadata(
                name=table_name.upper(),
                schema=database,
                columns=columns,
                primary_key=pk_columns,
                comment=comment,
                row_count_estimate=row_count,
            )

        except Exception as e:
            logger.error(f"Error getting Hive metadata for {database}.{table_name}: {e}")
            return None

    def _get_columns(self, database: str, table_name: str) -> List[ColumnMetadata]:
        """Get column metadata from DESCRIBE."""
        desc_df = self.spark.sql(f"DESCRIBE {database}.{table_name}")
        rows = desc_df.collect()

        columns = []
        in_partition = False

        for row in rows:
            col_name = row["col_name"]

            # Skip empty rows and partition info header
            if not col_name or col_name.startswith("#"):
                if "Partition" in str(col_name):
                    in_partition = True
                continue

            # Skip partition columns for now (handled separately)
            if in_partition:
                continue

            data_type = row["data_type"].lower()
            comment = row["comment"] if "comment" in row.asDict() else None

            # Parse type and get precision/scale for decimals
            precision = None
            scale = None
            max_length = None

            if "decimal" in data_type:
                # Parse decimal(p,s)
                import re
                match = re.search(r"decimal\((\d+),(\d+)\)", data_type)
                if match:
                    precision = int(match.group(1))
                    scale = int(match.group(2))
                mapped_type = DataType.DECIMAL
            elif "varchar" in data_type or "char" in data_type:
                import re
                match = re.search(r"(?:var)?char\((\d+)\)", data_type)
                if match:
                    max_length = int(match.group(1))
                mapped_type = DataType.STRING
            else:
                # Simple type lookup
                base_type = data_type.split("(")[0].strip()
                mapped_type = HIVE_TYPE_MAP.get(base_type, DataType.UNKNOWN)

            columns.append(ColumnMetadata(
                name=col_name.upper(),
                data_type=mapped_type,
                nullable=True,  # Hive columns are nullable by default
                precision=precision,
                scale=scale,
                max_length=max_length,
                comment=comment,
            ))

        return columns

    def _get_table_comment(self, database: str, table_name: str) -> Optional[str]:
        """Get table comment from extended description."""
        try:
            ext_df = self.spark.sql(f"DESCRIBE EXTENDED {database}.{table_name}")
            for row in ext_df.collect():
                if row["col_name"] and "comment" in row["col_name"].lower():
                    return row["data_type"]
        except Exception:
            pass
        return None

    def _estimate_row_count(self, database: str, table_name: str) -> Optional[int]:
        """Estimate row count from statistics."""
        try:
            # Try to get from table stats
            stats_df = self.spark.sql(f"DESCRIBE FORMATTED {database}.{table_name}")
            for row in stats_df.collect():
                if row["col_name"] and "numRows" in str(row["col_name"]):
                    return int(row["data_type"])
        except Exception:
            pass
        return None

    def _get_primary_key_from_properties(
        self,
        database: str,
        table_name: str,
    ) -> List[str]:
        """
        Try to get PK from table properties.

        Some Hive setups store PK info in TBLPROPERTIES as:
        - 'primary_key' = 'col1,col2'
        - or other conventions
        """
        try:
            props_df = self.spark.sql(f"SHOW TBLPROPERTIES {database}.{table_name}")
            for row in props_df.collect():
                if "primary_key" in str(row["key"]).lower():
                    return [c.strip().upper() for c in row["value"].split(",")]
        except Exception:
            pass
        return []

    def get_all_tables(self, database: str) -> List[str]:
        """Get all table names in a database."""
        try:
            tables = self.spark.catalog.listTables(database)
            return [t.name.upper() for t in tables]
        except Exception as e:
            logger.error(f"Error listing tables in {database}: {e}")
            return []

    def sample_table(
        self,
        database: str,
        table_name: str,
        strategy: str = "percent:1",
        max_rows: int = 100000,
    ) -> Any:
        """
        Sample data from a Hive table.

        Args:
            database: Hive database name
            table_name: Table name
            strategy: Sampling strategy (percent:N, rows:N, or full)
            max_rows: Maximum rows to return

        Returns:
            pandas DataFrame with sampled data
        """
        # Parse strategy
        if strategy.startswith("percent:"):
            fraction = float(strategy.split(":")[1]) / 100.0
            df = (
                self.spark
                .table(f"{database}.{table_name}")
                .sample(fraction=fraction)
                .limit(max_rows)
            )
        elif strategy.startswith("rows:"):
            n_rows = int(strategy.split(":")[1])
            df = (
                self.spark
                .table(f"{database}.{table_name}")
                .limit(min(n_rows, max_rows))
            )
        else:
            df = (
                self.spark
                .table(f"{database}.{table_name}")
                .limit(max_rows)
            )

        logger.info(f"Sampling {database}.{table_name} with strategy: {strategy}")
        return df.toPandas()

    def read_parquet(self, path: str) -> Any:
        """Read Parquet file(s) as DataFrame."""
        return self.spark.read.parquet(path).toPandas()

    def write_parquet(self, df: Any, path: str, mode: str = "overwrite") -> None:
        """Write DataFrame to Parquet."""
        spark_df = self.spark.createDataFrame(df)
        spark_df.write.mode(mode).parquet(path)
