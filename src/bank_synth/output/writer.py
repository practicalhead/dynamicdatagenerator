"""
Output Writer - Unified interface for writing generated data to various formats.

Supports:
- Hive ORC (recommended for Hive)
- Hive Parquet
- Oracle-compatible CSV
- Direct Hive table writes via Spark
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from bank_synth.models import GenerationConfig, ModelPack

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Unified output writer supporting multiple formats.

    Output Structure:
        output/<run_id>/
        ├── hive_orc/           # ORC format files
        │   ├── customers/
        │   │   └── customers.orc
        │   └── ...
        ├── hive_parquet/       # Parquet format files
        │   └── ...
        ├── oracle/             # CSV files for Oracle
        │   └── ...
        ├── ddl/                # DDL scripts
        │   ├── hive_orc_tables.sql
        │   ├── hive_parquet_tables.sql
        │   └── oracle_load.sql
        ├── report/             # Quality reports
        │   ├── quality.json
        │   └── quality.md
        └── manifest.json       # Generation manifest
    """

    def __init__(
        self,
        config: GenerationConfig,
        model_pack: ModelPack,
    ):
        """
        Initialize the output writer.

        Args:
            config: Generation configuration
            model_pack: Model pack with metadata
        """
        self.config = config
        self.model_pack = model_pack

        # Set up output directory
        self.output_dir = config.output_dir / config.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self) -> Path:
        """Return the output directory path."""
        return self.output_dir

    def write(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """
        Write generated data to all configured output formats.

        Args:
            data: Dict of table_name -> DataFrame
            partition_by: Optional dict of table_name -> partition columns

        Returns:
            Dict of format -> (table_name -> output_path)
        """
        output_paths: Dict[str, Dict[str, Path]] = {}
        formats = [f.lower() for f in self.config.output_formats]

        # Write to each format
        for fmt in formats:
            if fmt == "hive" or fmt == "hive_orc" or fmt == "orc":
                output_paths["hive_orc"] = self._write_hive_orc(data, partition_by)
            elif fmt == "hive_parquet" or fmt == "parquet":
                output_paths["hive_parquet"] = self._write_hive_parquet(data, partition_by)
            elif fmt == "oracle" or fmt == "csv":
                output_paths["oracle"] = self._write_oracle_csv(data)
            else:
                logger.warning(f"Unknown output format: {fmt}")

        # Write manifest
        self._write_manifest(data, output_paths)

        return output_paths

    def _write_hive_orc(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Path]:
        """Write to Hive ORC format."""
        from bank_synth.output.hive_orc import HiveOrcWriter

        writer = HiveOrcWriter(self.output_dir, self.model_pack)
        return writer.write(data, partition_by)

    def _write_hive_parquet(
        self,
        data: Dict[str, pd.DataFrame],
        partition_by: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Path]:
        """Write to Hive Parquet format."""
        from bank_synth.output.hive_parquet import HiveParquetWriter

        writer = HiveParquetWriter(self.output_dir, self.model_pack)
        return writer.write(data, partition_by)

    def _write_oracle_csv(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Path]:
        """Write to Oracle-compatible CSV format."""
        oracle_dir = self.output_dir / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        for table_name, df in data.items():
            output_path = oracle_dir / f"{table_name.lower()}.csv"

            # Write CSV with Oracle-friendly settings
            df.to_csv(
                output_path,
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
                na_rep="",
            )

            output_paths[table_name] = output_path
            logger.info(f"Wrote {len(df)} rows to {output_path}")

        # Generate SQL*Loader control files
        self._generate_oracle_ctl(data, oracle_dir)

        return output_paths

    def _generate_oracle_ctl(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> None:
        """Generate SQL*Loader control files for Oracle."""
        ddl_dir = self.output_dir / "ddl"
        ddl_dir.mkdir(parents=True, exist_ok=True)

        for table_name, df in data.items():
            table_lower = table_name.lower()
            ctl_lines = []

            ctl_lines.append(f"LOAD DATA")
            ctl_lines.append(f"INFILE '{table_lower}.csv'")
            ctl_lines.append("INTO TABLE " + table_name)
            ctl_lines.append("FIELDS TERMINATED BY ','")
            ctl_lines.append("OPTIONALLY ENCLOSED BY '\"'")
            ctl_lines.append("TRAILING NULLCOLS")
            ctl_lines.append("(")

            # Get column definitions
            table_meta = self.model_pack.relationship_graph.get_table(table_name)

            col_defs = []
            for col in df.columns:
                col_upper = col.upper()
                col_def = f"    {col_upper}"

                # Add type hints for dates
                if table_meta:
                    col_meta = table_meta.get_column(col)
                    if col_meta:
                        if col_meta.data_type.value == "timestamp":
                            col_def += ' TIMESTAMP "YYYY-MM-DD HH24:MI:SS"'
                        elif col_meta.data_type.value == "date":
                            col_def += ' DATE "YYYY-MM-DD"'

                col_defs.append(col_def)

            ctl_lines.append(",\n".join(col_defs))
            ctl_lines.append(")")

            ctl_path = ddl_dir / f"{table_lower}.ctl"
            ctl_path.write_text("\n".join(ctl_lines))

        logger.info(f"Generated SQL*Loader control files in {ddl_dir}")

    def _write_manifest(
        self,
        data: Dict[str, pd.DataFrame],
        output_paths: Dict[str, Dict[str, Path]],
    ) -> Path:
        """Write generation manifest."""
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "run_id": self.config.run_id,
            "model_version": self.model_pack.version,
            "schema": self.model_pack.schema_name,
            "target_table": self.config.target_table,
            "target_rows": self.config.target_rows,
            "seed": self.config.seed,
            "tables": {},
            "output_formats": list(output_paths.keys()),
        }

        for table_name, df in data.items():
            manifest["tables"][table_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_list": list(df.columns),
            }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Wrote manifest to {manifest_path}")
        return manifest_path

    def write_to_hive(
        self,
        data: Dict[str, pd.DataFrame],
        spark_session: Optional[Any] = None,
        database: Optional[str] = None,
        format: str = "orc",
    ) -> Dict[str, str]:
        """
        Write directly to Hive metastore using Spark.

        This is the preferred method for production Hive deployments.

        Args:
            data: Dict of table_name -> DataFrame
            spark_session: Optional SparkSession (creates one if not provided)
            database: Hive database name
            format: Storage format ("orc" or "parquet")

        Returns:
            Dict of table_name -> Hive table path
        """
        if format.lower() == "orc":
            from bank_synth.output.hive_orc import HiveOrcWriter
            writer = HiveOrcWriter(self.output_dir, self.model_pack)
        else:
            from bank_synth.output.hive_parquet import HiveParquetWriter
            writer = HiveParquetWriter(self.output_dir, self.model_pack)

        # For direct Hive write, we need special handling
        return self._write_to_hive_spark(data, spark_session, database, format)

    def _write_to_hive_spark(
        self,
        data: Dict[str, pd.DataFrame],
        spark_session: Optional[Any],
        database: Optional[str],
        format: str,
    ) -> Dict[str, str]:
        """Write to Hive using Spark."""
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

        output_tables = {}

        for table_name, df in data.items():
            table_lower = table_name.lower()
            full_table_name = f"{db_name}.{table_lower}"

            # Convert pandas to Spark DataFrame
            spark_df = spark_session.createDataFrame(df)

            # Write to Hive
            (spark_df
             .write
             .mode("overwrite")
             .format(format.lower())
             .saveAsTable(full_table_name))

            output_tables[table_name] = full_table_name
            logger.info(f"Wrote {len(df)} rows to Hive table {full_table_name}")

        return output_tables
