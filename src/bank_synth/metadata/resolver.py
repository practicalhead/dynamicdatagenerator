"""
Unified metadata resolver that combines Oracle and Hive metadata with
user-provided relationship overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    PrivacyLevel,
    Relationship,
    RelationshipGraph,
    TableMetadata,
)

logger = logging.getLogger(__name__)


class MetadataResolver:
    """
    Resolves metadata from multiple sources and builds a unified relationship graph.

    Sources (in order of precedence for relationships):
    1. User-provided relationships.yaml (authoritative)
    2. Oracle catalog (PK/FK constraints)
    3. Hive metastore (limited FK support)
    """

    def __init__(
        self,
        oracle_extractor: Optional[Any] = None,
        hive_extractor: Optional[Any] = None,
        relationships_file: Optional[Path] = None,
        privacy_policy_file: Optional[Path] = None,
    ):
        self.oracle_extractor = oracle_extractor
        self.hive_extractor = hive_extractor
        self.relationships_file = relationships_file
        self.privacy_policy_file = privacy_policy_file
        self.graph = RelationshipGraph()

        # User-provided overrides
        self._user_relationships: List[Dict[str, Any]] = []
        self._privacy_policy: Dict[str, Dict[str, str]] = {}

        # Load user files if provided
        if relationships_file:
            self._load_relationships_file(relationships_file)
        if privacy_policy_file:
            self._load_privacy_policy(privacy_policy_file)

    def _load_relationships_file(self, path: Path) -> None:
        """Load relationships from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Relationships file not found: {path}")
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        self._user_relationships = data.get("relationships", [])
        logger.info(f"Loaded {len(self._user_relationships)} relationships from {path}")

    def _load_privacy_policy(self, path: Path) -> None:
        """Load privacy policy from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Privacy policy file not found: {path}")
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        self._privacy_policy = data.get("tables", {})
        logger.info(f"Loaded privacy policy for {len(self._privacy_policy)} tables from {path}")

    def get_privacy_policy(self) -> Dict[str, Dict[str, str]]:
        """Return the loaded privacy policy."""
        return self._privacy_policy

    def resolve(
        self,
        schema: str,
        tables: List[str],
        source: str = "oracle",
    ) -> RelationshipGraph:
        """
        Resolve metadata for specified tables and build relationship graph.

        Args:
            schema: Schema/database name
            tables: List of table names to process
            source: Primary source ("oracle", "hive", or "both")

        Returns:
            RelationshipGraph with all tables and relationships
        """
        logger.info(f"Resolving metadata for {len(tables)} tables from {source}")

        # Extract metadata from database sources
        if source in ("oracle", "both") and self.oracle_extractor:
            self._extract_oracle_metadata(schema, tables)
        if source in ("hive", "both") and self.hive_extractor:
            self._extract_hive_metadata(schema, tables)

        # Apply user-provided relationships (override database-derived)
        self._apply_user_relationships()

        # Apply privacy policy
        self._apply_privacy_policy()

        logger.info(
            f"Resolved {len(self.graph.tables)} tables with "
            f"{len(self.graph.relationships)} relationships"
        )

        return self.graph

    def resolve_from_samples(
        self,
        schema: str,
        sample_dir: Path,
        tables_config: Optional[Dict[str, Any]] = None,
    ) -> RelationshipGraph:
        """
        Resolve metadata from sample data files (offline mode).

        Args:
            schema: Schema name to use
            sample_dir: Directory containing sample Parquet/CSV files
            tables_config: Optional tables configuration with column hints

        Returns:
            RelationshipGraph with inferred metadata
        """
        import pandas as pd
        import pyarrow.parquet as pq

        sample_dir = Path(sample_dir)
        logger.info(f"Resolving metadata from samples in {sample_dir}")

        # Find all sample files
        parquet_files = list(sample_dir.glob("**/*.parquet"))
        csv_files = list(sample_dir.glob("**/*.csv"))

        all_files = parquet_files + csv_files
        logger.info(f"Found {len(all_files)} sample files")

        for file_path in all_files:
            table_name = file_path.stem.upper()

            # Read schema from file
            if file_path.suffix == ".parquet":
                pq_file = pq.ParquetFile(file_path)
                arrow_schema = pq_file.schema_arrow
                df = pq_file.read().to_pandas().head(100)
                columns = self._infer_columns_from_arrow(arrow_schema, df)
            else:
                df = pd.read_csv(file_path, nrows=100)
                columns = self._infer_columns_from_pandas(df)

            # Apply column config if provided
            if tables_config and table_name.lower() in tables_config:
                table_conf = tables_config[table_name.lower()]
                columns = self._apply_column_config(columns, table_conf)

            # Create table metadata
            table = TableMetadata(
                name=table_name,
                schema=schema,
                columns=columns,
                primary_key=self._infer_primary_key(columns),
            )
            self.graph.add_table(table)

        # Apply user relationships
        self._apply_user_relationships()

        # Apply privacy policy
        self._apply_privacy_policy()

        return self.graph

    def _extract_oracle_metadata(self, schema: str, tables: List[str]) -> None:
        """Extract metadata from Oracle catalog."""
        for table_name in tables:
            try:
                table_meta = self.oracle_extractor.get_table_metadata(schema, table_name)
                if table_meta:
                    self.graph.add_table(table_meta)

                # Get FK relationships
                fk_rels = self.oracle_extractor.get_foreign_keys(schema, table_name)
                for rel in fk_rels:
                    self.graph.add_relationship(rel)

            except Exception as e:
                logger.error(f"Error extracting Oracle metadata for {table_name}: {e}")

    def _extract_hive_metadata(self, schema: str, tables: List[str]) -> None:
        """Extract metadata from Hive metastore."""
        for table_name in tables:
            try:
                # Only add if not already from Oracle
                if not self.graph.get_table(table_name):
                    table_meta = self.hive_extractor.get_table_metadata(schema, table_name)
                    if table_meta:
                        self.graph.add_table(table_meta)
            except Exception as e:
                logger.error(f"Error extracting Hive metadata for {table_name}: {e}")

    def _apply_user_relationships(self) -> None:
        """Apply user-provided relationship definitions."""
        for rel_def in self._user_relationships:
            rel = Relationship(
                name=rel_def.get("name", f"fk_{rel_def['child_table']}_{rel_def['parent_table']}"),
                parent_table=rel_def["parent_table"],
                parent_columns=rel_def.get("parent_columns", [rel_def.get("parent_column")]),
                child_table=rel_def["child_table"],
                child_columns=rel_def.get("child_columns", [rel_def.get("child_column")]),
                cardinality=rel_def.get("cardinality", "1:N"),
                is_optional=rel_def.get("is_optional", True),
            )

            # Remove any existing relationship with same name or same tables+columns
            self.graph.relationships = [
                r for r in self.graph.relationships
                if not (
                    r.name == rel.name or
                    (r.parent_table == rel.parent_table and
                     r.child_table == rel.child_table and
                     r.child_columns == rel.child_columns)
                )
            ]

            self.graph.add_relationship(rel)

            # Update column metadata
            child_table = self.graph.get_table(rel.child_table)
            if child_table:
                for col_name in rel.child_columns:
                    col = child_table.get_column(col_name)
                    if col:
                        col.is_foreign_key = True
                        parent_col = rel.parent_columns[rel.child_columns.index(col_name)]
                        col.fk_reference = (rel.parent_table, parent_col)

    def _apply_privacy_policy(self) -> None:
        """Apply privacy policy to columns."""
        for table_name, column_policies in self._privacy_policy.items():
            table = self.graph.get_table(table_name)
            if not table:
                continue

            for col_name, policy in column_policies.items():
                col = table.get_column(col_name)
                if not col:
                    continue

                # Parse policy
                if isinstance(policy, str):
                    col.privacy_level = PrivacyLevel(policy.lower())
                elif isinstance(policy, dict):
                    col.privacy_level = PrivacyLevel(policy.get("level", "public").lower())
                    if "format_pattern" in policy:
                        col.format_pattern = policy["format_pattern"]
                    if "faker_provider" in policy:
                        col.faker_provider = policy["faker_provider"]

    def _infer_columns_from_arrow(
        self,
        arrow_schema,
        df,
    ) -> List[ColumnMetadata]:
        """Infer column metadata from Arrow schema."""
        import pyarrow as pa

        columns = []
        for field in arrow_schema:
            col_name = field.name
            data_type = self._map_arrow_type(field.type)
            nullable = field.nullable

            # Check if might be PK (unique, not null, first column or named *_id)
            is_pk = (
                not nullable and
                (col_name.lower().endswith("_id") or arrow_schema.get_field_index(col_name) == 0)
            )

            columns.append(ColumnMetadata(
                name=col_name,
                data_type=data_type,
                nullable=nullable,
                is_primary_key=is_pk,
            ))

        return columns

    def _infer_columns_from_pandas(self, df) -> List[ColumnMetadata]:
        """Infer column metadata from pandas DataFrame."""
        columns = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            data_type = self._map_pandas_type(dtype)
            nullable = df[col_name].isna().any()

            is_pk = (
                not nullable and
                (col_name.lower().endswith("_id") or df.columns.get_loc(col_name) == 0)
            )

            columns.append(ColumnMetadata(
                name=col_name,
                data_type=data_type,
                nullable=nullable,
                is_primary_key=is_pk,
            ))

        return columns

    def _map_arrow_type(self, arrow_type) -> DataType:
        """Map Arrow type to normalized DataType."""
        import pyarrow as pa

        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return DataType.STRING
        elif pa.types.is_int32(arrow_type) or pa.types.is_int16(arrow_type):
            return DataType.INTEGER
        elif pa.types.is_int64(arrow_type):
            return DataType.BIGINT
        elif pa.types.is_decimal(arrow_type):
            return DataType.DECIMAL
        elif pa.types.is_float32(arrow_type):
            return DataType.FLOAT
        elif pa.types.is_float64(arrow_type):
            return DataType.DOUBLE
        elif pa.types.is_date(arrow_type):
            return DataType.DATE
        elif pa.types.is_timestamp(arrow_type):
            return DataType.TIMESTAMP
        elif pa.types.is_boolean(arrow_type):
            return DataType.BOOLEAN
        elif pa.types.is_binary(arrow_type):
            return DataType.BINARY
        else:
            return DataType.UNKNOWN

    def _map_pandas_type(self, dtype) -> DataType:
        """Map pandas dtype to normalized DataType."""
        dtype_str = str(dtype).lower()

        if "int64" in dtype_str or "int32" in dtype_str:
            return DataType.BIGINT
        elif "float" in dtype_str:
            return DataType.DOUBLE
        elif "datetime" in dtype_str:
            return DataType.TIMESTAMP
        elif "bool" in dtype_str:
            return DataType.BOOLEAN
        elif "object" in dtype_str or "string" in dtype_str:
            return DataType.STRING
        else:
            return DataType.UNKNOWN

    def _apply_column_config(
        self,
        columns: List[ColumnMetadata],
        table_config: Dict[str, Any],
    ) -> List[ColumnMetadata]:
        """Apply column configuration overrides."""
        pk_columns = table_config.get("primary_key", [])
        col_configs = table_config.get("columns", {})

        for col in columns:
            # Set PK
            if col.name.lower() in [pk.lower() for pk in pk_columns]:
                col.is_primary_key = True

            # Apply column-specific config
            if col.name.lower() in col_configs:
                conf = col_configs[col.name.lower()]
                if "data_type" in conf:
                    col.data_type = DataType(conf["data_type"])
                if "nullable" in conf:
                    col.nullable = conf["nullable"]
                if "format_pattern" in conf:
                    col.format_pattern = conf["format_pattern"]
                if "faker_provider" in conf:
                    col.faker_provider = conf["faker_provider"]
                if "allowed_values" in conf:
                    col.allowed_values = conf["allowed_values"]

        return columns

    def _infer_primary_key(self, columns: List[ColumnMetadata]) -> List[str]:
        """Infer primary key columns."""
        return [c.name for c in columns if c.is_primary_key]
