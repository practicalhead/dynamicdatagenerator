"""
Auto-Discovery Pipeline - Zero-intervention relationship discovery.

This module provides a complete pipeline for automatically discovering
table relationships without manual configuration, using:
- Enterprise reporting queries
- ETL queries
- Dynamic test data queries
- PK/FK metadata from database catalogs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    PrivacyLevel,
    RelationshipGraph,
    TableMetadata,
)
from bank_synth.discovery.query_parser import QueryParser, ParsedQuery
from bank_synth.discovery.relationship_inferrer import (
    RelationshipInferrer,
    PKFKMetadata,
    infer_relationships,
)

logger = logging.getLogger(__name__)


class AutoDiscoveryPipeline:
    """
    Automated pipeline for discovering table relationships.

    This pipeline eliminates manual relationship configuration by:
    1. Extracting metadata from database catalogs
    2. Parsing SQL queries to discover relationships
    3. Using PK/FK constraints as authoritative sources
    4. Inferring relationships from column naming patterns

    Zero-Intervention Usage:
        pipeline = AutoDiscoveryPipeline()
        graph = pipeline.discover(
            queries_dir="./queries",  # ETL and reporting queries
            database_connection=conn,  # For PK/FK metadata
        )
    """

    def __init__(
        self,
        oracle_extractor: Optional[Any] = None,
        hive_extractor: Optional[Any] = None,
        privacy_policy_file: Optional[Path] = None,
    ):
        """
        Initialize the auto-discovery pipeline.

        Args:
            oracle_extractor: Optional Oracle metadata extractor
            hive_extractor: Optional Hive metadata extractor
            privacy_policy_file: Optional path to privacy policy YAML
        """
        self.oracle_extractor = oracle_extractor
        self.hive_extractor = hive_extractor
        self.privacy_policy_file = privacy_policy_file
        self._privacy_policy: Dict[str, Dict[str, Any]] = {}

        if privacy_policy_file:
            self._load_privacy_policy(privacy_policy_file)

    def discover(
        self,
        schema: str,
        queries: Optional[Union[List[Tuple[str, str]], Path, str]] = None,
        tables: Optional[List[str]] = None,
        sample_dir: Optional[Path] = None,
        source: str = "auto",
        min_confidence: float = 0.3,
    ) -> RelationshipGraph:
        """
        Automatically discover table relationships.

        This is the main entry point for zero-intervention discovery.

        Args:
            schema: Database schema name
            queries: SQL queries as list of (name, sql) tuples, or path to query files
            tables: Optional list of tables (auto-discovered from queries if not provided)
            sample_dir: Optional directory with sample data files
            source: Data source ("oracle", "hive", "samples", "auto")
            min_confidence: Minimum confidence for relationship inclusion

        Returns:
            RelationshipGraph with auto-discovered relationships
        """
        logger.info(f"Starting auto-discovery for schema: {schema}")

        # Step 1: Parse queries to discover tables
        parsed_queries = self._load_queries(queries)
        query_list = [(q.query_name, "") for q in parsed_queries]  # For inferrer

        # Extract tables from queries if not provided
        if not tables:
            tables = self._extract_tables_from_queries(parsed_queries)
            logger.info(f"Auto-discovered {len(tables)} tables from queries")

        if not tables:
            logger.warning("No tables discovered. Please provide queries or table list.")
            return RelationshipGraph()

        # Step 2: Get table metadata
        table_metadata = self._get_table_metadata(
            schema=schema,
            tables=tables,
            source=source,
            sample_dir=sample_dir,
        )

        # Step 3: Get PK/FK metadata from database
        pkfk_metadata = self._get_pkfk_metadata(schema, tables, source)

        # Step 4: Rebuild query list with actual SQL for inference
        query_sql_list = [(pq.query_name, "") for pq in parsed_queries]  # Parser already extracted joins

        # Step 5: Infer relationships
        inferrer = RelationshipInferrer(table_metadata, pkfk_metadata)

        # First process catalog FK metadata
        inferrer.infer_from_pkfk_metadata()

        # Add relationships from parsed queries
        for pq in parsed_queries:
            for join in pq.joins:
                inferrer._add_from_join(join, confidence_boost=0.2)

        # Fill gaps with pattern matching
        inferrer.infer_from_column_patterns()

        # Build the graph
        graph = inferrer.build_relationship_graph(min_confidence=min_confidence)

        # Step 6: Apply privacy policy
        self._apply_privacy_policy(graph)

        logger.info(
            f"Auto-discovery complete: {len(graph.tables)} tables, "
            f"{len(graph.relationships)} relationships"
        )

        return graph

    def discover_from_files(
        self,
        schema: str,
        queries_dir: Optional[Path] = None,
        sample_dir: Optional[Path] = None,
        min_confidence: float = 0.3,
    ) -> RelationshipGraph:
        """
        Discover relationships from file-based inputs.

        Simplified interface for file-based discovery.

        Args:
            schema: Schema name
            queries_dir: Directory containing SQL query files
            sample_dir: Directory containing sample data files
            min_confidence: Minimum confidence threshold

        Returns:
            RelationshipGraph with discovered relationships
        """
        queries = None
        if queries_dir:
            queries = Path(queries_dir)

        return self.discover(
            schema=schema,
            queries=queries,
            sample_dir=Path(sample_dir) if sample_dir else None,
            source="samples" if sample_dir else "auto",
            min_confidence=min_confidence,
        )

    def _load_queries(
        self,
        queries: Optional[Union[List[Tuple[str, str]], Path, str]],
    ) -> List[ParsedQuery]:
        """Load and parse queries from various sources."""
        if queries is None:
            return []

        parser = QueryParser()

        if isinstance(queries, list):
            # List of (name, sql) tuples
            return parser.parse_multiple(queries)

        # Path to file or directory
        queries_path = Path(queries)

        if queries_path.is_file():
            return parser.parse_file(str(queries_path))

        if queries_path.is_dir():
            all_parsed = []
            for sql_file in queries_path.glob("**/*.sql"):
                all_parsed.extend(parser.parse_file(str(sql_file)))
            return all_parsed

        logger.warning(f"Query source not found: {queries}")
        return []

    def _extract_tables_from_queries(
        self,
        parsed_queries: List[ParsedQuery],
    ) -> List[str]:
        """Extract unique table names from parsed queries."""
        tables = set()
        for pq in parsed_queries:
            tables.update(pq.tables)
        return sorted(tables)

    def _get_table_metadata(
        self,
        schema: str,
        tables: List[str],
        source: str,
        sample_dir: Optional[Path],
    ) -> Dict[str, TableMetadata]:
        """Get metadata for all tables."""
        metadata: Dict[str, TableMetadata] = {}

        # Try Oracle first
        if source in ("oracle", "auto", "both") and self.oracle_extractor:
            for table_name in tables:
                try:
                    table_meta = self.oracle_extractor.get_table_metadata(schema, table_name)
                    if table_meta:
                        metadata[table_name.upper()] = table_meta
                except Exception as e:
                    logger.debug(f"Could not get Oracle metadata for {table_name}: {e}")

        # Try Hive
        if source in ("hive", "auto", "both") and self.hive_extractor:
            for table_name in tables:
                if table_name.upper() not in metadata:
                    try:
                        table_meta = self.hive_extractor.get_table_metadata(schema, table_name)
                        if table_meta:
                            metadata[table_name.upper()] = table_meta
                    except Exception as e:
                        logger.debug(f"Could not get Hive metadata for {table_name}: {e}")

        # Try sample files
        if source in ("samples", "auto") and sample_dir:
            metadata.update(self._infer_from_samples(schema, tables, sample_dir))

        # Create placeholder metadata for any remaining tables
        for table_name in tables:
            if table_name.upper() not in metadata:
                logger.warning(f"No metadata found for {table_name}, creating placeholder")
                metadata[table_name.upper()] = TableMetadata(
                    name=table_name.upper(),
                    schema=schema,
                    columns=[],
                )

        return metadata

    def _get_pkfk_metadata(
        self,
        schema: str,
        tables: List[str],
        source: str,
    ) -> List[PKFKMetadata]:
        """Get PK/FK metadata from database catalogs."""
        pkfk_list = []

        if source in ("oracle", "auto", "both") and self.oracle_extractor:
            for table_name in tables:
                try:
                    pkfk = self._extract_oracle_pkfk(schema, table_name)
                    if pkfk:
                        pkfk_list.append(pkfk)
                except Exception as e:
                    logger.debug(f"Could not get Oracle PK/FK for {table_name}: {e}")

        if source in ("hive", "auto", "both") and self.hive_extractor:
            for table_name in tables:
                try:
                    pkfk = self._extract_hive_pkfk(schema, table_name)
                    if pkfk and not any(p.table_name.upper() == table_name.upper() for p in pkfk_list):
                        pkfk_list.append(pkfk)
                except Exception as e:
                    logger.debug(f"Could not get Hive PK/FK for {table_name}: {e}")

        return pkfk_list

    def _extract_oracle_pkfk(self, schema: str, table_name: str) -> Optional[PKFKMetadata]:
        """Extract PK/FK metadata from Oracle catalog."""
        if not self.oracle_extractor:
            return None

        try:
            # Get primary keys
            pks = self.oracle_extractor.get_primary_keys(schema, table_name)

            # Get foreign keys
            fks = self.oracle_extractor.get_foreign_keys(schema, table_name)
            fk_list = []
            for fk in fks:
                fk_list.append({
                    "column": fk.child_columns[0] if fk.child_columns else "",
                    "ref_table": fk.parent_table,
                    "ref_column": fk.parent_columns[0] if fk.parent_columns else "",
                    "nullable": fk.is_optional,
                })

            return PKFKMetadata(
                table_name=table_name,
                primary_keys=pks if pks else [],
                foreign_keys=fk_list,
            )
        except Exception as e:
            logger.debug(f"Error extracting Oracle PK/FK: {e}")
            return None

    def _extract_hive_pkfk(self, schema: str, table_name: str) -> Optional[PKFKMetadata]:
        """Extract PK/FK metadata from Hive (limited support)."""
        if not self.hive_extractor:
            return None

        try:
            # Hive has limited PK/FK support
            table_meta = self.hive_extractor.get_table_metadata(schema, table_name)
            if table_meta:
                pks = [c.name for c in table_meta.columns if c.is_primary_key]
                return PKFKMetadata(
                    table_name=table_name,
                    primary_keys=pks,
                    foreign_keys=[],  # Hive typically doesn't have FK constraints
                )
        except Exception as e:
            logger.debug(f"Error extracting Hive PK/FK: {e}")
            return None

        return None

    def _infer_from_samples(
        self,
        schema: str,
        tables: List[str],
        sample_dir: Path,
    ) -> Dict[str, TableMetadata]:
        """Infer table metadata from sample files."""
        import pandas as pd

        metadata = {}
        sample_dir = Path(sample_dir)

        for table_name in tables:
            table_upper = table_name.upper()
            table_lower = table_name.lower()

            # Try to find matching sample file
            patterns = [
                f"{table_lower}.parquet",
                f"{table_upper}.parquet",
                f"{table_lower}.csv",
                f"{table_upper}.csv",
                f"**/{table_lower}.parquet",
                f"**/{table_lower}.csv",
            ]

            for pattern in patterns:
                matches = list(sample_dir.glob(pattern))
                if matches:
                    file_path = matches[0]

                    try:
                        if file_path.suffix == ".parquet":
                            import pyarrow.parquet as pq
                            pq_file = pq.ParquetFile(file_path)
                            arrow_schema = pq_file.schema_arrow
                            columns = self._infer_columns_from_arrow(arrow_schema)
                        else:
                            df = pd.read_csv(file_path, nrows=100)
                            columns = self._infer_columns_from_pandas(df)

                        metadata[table_upper] = TableMetadata(
                            name=table_upper,
                            schema=schema,
                            columns=columns,
                            primary_key=self._infer_primary_key(columns),
                        )
                        break

                    except Exception as e:
                        logger.warning(f"Error reading sample {file_path}: {e}")

        return metadata

    def _infer_columns_from_arrow(self, arrow_schema) -> List[ColumnMetadata]:
        """Infer column metadata from Arrow schema."""
        import pyarrow as pa

        columns = []
        for i, field in enumerate(arrow_schema):
            col_name = field.name
            data_type = self._map_arrow_type(field.type)
            nullable = field.nullable

            # Heuristic for PK
            is_pk = (
                not nullable and
                (col_name.lower().endswith("_id") and i == 0) or
                (col_name.lower() == "id")
            )

            columns.append(ColumnMetadata(
                name=col_name.upper(),
                data_type=data_type,
                nullable=nullable,
                is_primary_key=is_pk,
            ))

        return columns

    def _infer_columns_from_pandas(self, df) -> List[ColumnMetadata]:
        """Infer column metadata from pandas DataFrame."""
        columns = []
        for i, col_name in enumerate(df.columns):
            dtype = df[col_name].dtype
            data_type = self._map_pandas_type(dtype)
            nullable = df[col_name].isna().any()

            is_pk = (
                not nullable and
                (col_name.lower().endswith("_id") and i == 0) or
                (col_name.lower() == "id")
            )

            columns.append(ColumnMetadata(
                name=col_name.upper(),
                data_type=data_type,
                nullable=nullable,
                is_primary_key=is_pk,
            ))

        return columns

    def _map_arrow_type(self, arrow_type) -> DataType:
        """Map Arrow type to DataType."""
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
        """Map pandas dtype to DataType."""
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

    def _infer_primary_key(self, columns: List[ColumnMetadata]) -> List[str]:
        """Infer primary key columns."""
        return [c.name for c in columns if c.is_primary_key]

    def _load_privacy_policy(self, path: Path) -> None:
        """Load privacy policy from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Privacy policy file not found: {path}")
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        self._privacy_policy = data.get("tables", {})
        logger.info(f"Loaded privacy policy for {len(self._privacy_policy)} tables")

    def _apply_privacy_policy(self, graph: RelationshipGraph) -> None:
        """Apply privacy policy to columns in the graph."""
        for table_name, column_policies in self._privacy_policy.items():
            table = graph.get_table(table_name)
            if not table:
                continue

            for col_name, policy in column_policies.items():
                col = table.get_column(col_name)
                if not col:
                    continue

                if isinstance(policy, str):
                    col.privacy_level = PrivacyLevel(policy.lower())
                elif isinstance(policy, dict):
                    col.privacy_level = PrivacyLevel(policy.get("level", "public").lower())
                    if "format_pattern" in policy:
                        col.format_pattern = policy["format_pattern"]
                    if "faker_provider" in policy:
                        col.faker_provider = policy["faker_provider"]

    def get_privacy_policy(self) -> Dict[str, Dict[str, Any]]:
        """Return the loaded privacy policy."""
        return self._privacy_policy


def auto_discover(
    schema: str,
    queries: Optional[Union[List[Tuple[str, str]], Path, str]] = None,
    sample_dir: Optional[Path] = None,
    oracle_conn: Optional[str] = None,
    hive_spark: bool = False,
    privacy_policy: Optional[Path] = None,
    min_confidence: float = 0.3,
) -> RelationshipGraph:
    """
    Convenience function for zero-intervention discovery.

    This is the simplest interface for automatic relationship discovery.

    Args:
        schema: Database schema name
        queries: SQL queries (list, file path, or directory)
        sample_dir: Optional directory with sample data
        oracle_conn: Optional Oracle connection string
        hive_spark: Whether to use Spark for Hive access
        privacy_policy: Optional path to privacy policy YAML
        min_confidence: Minimum confidence for relationships

    Returns:
        RelationshipGraph with auto-discovered relationships

    Example:
        # Minimal usage - just provide queries
        graph = auto_discover(
            schema="CORE",
            queries="./etl_queries/",
        )

        # With sample data
        graph = auto_discover(
            schema="CORE",
            queries="./etl_queries/",
            sample_dir="./samples/",
        )

        # Full database access
        graph = auto_discover(
            schema="CORE",
            queries="./etl_queries/",
            oracle_conn="user/pwd@host:1521/SID",
        )
    """
    oracle_extractor = None
    hive_extractor = None

    if oracle_conn:
        try:
            from bank_synth.metadata import OracleMetadataExtractor
            oracle_extractor = OracleMetadataExtractor(oracle_conn)
        except Exception as e:
            logger.warning(f"Could not initialize Oracle extractor: {e}")

    if hive_spark:
        try:
            from bank_synth.metadata import HiveMetadataExtractor
            hive_extractor = HiveMetadataExtractor()
        except Exception as e:
            logger.warning(f"Could not initialize Hive extractor: {e}")

    pipeline = AutoDiscoveryPipeline(
        oracle_extractor=oracle_extractor,
        hive_extractor=hive_extractor,
        privacy_policy_file=Path(privacy_policy) if privacy_policy else None,
    )

    return pipeline.discover(
        schema=schema,
        queries=queries,
        sample_dir=Path(sample_dir) if sample_dir else None,
        min_confidence=min_confidence,
    )
