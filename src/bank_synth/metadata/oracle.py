"""
Oracle metadata extractor using oracledb.

Extracts table metadata, column definitions, and PK/FK relationships
from Oracle data dictionary views.
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


# Oracle type mapping
ORACLE_TYPE_MAP = {
    "NUMBER": DataType.DECIMAL,
    "INTEGER": DataType.INTEGER,
    "FLOAT": DataType.FLOAT,
    "BINARY_FLOAT": DataType.FLOAT,
    "BINARY_DOUBLE": DataType.DOUBLE,
    "VARCHAR2": DataType.STRING,
    "NVARCHAR2": DataType.STRING,
    "CHAR": DataType.STRING,
    "NCHAR": DataType.STRING,
    "CLOB": DataType.STRING,
    "NCLOB": DataType.STRING,
    "DATE": DataType.TIMESTAMP,  # Oracle DATE includes time
    "TIMESTAMP": DataType.TIMESTAMP,
    "TIMESTAMP WITH TIME ZONE": DataType.TIMESTAMP,
    "TIMESTAMP WITH LOCAL TIME ZONE": DataType.TIMESTAMP,
    "RAW": DataType.BINARY,
    "BLOB": DataType.BINARY,
    "LONG": DataType.STRING,
    "LONG RAW": DataType.BINARY,
}


class OracleMetadataExtractor:
    """
    Extracts metadata from Oracle database catalog.

    Uses Oracle data dictionary views:
    - ALL_TABLES / USER_TABLES
    - ALL_TAB_COLUMNS / USER_TAB_COLUMNS
    - ALL_CONSTRAINTS / USER_CONSTRAINTS
    - ALL_CONS_COLUMNS / USER_CONS_COLUMNS
    """

    def __init__(self, connection_string: str):
        """
        Initialize extractor with Oracle connection.

        Args:
            connection_string: Oracle connection string (user/pwd@host:port/service)
        """
        self.connection_string = connection_string
        self._conn = None

    def connect(self) -> None:
        """Establish database connection."""
        import oracledb

        # Parse connection string: user/pwd@host:port/service
        parts = self.connection_string.split("@")
        user_pwd = parts[0]
        host_service = parts[1] if len(parts) > 1 else ""

        user, password = user_pwd.split("/") if "/" in user_pwd else (user_pwd, "")

        # Build DSN
        if ":" in host_service:
            host_port, service = host_service.rsplit("/", 1) if "/" in host_service else (host_service, "")
            host, port = host_port.split(":") if ":" in host_port else (host_port, "1521")
            dsn = oracledb.makedsn(host, int(port), service_name=service)
        else:
            dsn = host_service

        self._conn = oracledb.connect(user=user, password=password, dsn=dsn)
        logger.info(f"Connected to Oracle database as {user}")

    def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def get_table_metadata(self, schema: str, table_name: str) -> Optional[TableMetadata]:
        """
        Get metadata for a specific table.

        Args:
            schema: Schema/owner name
            table_name: Table name

        Returns:
            TableMetadata or None if table not found
        """
        if not self._conn:
            self.connect()

        cursor = self._conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT table_name, num_rows, comments
            FROM all_tables t
            LEFT JOIN all_tab_comments c
                ON t.owner = c.owner AND t.table_name = c.table_name
            WHERE t.owner = :owner AND t.table_name = :table_name
        """, owner=schema.upper(), table_name=table_name.upper())

        row = cursor.fetchone()
        if not row:
            logger.warning(f"Table not found: {schema}.{table_name}")
            return None

        table_name_db, row_count, comment = row

        # Get columns
        columns = self._get_columns(cursor, schema.upper(), table_name_db)

        # Get primary key
        pk_columns = self._get_primary_key(cursor, schema.upper(), table_name_db)

        # Mark PK columns
        for col in columns:
            if col.name in pk_columns:
                col.is_primary_key = True

        # Get FK columns and mark them
        fk_info = self._get_foreign_key_columns(cursor, schema.upper(), table_name_db)
        for col in columns:
            if col.name in fk_info:
                col.is_foreign_key = True
                col.fk_reference = fk_info[col.name]

        cursor.close()

        return TableMetadata(
            name=table_name_db,
            schema=schema.upper(),
            columns=columns,
            primary_key=pk_columns,
            comment=comment,
            row_count_estimate=row_count,
        )

    def _get_columns(
        self,
        cursor,
        schema: str,
        table_name: str,
    ) -> List[ColumnMetadata]:
        """Get column metadata for a table."""
        cursor.execute("""
            SELECT
                column_name,
                data_type,
                nullable,
                data_length,
                data_precision,
                data_scale,
                data_default
            FROM all_tab_columns
            WHERE owner = :owner AND table_name = :table_name
            ORDER BY column_id
        """, owner=schema, table_name=table_name)

        columns = []
        for row in cursor:
            col_name, data_type, nullable, data_length, precision, scale, default = row

            # Map Oracle type to normalized type
            mapped_type = ORACLE_TYPE_MAP.get(data_type.upper(), DataType.UNKNOWN)

            # Adjust type based on precision/scale for NUMBER
            if data_type.upper() == "NUMBER":
                if scale == 0 and precision is not None:
                    if precision <= 9:
                        mapped_type = DataType.INTEGER
                    else:
                        mapped_type = DataType.BIGINT

            columns.append(ColumnMetadata(
                name=col_name,
                data_type=mapped_type,
                nullable=nullable == "Y",
                precision=precision,
                scale=scale,
                max_length=data_length if mapped_type == DataType.STRING else None,
                default_value=default.strip() if default else None,
            ))

        return columns

    def _get_primary_key(self, cursor, schema: str, table_name: str) -> List[str]:
        """Get primary key columns for a table."""
        cursor.execute("""
            SELECT cc.column_name
            FROM all_constraints c
            JOIN all_cons_columns cc
                ON c.owner = cc.owner
                AND c.constraint_name = cc.constraint_name
            WHERE c.owner = :owner
                AND c.table_name = :table_name
                AND c.constraint_type = 'P'
            ORDER BY cc.position
        """, owner=schema, table_name=table_name)

        return [row[0] for row in cursor]

    def _get_foreign_key_columns(
        self,
        cursor,
        schema: str,
        table_name: str,
    ) -> Dict[str, tuple]:
        """Get FK column info: {column_name: (ref_table, ref_column)}."""
        cursor.execute("""
            SELECT
                cc.column_name,
                rc.table_name as ref_table,
                rcc.column_name as ref_column
            FROM all_constraints c
            JOIN all_cons_columns cc
                ON c.owner = cc.owner
                AND c.constraint_name = cc.constraint_name
            JOIN all_constraints rc
                ON c.r_owner = rc.owner
                AND c.r_constraint_name = rc.constraint_name
            JOIN all_cons_columns rcc
                ON rc.owner = rcc.owner
                AND rc.constraint_name = rcc.constraint_name
                AND cc.position = rcc.position
            WHERE c.owner = :owner
                AND c.table_name = :table_name
                AND c.constraint_type = 'R'
        """, owner=schema, table_name=table_name)

        return {row[0]: (row[1], row[2]) for row in cursor}

    def get_foreign_keys(self, schema: str, table_name: str) -> List[Relationship]:
        """
        Get all foreign key relationships for a table.

        Args:
            schema: Schema/owner name
            table_name: Table name

        Returns:
            List of Relationship objects where table is the child
        """
        if not self._conn:
            self.connect()

        cursor = self._conn.cursor()

        cursor.execute("""
            SELECT
                c.constraint_name,
                c.table_name as child_table,
                rc.table_name as parent_table,
                c.delete_rule
            FROM all_constraints c
            JOIN all_constraints rc
                ON c.r_owner = rc.owner
                AND c.r_constraint_name = rc.constraint_name
            WHERE c.owner = :owner
                AND c.table_name = :table_name
                AND c.constraint_type = 'R'
        """, owner=schema.upper(), table_name=table_name.upper())

        relationships = []
        for row in cursor:
            constraint_name, child_table, parent_table, delete_rule = row

            # Get column mappings
            cursor2 = self._conn.cursor()
            cursor2.execute("""
                SELECT
                    cc.column_name as child_col,
                    rcc.column_name as parent_col
                FROM all_cons_columns cc
                JOIN all_constraints c
                    ON cc.owner = c.owner AND cc.constraint_name = c.constraint_name
                JOIN all_constraints rc
                    ON c.r_owner = rc.owner AND c.r_constraint_name = rc.constraint_name
                JOIN all_cons_columns rcc
                    ON rc.owner = rcc.owner
                    AND rc.constraint_name = rcc.constraint_name
                    AND cc.position = rcc.position
                WHERE cc.owner = :owner
                    AND cc.constraint_name = :constraint_name
                ORDER BY cc.position
            """, owner=schema.upper(), constraint_name=constraint_name)

            child_cols = []
            parent_cols = []
            for col_row in cursor2:
                child_cols.append(col_row[0])
                parent_cols.append(col_row[1])
            cursor2.close()

            relationships.append(Relationship(
                name=constraint_name,
                parent_table=parent_table,
                parent_columns=parent_cols,
                child_table=child_table,
                child_columns=child_cols,
                cardinality="1:N",
                is_optional=delete_rule != "CASCADE",
            ))

        cursor.close()
        return relationships

    def get_all_tables(self, schema: str) -> List[str]:
        """Get all table names in a schema."""
        if not self._conn:
            self.connect()

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM all_tables
            WHERE owner = :owner
            ORDER BY table_name
        """, owner=schema.upper())

        tables = [row[0] for row in cursor]
        cursor.close()
        return tables

    def sample_table(
        self,
        schema: str,
        table_name: str,
        strategy: str = "percent:1",
        max_rows: int = 100000,
    ) -> Any:
        """
        Sample data from a table.

        Args:
            schema: Schema/owner name
            table_name: Table name
            strategy: Sampling strategy (percent:N, rows:N, or full)
            max_rows: Maximum rows to return

        Returns:
            pandas DataFrame with sampled data
        """
        import pandas as pd

        if not self._conn:
            self.connect()

        # Parse strategy
        if strategy.startswith("percent:"):
            pct = float(strategy.split(":")[1])
            sql = f"""
                SELECT * FROM {schema}.{table_name}
                SAMPLE ({pct})
                FETCH FIRST {max_rows} ROWS ONLY
            """
        elif strategy.startswith("rows:"):
            n_rows = int(strategy.split(":")[1])
            sql = f"""
                SELECT * FROM {schema}.{table_name}
                FETCH FIRST {min(n_rows, max_rows)} ROWS ONLY
            """
        else:
            sql = f"""
                SELECT * FROM {schema}.{table_name}
                FETCH FIRST {max_rows} ROWS ONLY
            """

        logger.info(f"Sampling {schema}.{table_name} with strategy: {strategy}")
        return pd.read_sql(sql, self._conn)
