"""
SQL Query Parser for extracting table relationships from queries.

Parses ETL queries, reporting queries, and dynamic test data queries to
automatically discover table relationships from JOIN conditions.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class JoinInfo:
    """Information about a single JOIN relationship extracted from a query."""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str  # INNER, LEFT, RIGHT, FULL, CROSS
    query_source: Optional[str] = None  # Name/identifier of source query
    confidence: float = 1.0  # How confident we are in this relationship

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_table": self.left_table,
            "left_column": self.left_column,
            "right_table": self.right_table,
            "right_column": self.right_column,
            "join_type": self.join_type,
            "query_source": self.query_source,
            "confidence": self.confidence,
        }


@dataclass
class ParsedQuery:
    """Result of parsing a SQL query."""
    query_name: str
    tables: Set[str] = field(default_factory=set)
    joins: List[JoinInfo] = field(default_factory=list)
    columns_used: Dict[str, Set[str]] = field(default_factory=dict)  # table -> columns
    is_valid: bool = True
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_name": self.query_name,
            "tables": list(self.tables),
            "joins": [j.to_dict() for j in self.joins],
            "columns_used": {k: list(v) for k, v in self.columns_used.items()},
            "is_valid": self.is_valid,
            "parse_errors": self.parse_errors,
        }


class QueryParser:
    """
    Parses SQL queries to extract table relationships.

    Supports:
    - Standard SQL JOINs (INNER, LEFT, RIGHT, FULL, CROSS)
    - Oracle-style (+) outer join syntax
    - Subqueries and CTEs
    - Multiple join conditions (AND/OR)
    - Various SQL dialects (Oracle, Hive, ANSI)
    """

    # Patterns for parsing SQL
    TABLE_ALIAS_PATTERN = re.compile(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?',
        re.IGNORECASE
    )

    JOIN_PATTERN = re.compile(
        r'(INNER\s+JOIN|LEFT\s+(?:OUTER\s+)?JOIN|RIGHT\s+(?:OUTER\s+)?JOIN|FULL\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN|JOIN)\s+'
        r'([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?\s*'
        r'(?:ON\s+(.+?))?(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|HAVING|UNION|LIMIT|;|$))',
        re.IGNORECASE | re.DOTALL
    )

    ON_CONDITION_PATTERN = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)',
        re.IGNORECASE
    )

    # Oracle (+) outer join pattern
    ORACLE_JOIN_PATTERN = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*'
        r'(\(\+\))?\s*=\s*'
        r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*'
        r'(\(\+\))?',
        re.IGNORECASE
    )

    FROM_TABLES_PATTERN = re.compile(
        r'FROM\s+((?:[a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?\s*,?\s*)+)',
        re.IGNORECASE
    )

    CTE_PATTERN = re.compile(
        r'WITH\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\((.+?)\)(?:\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\((.+?)\))*',
        re.IGNORECASE | re.DOTALL
    )

    def __init__(self):
        self._alias_map: Dict[str, str] = {}  # alias -> actual table name

    def parse(self, sql: str, query_name: str = "unnamed") -> ParsedQuery:
        """
        Parse a SQL query and extract table relationships.

        Args:
            sql: The SQL query to parse
            query_name: Name/identifier for this query (for tracking)

        Returns:
            ParsedQuery with extracted tables and joins
        """
        result = ParsedQuery(query_name=query_name)
        self._alias_map = {}

        try:
            # Normalize the SQL
            sql = self._normalize_sql(sql)

            # Handle CTEs (WITH clause)
            cte_tables = self._extract_ctes(sql)

            # Extract tables and aliases from FROM clauses
            self._extract_tables_and_aliases(sql, result)

            # Extract explicit JOINs
            self._extract_explicit_joins(sql, result)

            # Extract implicit joins (Oracle-style WHERE conditions)
            self._extract_implicit_joins(sql, result)

            # Remove CTE names from tables (they're not real tables)
            result.tables -= cte_tables

            # Deduplicate joins
            result.joins = self._deduplicate_joins(result.joins)

        except Exception as e:
            result.is_valid = False
            result.parse_errors.append(f"Parse error: {str(e)}")
            logger.warning(f"Failed to parse query '{query_name}': {e}")

        return result

    def parse_multiple(
        self,
        queries: List[Tuple[str, str]],
    ) -> List[ParsedQuery]:
        """
        Parse multiple queries.

        Args:
            queries: List of (query_name, sql) tuples

        Returns:
            List of ParsedQuery results
        """
        return [self.parse(sql, name) for name, sql in queries]

    def parse_file(self, file_path: str) -> List[ParsedQuery]:
        """
        Parse all SQL queries from a file.

        Supports files with multiple queries separated by semicolons,
        or named queries with -- @name: annotations.

        Args:
            file_path: Path to SQL file

        Returns:
            List of ParsedQuery results
        """
        from pathlib import Path

        content = Path(file_path).read_text()
        queries = self._split_queries(content, file_path)
        return [self.parse(sql, name) for name, sql in queries]

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for easier parsing."""
        # Remove comments
        sql = re.sub(r'--.*?$', ' ', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.strip()

        return sql

    def _extract_ctes(self, sql: str) -> Set[str]:
        """Extract CTE (Common Table Expression) names."""
        cte_names = set()

        # Simple CTE extraction
        cte_match = re.match(r'WITH\s+(.+?)\s+SELECT', sql, re.IGNORECASE | re.DOTALL)
        if cte_match:
            cte_section = cte_match.group(1)
            # Find all CTE names before AS
            cte_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', re.IGNORECASE)
            for match in cte_pattern.finditer(cte_section):
                cte_names.add(match.group(1).upper())

        return cte_names

    def _extract_tables_and_aliases(self, sql: str, result: ParsedQuery) -> None:
        """Extract tables and their aliases from the query."""
        # Find FROM clause tables (comma-separated)
        from_match = self.FROM_TABLES_PATTERN.search(sql)
        if from_match:
            tables_section = from_match.group(1)
            # Split by comma for multiple tables
            for table_def in tables_section.split(','):
                table_def = table_def.strip()
                if not table_def:
                    continue

                # Parse "table alias" or "table AS alias"
                parts = re.split(r'\s+(?:AS\s+)?', table_def, maxsplit=1, flags=re.IGNORECASE)
                table_name = self._extract_table_name(parts[0].strip())

                if table_name:
                    result.tables.add(table_name)

                    if len(parts) > 1:
                        alias = parts[1].strip()
                        if alias and not re.match(r'(?:INNER|LEFT|RIGHT|FULL|CROSS|JOIN|WHERE|ON)', alias, re.IGNORECASE):
                            self._alias_map[alias.upper()] = table_name

        # Find tables in JOIN clauses
        for match in self.JOIN_PATTERN.finditer(sql):
            table_name = self._extract_table_name(match.group(2))
            alias = match.group(3)

            if table_name:
                result.tables.add(table_name)
                if alias:
                    self._alias_map[alias.upper()] = table_name

    def _extract_explicit_joins(self, sql: str, result: ParsedQuery) -> None:
        """Extract relationships from explicit JOIN ... ON clauses."""
        # Find the FROM table first
        from_match = self.FROM_TABLES_PATTERN.search(sql)
        from_table = None
        if from_match:
            tables_section = from_match.group(1).split(',')[0]
            parts = re.split(r'\s+(?:AS\s+)?', tables_section.strip(), maxsplit=1, flags=re.IGNORECASE)
            from_table = self._extract_table_name(parts[0].strip())
            if len(parts) > 1:
                alias = parts[1].strip()
                if alias and from_table:
                    self._alias_map[alias.upper()] = from_table

        # Process JOINs
        for match in self.JOIN_PATTERN.finditer(sql):
            join_type = self._normalize_join_type(match.group(1))
            right_table = self._extract_table_name(match.group(2))
            alias = match.group(3)
            on_clause = match.group(4)

            if alias and right_table:
                self._alias_map[alias.upper()] = right_table

            if on_clause:
                # Parse ON conditions
                for cond_match in self.ON_CONDITION_PATTERN.finditer(on_clause):
                    left_ref = cond_match.group(1)
                    right_ref = cond_match.group(2)

                    left_table, left_col = self._resolve_column_reference(left_ref)
                    right_table_resolved, right_col = self._resolve_column_reference(right_ref)

                    if left_table and right_table_resolved and left_col and right_col:
                        result.joins.append(JoinInfo(
                            left_table=left_table,
                            left_column=left_col,
                            right_table=right_table_resolved,
                            right_column=right_col,
                            join_type=join_type,
                            query_source=result.query_name,
                        ))

    def _extract_implicit_joins(self, sql: str, result: ParsedQuery) -> None:
        """Extract relationships from WHERE clause (implicit joins, Oracle-style)."""
        # Find WHERE clause
        where_match = re.search(
            r'WHERE\s+(.+?)(?=\s+(?:GROUP|ORDER|HAVING|UNION|LIMIT|;|$))',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if not where_match:
            return

        where_clause = where_match.group(1)

        # Look for join conditions with Oracle (+) syntax or simple equality
        for match in self.ORACLE_JOIN_PATTERN.finditer(where_clause):
            left_ref = match.group(1)
            left_outer = match.group(2)  # (+) on left side
            right_ref = match.group(3)
            right_outer = match.group(4)  # (+) on right side

            left_table, left_col = self._resolve_column_reference(left_ref)
            right_table, right_col = self._resolve_column_reference(right_ref)

            # Skip if same table (not a join condition)
            if left_table == right_table:
                continue

            # Skip if not a join (single column without table prefix)
            if not left_table or not right_table:
                continue

            # Determine join type from (+) markers
            if left_outer:
                join_type = "RIGHT"
            elif right_outer:
                join_type = "LEFT"
            else:
                join_type = "INNER"

            if left_table and right_table and left_col and right_col:
                result.joins.append(JoinInfo(
                    left_table=left_table,
                    left_column=left_col,
                    right_table=right_table,
                    right_column=right_col,
                    join_type=join_type,
                    query_source=result.query_name,
                ))

    def _extract_table_name(self, ref: str) -> Optional[str]:
        """Extract table name, handling schema.table format."""
        ref = ref.strip()
        if not ref:
            return None

        # Handle schema.table
        if '.' in ref:
            parts = ref.split('.')
            return parts[-1].upper()

        return ref.upper()

    def _resolve_column_reference(self, ref: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve a column reference like 'table.column' or 'alias.column'.

        Returns:
            Tuple of (table_name, column_name) or (None, None)
        """
        ref = ref.strip()

        if '.' in ref:
            parts = ref.split('.')
            if len(parts) >= 2:
                table_or_alias = parts[-2].upper()
                column = parts[-1].upper()

                # Resolve alias to actual table name
                actual_table = self._alias_map.get(table_or_alias, table_or_alias)
                return actual_table, column

        # No table prefix - can't determine the table
        return None, ref.upper() if ref else None

    def _normalize_join_type(self, join_str: str) -> str:
        """Normalize join type to standard form."""
        join_str = join_str.upper()

        if 'LEFT' in join_str:
            return 'LEFT'
        elif 'RIGHT' in join_str:
            return 'RIGHT'
        elif 'FULL' in join_str:
            return 'FULL'
        elif 'CROSS' in join_str:
            return 'CROSS'
        else:
            return 'INNER'

    def _deduplicate_joins(self, joins: List[JoinInfo]) -> List[JoinInfo]:
        """Remove duplicate joins, keeping higher confidence ones."""
        seen: Dict[tuple, JoinInfo] = {}

        for join in joins:
            # Create a canonical key (sorted table names)
            key = tuple(sorted([
                (join.left_table, join.left_column),
                (join.right_table, join.right_column),
            ]))

            if key not in seen or join.confidence > seen[key].confidence:
                seen[key] = join

        return list(seen.values())

    def _split_queries(self, content: str, file_name: str) -> List[Tuple[str, str]]:
        """Split file content into individual named queries."""
        queries = []

        # Look for named queries with -- @name: or -- name: annotations
        named_pattern = re.compile(r'--\s*@?name:\s*(.+?)\n', re.IGNORECASE)

        # Split by semicolons first
        parts = content.split(';')

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Check for name annotation
            name_match = named_pattern.search(part)
            if name_match:
                name = name_match.group(1).strip()
                # Remove the annotation from the SQL
                sql = named_pattern.sub('', part)
            else:
                name = f"{file_name}_query_{i+1}"
                sql = part

            if sql.strip():
                queries.append((name, sql))

        return queries


def extract_relationships_from_queries(
    queries: List[Tuple[str, str]],
) -> Tuple[Set[str], List[JoinInfo]]:
    """
    Convenience function to extract all relationships from multiple queries.

    Args:
        queries: List of (query_name, sql) tuples

    Returns:
        Tuple of (all_tables, all_joins)
    """
    parser = QueryParser()
    all_tables: Set[str] = set()
    all_joins: List[JoinInfo] = []

    for name, sql in queries:
        result = parser.parse(sql, name)
        all_tables.update(result.tables)
        all_joins.extend(result.joins)

    return all_tables, all_joins
