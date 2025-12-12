"""
Relationship Inferrer - Automatically discovers table relationships.

Combines multiple sources to infer relationships:
1. SQL queries (ETL, reporting, dynamic test queries)
2. PK/FK metadata from database catalogs
3. Column naming conventions
4. Data analysis (optional)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    Relationship,
    RelationshipGraph,
    TableMetadata,
)
from bank_synth.discovery.query_parser import JoinInfo, ParsedQuery, QueryParser

logger = logging.getLogger(__name__)


@dataclass
class InferredRelationship:
    """A relationship discovered through inference."""
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    confidence: float  # 0.0 to 1.0
    sources: List[str] = field(default_factory=list)  # Where this relationship was found
    cardinality: str = "1:N"
    is_optional: bool = True

    @property
    def name(self) -> str:
        return f"fk_{self.child_table.lower()}_{self.child_column.lower()}"


@dataclass
class PKFKMetadata:
    """Primary key and foreign key metadata from database catalog."""
    table_name: str
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)  # [{column, ref_table, ref_column}]


class RelationshipInferrer:
    """
    Infers table relationships from multiple sources.

    This class eliminates the need for manual relationship configuration by
    automatically discovering relationships from:

    1. SQL Queries (highest priority for enterprise systems)
       - ETL queries show how data flows between tables
       - Reporting queries show business relationships
       - Dynamic test data queries show test dependencies

    2. PK/FK Metadata (authoritative when available)
       - Database catalog constraints
       - Explicitly defined foreign keys

    3. Column Naming Conventions (fallback)
       - *_id columns matching table names
       - Common patterns like customer_id -> customers.id
    """

    # Common FK column patterns
    FK_PATTERNS = [
        # {table}_id -> {table}.{table}_id
        (r'^([a-z_]+)_id$', lambda m: [(m.group(1), f"{m.group(1)}_id")]),
        # {table}id -> {table}.{table}id
        (r'^([a-z_]+)id$', lambda m: [(m.group(1), f"{m.group(1)}id")]),
        # fk_{table} -> {table}
        (r'^fk_([a-z_]+)$', lambda m: [(m.group(1), f"{m.group(1)}_id")]),
        # {table}_code -> {table}.{table}_code
        (r'^([a-z_]+)_code$', lambda m: [(m.group(1), f"{m.group(1)}_code")]),
        # {table}_type_id -> {table}_types.{table}_type_id
        (r'^([a-z_]+)_type_id$', lambda m: [(f"{m.group(1)}_types", f"{m.group(1)}_type_id")]),
    ]

    def __init__(
        self,
        table_metadata: Dict[str, TableMetadata],
        pkfk_metadata: Optional[List[PKFKMetadata]] = None,
    ):
        """
        Initialize the inferrer.

        Args:
            table_metadata: Dict of table_name -> TableMetadata
            pkfk_metadata: Optional PK/FK metadata from database
        """
        self.table_metadata = {k.upper(): v for k, v in table_metadata.items()}
        self.pkfk_metadata = pkfk_metadata or []
        self._pkfk_by_table: Dict[str, PKFKMetadata] = {
            m.table_name.upper(): m for m in self.pkfk_metadata
        }

        # Inferred relationships with confidence scores
        self._relationships: Dict[str, InferredRelationship] = {}

    def infer_from_queries(
        self,
        queries: List[Tuple[str, str]],
        confidence_boost: float = 0.3,
    ) -> List[InferredRelationship]:
        """
        Infer relationships from SQL queries.

        Queries that are used in production ETL/reporting provide strong evidence
        of actual table relationships.

        Args:
            queries: List of (query_name, sql) tuples
            confidence_boost: Extra confidence for each query a relationship appears in

        Returns:
            List of inferred relationships
        """
        parser = QueryParser()
        parsed = parser.parse_multiple(queries)

        for result in parsed:
            if not result.is_valid:
                logger.warning(f"Skipping invalid query: {result.query_name}")
                continue

            for join in result.joins:
                self._add_from_join(join, confidence_boost)

        logger.info(f"Inferred {len(self._relationships)} relationships from {len(queries)} queries")
        return list(self._relationships.values())

    def infer_from_pkfk_metadata(
        self,
        base_confidence: float = 0.95,
    ) -> List[InferredRelationship]:
        """
        Infer relationships from PK/FK database metadata.

        This is the most authoritative source when available.

        Args:
            base_confidence: Confidence level for catalog-defined FKs

        Returns:
            List of inferred relationships
        """
        for pkfk in self.pkfk_metadata:
            table_name = pkfk.table_name.upper()

            for fk in pkfk.foreign_keys:
                rel_key = self._relationship_key(
                    fk["ref_table"].upper(),
                    fk["ref_column"].upper(),
                    table_name,
                    fk["column"].upper(),
                )

                if rel_key in self._relationships:
                    # Boost existing relationship
                    self._relationships[rel_key].confidence = min(
                        1.0, self._relationships[rel_key].confidence + 0.2
                    )
                    self._relationships[rel_key].sources.append("catalog_fk")
                else:
                    self._relationships[rel_key] = InferredRelationship(
                        parent_table=fk["ref_table"].upper(),
                        parent_column=fk["ref_column"].upper(),
                        child_table=table_name,
                        child_column=fk["column"].upper(),
                        confidence=base_confidence,
                        sources=["catalog_fk"],
                        is_optional=fk.get("nullable", True),
                    )

        logger.info(f"Processed {len(self.pkfk_metadata)} PK/FK metadata entries")
        return list(self._relationships.values())

    def infer_from_column_patterns(
        self,
        base_confidence: float = 0.5,
    ) -> List[InferredRelationship]:
        """
        Infer relationships from column naming patterns.

        This is a fallback when queries and PK/FK metadata are unavailable.

        Args:
            base_confidence: Base confidence for pattern-matched relationships

        Returns:
            List of inferred relationships
        """
        for table_name, table_meta in self.table_metadata.items():
            for col in table_meta.columns:
                col_name_lower = col.name.lower()

                # Skip if already marked as FK or PK
                if col.is_primary_key:
                    continue

                # Try each pattern
                for pattern, extractor in self.FK_PATTERNS:
                    match = re.match(pattern, col_name_lower)
                    if match:
                        potential_refs = extractor(match)
                        for ref_table, ref_col in potential_refs:
                            ref_table_upper = ref_table.upper()

                            # Check if referenced table exists
                            if ref_table_upper in self.table_metadata:
                                ref_meta = self.table_metadata[ref_table_upper]

                                # Check if referenced column exists
                                if ref_meta.get_column(ref_col):
                                    self._add_inferred_relationship(
                                        parent_table=ref_table_upper,
                                        parent_column=ref_col.upper(),
                                        child_table=table_name,
                                        child_column=col.name.upper(),
                                        confidence=base_confidence,
                                        source="pattern_match",
                                    )
                                    break

                            # Also try singular/plural variants
                            variants = self._get_table_variants(ref_table)
                            for variant in variants:
                                if variant.upper() in self.table_metadata:
                                    ref_meta = self.table_metadata[variant.upper()]
                                    for potential_col in [ref_col, "id", f"{ref_table}_id"]:
                                        if ref_meta.get_column(potential_col):
                                            self._add_inferred_relationship(
                                                parent_table=variant.upper(),
                                                parent_column=potential_col.upper(),
                                                child_table=table_name,
                                                child_column=col.name.upper(),
                                                confidence=base_confidence * 0.9,
                                                source="pattern_match_variant",
                                            )
                                            break
                                    break

        logger.info(f"Inferred {len(self._relationships)} relationships from column patterns")
        return list(self._relationships.values())

    def infer_all(
        self,
        queries: Optional[List[Tuple[str, str]]] = None,
        min_confidence: float = 0.3,
    ) -> List[InferredRelationship]:
        """
        Run all inference methods and return combined results.

        Priority order:
        1. PK/FK metadata (highest confidence)
        2. SQL queries (high confidence, especially when multiple queries agree)
        3. Column patterns (fallback)

        Args:
            queries: Optional list of (query_name, sql) tuples
            min_confidence: Minimum confidence threshold for inclusion

        Returns:
            List of inferred relationships meeting confidence threshold
        """
        # Start with PK/FK metadata (most authoritative)
        self.infer_from_pkfk_metadata()

        # Add query-based inference
        if queries:
            self.infer_from_queries(queries)

        # Fill in gaps with pattern matching
        self.infer_from_column_patterns()

        # Filter by confidence
        results = [
            rel for rel in self._relationships.values()
            if rel.confidence >= min_confidence
        ]

        # Sort by confidence (descending)
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            f"Total inferred relationships: {len(results)} "
            f"(filtered from {len(self._relationships)} at confidence >= {min_confidence})"
        )

        return results

    def build_relationship_graph(
        self,
        queries: Optional[List[Tuple[str, str]]] = None,
        min_confidence: float = 0.3,
    ) -> RelationshipGraph:
        """
        Build a complete RelationshipGraph from inferred relationships.

        Args:
            queries: Optional SQL queries for inference
            min_confidence: Minimum confidence threshold

        Returns:
            RelationshipGraph with tables and relationships
        """
        inferred = self.infer_all(queries, min_confidence)

        graph = RelationshipGraph()

        # Add all tables
        for table_name, table_meta in self.table_metadata.items():
            graph.add_table(table_meta)

        # Add relationships
        for rel in inferred:
            # Update column metadata
            child_table = graph.get_table(rel.child_table)
            if child_table:
                col = child_table.get_column(rel.child_column)
                if col:
                    col.is_foreign_key = True
                    col.fk_reference = (rel.parent_table, rel.parent_column)

            # Create relationship
            relationship = Relationship(
                name=rel.name,
                parent_table=rel.parent_table,
                parent_columns=[rel.parent_column],
                child_table=rel.child_table,
                child_columns=[rel.child_column],
                cardinality=rel.cardinality,
                is_optional=rel.is_optional,
            )
            graph.add_relationship(relationship)

        return graph

    def _add_from_join(self, join: JoinInfo, confidence_boost: float) -> None:
        """Add relationship from a parsed JOIN."""
        # Determine parent/child based on join type and naming
        parent_table, parent_col, child_table, child_col = self._determine_direction(
            join.left_table, join.left_column,
            join.right_table, join.right_column,
        )

        base_confidence = 0.7  # Queries provide good evidence

        # Adjust based on join type
        if join.join_type == "INNER":
            is_optional = False
        else:
            is_optional = True

        self._add_inferred_relationship(
            parent_table=parent_table,
            parent_column=parent_col,
            child_table=child_table,
            child_column=child_col,
            confidence=base_confidence,
            source=f"query:{join.query_source}",
            is_optional=is_optional,
            confidence_boost=confidence_boost,
        )

    def _add_inferred_relationship(
        self,
        parent_table: str,
        parent_column: str,
        child_table: str,
        child_column: str,
        confidence: float,
        source: str,
        is_optional: bool = True,
        confidence_boost: float = 0.0,
    ) -> None:
        """Add or update an inferred relationship."""
        rel_key = self._relationship_key(parent_table, parent_column, child_table, child_column)

        if rel_key in self._relationships:
            existing = self._relationships[rel_key]
            # Boost confidence for multiple sources
            existing.confidence = min(1.0, existing.confidence + confidence_boost)
            if source not in existing.sources:
                existing.sources.append(source)
        else:
            self._relationships[rel_key] = InferredRelationship(
                parent_table=parent_table,
                parent_column=parent_column,
                child_table=child_table,
                child_column=child_column,
                confidence=confidence,
                sources=[source],
                is_optional=is_optional,
            )

    def _relationship_key(
        self,
        parent_table: str,
        parent_col: str,
        child_table: str,
        child_col: str,
    ) -> str:
        """Create a unique key for a relationship."""
        return f"{parent_table}.{parent_col}->{child_table}.{child_col}"

    def _determine_direction(
        self,
        table1: str,
        col1: str,
        table2: str,
        col2: str,
    ) -> Tuple[str, str, str, str]:
        """
        Determine parent/child direction of a relationship.

        Uses multiple heuristics:
        1. PK/FK metadata
        2. Column naming (parent usually has PK, child has FK)
        3. Table relationships in metadata

        Returns:
            (parent_table, parent_col, child_table, child_col)
        """
        table1_upper = table1.upper()
        table2_upper = table2.upper()

        # Check if either column is a PK
        table1_meta = self.table_metadata.get(table1_upper)
        table2_meta = self.table_metadata.get(table2_upper)

        col1_is_pk = False
        col2_is_pk = False

        if table1_meta:
            col1_meta = table1_meta.get_column(col1)
            col1_is_pk = col1_meta.is_primary_key if col1_meta else False

        if table2_meta:
            col2_meta = table2_meta.get_column(col2)
            col2_is_pk = col2_meta.is_primary_key if col2_meta else False

        # If one is PK, it's the parent
        if col1_is_pk and not col2_is_pk:
            return table1_upper, col1.upper(), table2_upper, col2.upper()
        if col2_is_pk and not col1_is_pk:
            return table2_upper, col2.upper(), table1_upper, col1.upper()

        # Check PKFK metadata
        if table1_upper in self._pkfk_by_table:
            pkfk = self._pkfk_by_table[table1_upper]
            for fk in pkfk.foreign_keys:
                if fk["column"].upper() == col1.upper():
                    # col1 is FK, so table1 is child
                    return table2_upper, col2.upper(), table1_upper, col1.upper()

        if table2_upper in self._pkfk_by_table:
            pkfk = self._pkfk_by_table[table2_upper]
            for fk in pkfk.foreign_keys:
                if fk["column"].upper() == col2.upper():
                    # col2 is FK, so table2 is child
                    return table1_upper, col1.upper(), table2_upper, col2.upper()

        # Heuristic: column name containing table name is likely the FK
        # e.g., customer_id in accounts table -> customers is parent
        col1_lower = col1.lower()
        col2_lower = col2.lower()

        # Check if col2 looks like it references table1
        if any(x in col2_lower for x in [table1_upper.lower(), table1_upper.lower().rstrip('s')]):
            return table1_upper, col1.upper(), table2_upper, col2.upper()

        # Check if col1 looks like it references table2
        if any(x in col1_lower for x in [table2_upper.lower(), table2_upper.lower().rstrip('s')]):
            return table2_upper, col2.upper(), table1_upper, col1.upper()

        # Default: alphabetically first table is parent (arbitrary but consistent)
        if table1_upper < table2_upper:
            return table1_upper, col1.upper(), table2_upper, col2.upper()
        return table2_upper, col2.upper(), table1_upper, col1.upper()

    def _get_table_variants(self, table_name: str) -> List[str]:
        """Get singular/plural variants of a table name."""
        variants = []
        table_lower = table_name.lower()

        # Plural -> singular
        if table_lower.endswith('ies'):
            variants.append(table_lower[:-3] + 'y')
        elif table_lower.endswith('es'):
            variants.append(table_lower[:-2])
        elif table_lower.endswith('s'):
            variants.append(table_lower[:-1])

        # Singular -> plural
        if table_lower.endswith('y'):
            variants.append(table_lower[:-1] + 'ies')
        elif table_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
            variants.append(table_lower + 'es')
        else:
            variants.append(table_lower + 's')

        return variants


def infer_relationships(
    tables: Dict[str, TableMetadata],
    queries: Optional[List[Tuple[str, str]]] = None,
    pkfk_metadata: Optional[List[PKFKMetadata]] = None,
    min_confidence: float = 0.3,
) -> RelationshipGraph:
    """
    Convenience function to infer relationships and build a graph.

    Args:
        tables: Dict of table_name -> TableMetadata
        queries: Optional SQL queries for inference
        pkfk_metadata: Optional PK/FK metadata
        min_confidence: Minimum confidence threshold

    Returns:
        RelationshipGraph with inferred relationships
    """
    inferrer = RelationshipInferrer(tables, pkfk_metadata)
    return inferrer.build_relationship_graph(queries, min_confidence)
