"""
Core data models for the bank_synth package.

Defines the fundamental data structures used throughout the system including
metadata, relationships, and configuration objects.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DataType(str, Enum):
    """Normalized data types across Oracle and Hive."""
    STRING = "string"
    INTEGER = "integer"
    BIGINT = "bigint"
    DECIMAL = "decimal"
    FLOAT = "float"
    DOUBLE = "double"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    BINARY = "binary"
    UNKNOWN = "unknown"


class PrivacyLevel(str, Enum):
    """Privacy classification for columns."""
    PUBLIC = "public"           # No restrictions
    INTERNAL = "internal"       # Mask in reports, generate synthetic
    SENSITIVE = "sensitive"     # Format-only generation (e.g., SSN pattern)
    PII = "pii"                 # Drop or fully anonymize
    RESTRICTED = "restricted"   # Drop from generation


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    data_type: DataType
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_reference: Optional[Tuple[str, str]] = None  # (table, column)
    precision: Optional[int] = None
    scale: Optional[int] = None
    max_length: Optional[int] = None
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    default_value: Optional[Any] = None
    comment: Optional[str] = None

    # Generation hints
    format_pattern: Optional[str] = None  # Regex or pattern for generation
    value_range: Optional[Tuple[Any, Any]] = None  # (min, max)
    allowed_values: Optional[List[Any]] = None  # Enum values
    faker_provider: Optional[str] = None  # Faker method name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "fk_reference": self.fk_reference,
            "precision": self.precision,
            "scale": self.scale,
            "max_length": self.max_length,
            "privacy_level": self.privacy_level.value,
            "default_value": self.default_value,
            "comment": self.comment,
            "format_pattern": self.format_pattern,
            "value_range": self.value_range,
            "allowed_values": self.allowed_values,
            "faker_provider": self.faker_provider,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ColumnMetadata:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            data_type=DataType(data["data_type"]),
            nullable=data.get("nullable", True),
            is_primary_key=data.get("is_primary_key", False),
            is_foreign_key=data.get("is_foreign_key", False),
            fk_reference=tuple(data["fk_reference"]) if data.get("fk_reference") else None,
            precision=data.get("precision"),
            scale=data.get("scale"),
            max_length=data.get("max_length"),
            privacy_level=PrivacyLevel(data.get("privacy_level", "public")),
            default_value=data.get("default_value"),
            comment=data.get("comment"),
            format_pattern=data.get("format_pattern"),
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,
            allowed_values=data.get("allowed_values"),
            faker_provider=data.get("faker_provider"),
        )


@dataclass
class TableMetadata:
    """Metadata for a database table."""
    name: str
    schema: Optional[str] = None
    columns: List[ColumnMetadata] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    row_count_estimate: Optional[int] = None
    is_reference_table: bool = False  # Code/lookup tables

    @property
    def full_name(self) -> str:
        """Return schema-qualified table name."""
        return f"{self.schema}.{self.name}" if self.schema else self.name

    @property
    def column_names(self) -> List[str]:
        """Return list of column names."""
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnMetadata]:
        """Get column by name (case-insensitive)."""
        name_lower = name.lower()
        for col in self.columns:
            if col.name.lower() == name_lower:
                return col
        return None

    def get_pk_columns(self) -> List[ColumnMetadata]:
        """Get primary key columns."""
        return [c for c in self.columns if c.is_primary_key]

    def get_fk_columns(self) -> List[ColumnMetadata]:
        """Get foreign key columns."""
        return [c for c in self.columns if c.is_foreign_key]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "schema": self.schema,
            "columns": [c.to_dict() for c in self.columns],
            "primary_key": self.primary_key,
            "comment": self.comment,
            "row_count_estimate": self.row_count_estimate,
            "is_reference_table": self.is_reference_table,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableMetadata:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            schema=data.get("schema"),
            columns=[ColumnMetadata.from_dict(c) for c in data.get("columns", [])],
            primary_key=data.get("primary_key", []),
            comment=data.get("comment"),
            row_count_estimate=data.get("row_count_estimate"),
            is_reference_table=data.get("is_reference_table", False),
        )


@dataclass
class Relationship:
    """Represents a foreign key relationship between tables."""
    name: str
    parent_table: str
    parent_columns: List[str]
    child_table: str
    child_columns: List[str]
    cardinality: str = "1:N"  # 1:1, 1:N, N:M
    is_optional: bool = True  # Can child exist without parent?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "parent_table": self.parent_table,
            "parent_columns": self.parent_columns,
            "child_table": self.child_table,
            "child_columns": self.child_columns,
            "cardinality": self.cardinality,
            "is_optional": self.is_optional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relationship:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            parent_table=data["parent_table"],
            parent_columns=data["parent_columns"],
            child_table=data["child_table"],
            child_columns=data["child_columns"],
            cardinality=data.get("cardinality", "1:N"),
            is_optional=data.get("is_optional", True),
        )


@dataclass
class RelationshipGraph:
    """Graph of table relationships for dependency resolution."""
    tables: Dict[str, TableMetadata] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)

    def add_table(self, table: TableMetadata) -> None:
        """Add a table to the graph."""
        self.tables[table.name.lower()] = table

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the graph."""
        self.relationships.append(rel)

    def get_table(self, name: str) -> Optional[TableMetadata]:
        """Get table by name (case-insensitive)."""
        return self.tables.get(name.lower())

    def get_parent_tables(self, table_name: str) -> List[str]:
        """Get all parent tables (tables this table depends on via FK)."""
        table_lower = table_name.lower()
        parents = []
        for rel in self.relationships:
            if rel.child_table.lower() == table_lower:
                parents.append(rel.parent_table)
        return parents

    def get_child_tables(self, table_name: str) -> List[str]:
        """Get all child tables (tables that depend on this table via FK)."""
        table_lower = table_name.lower()
        children = []
        for rel in self.relationships:
            if rel.parent_table.lower() == table_lower:
                children.append(rel.child_table)
        return children

    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """Get all relationships involving a table (as parent or child)."""
        table_lower = table_name.lower()
        return [
            rel for rel in self.relationships
            if rel.parent_table.lower() == table_lower or rel.child_table.lower() == table_lower
        ]

    def get_dependency_closure(
        self,
        target_table: str,
        include_children: bool = False
    ) -> Tuple[Set[str], List[str]]:
        """
        Get all tables needed to generate the target table.

        Returns:
            Tuple of (set of all tables in closure, topologically sorted list)
        """
        closure: Set[str] = set()

        def add_parents(table: str) -> None:
            """Recursively add parent tables."""
            if table.lower() in closure:
                return
            closure.add(table.lower())
            for parent in self.get_parent_tables(table):
                add_parents(parent)

        def add_children(table: str) -> None:
            """Recursively add child tables."""
            if table.lower() in closure:
                return
            closure.add(table.lower())
            for child in self.get_child_tables(table):
                add_children(child)

        # Add target and all parents
        add_parents(target_table)

        # Optionally add children
        if include_children:
            for table in list(closure):
                add_children(table)

        # Topologically sort
        sorted_tables = self._topological_sort(closure)

        return closure, sorted_tables

    def _topological_sort(self, tables: Set[str]) -> List[str]:
        """Topologically sort tables by dependencies."""
        # Build adjacency list for tables in closure
        in_degree: Dict[str, int] = {t: 0 for t in tables}
        adj: Dict[str, List[str]] = {t: [] for t in tables}

        for rel in self.relationships:
            parent = rel.parent_table.lower()
            child = rel.child_table.lower()
            if parent in tables and child in tables:
                adj[parent].append(child)
                in_degree[child] += 1

        # Kahn's algorithm
        queue = [t for t in tables if in_degree[t] == 0]
        result = []

        while queue:
            # Sort for determinism
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(tables):
            # Cycle detected - fall back to simple sort
            remaining = tables - set(result)
            result.extend(sorted(remaining))

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tables": {name: t.to_dict() for name, t in self.tables.items()},
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationshipGraph:
        """Create from dictionary."""
        graph = cls()
        for name, tdata in data.get("tables", {}).items():
            graph.tables[name] = TableMetadata.from_dict(tdata)
        for rdata in data.get("relationships", []):
            graph.relationships.append(Relationship.from_dict(rdata))
        return graph


@dataclass
class GenerationConfig:
    """Configuration for a generation run."""
    target_table: str
    target_rows: int
    include_children: bool = False
    seed: Optional[int] = None
    output_formats: List[str] = field(default_factory=lambda: ["hive", "oracle"])
    output_dir: Path = field(default_factory=lambda: Path("output"))
    run_id: Optional[str] = None

    # Row count overrides for specific tables
    table_row_counts: Dict[str, int] = field(default_factory=dict)

    # Scale factor for parent tables
    parent_scale_factor: float = 0.1  # Parents get 10% of target rows by default

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ColumnStats:
    """Statistical summary of a column for training."""
    name: str
    data_type: DataType
    null_fraction: float = 0.0
    distinct_count: int = 0
    total_count: int = 0

    # Type-specific stats
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    # Distribution info
    value_frequencies: Optional[Dict[Any, int]] = None
    is_unique: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "null_fraction": self.null_fraction,
            "distinct_count": self.distinct_count,
            "total_count": self.total_count,
            "min_value": self.min_value if not isinstance(self.min_value, (datetime,)) else str(self.min_value),
            "max_value": self.max_value if not isinstance(self.max_value, (datetime,)) else str(self.max_value),
            "mean": self.mean,
            "std": self.std,
            "value_frequencies": self.value_frequencies,
            "is_unique": self.is_unique,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ColumnStats:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            data_type=DataType(data["data_type"]),
            null_fraction=data.get("null_fraction", 0.0),
            distinct_count=data.get("distinct_count", 0),
            total_count=data.get("total_count", 0),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            mean=data.get("mean"),
            std=data.get("std"),
            value_frequencies=data.get("value_frequencies"),
            is_unique=data.get("is_unique", False),
        )


@dataclass
class TableStats:
    """Statistical summary of a table for training."""
    name: str
    row_count: int
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_stats": {k: v.to_dict() for k, v in self.column_stats.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableStats:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            row_count=data["row_count"],
            column_stats={k: ColumnStats.from_dict(v) for k, v in data.get("column_stats", {}).items()},
        )


@dataclass
class ModelPack:
    """
    Portable model artifact containing everything needed for generation.

    This is the output of training and the input for generation.
    """
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Metadata
    schema_name: Optional[str] = None
    relationship_graph: RelationshipGraph = field(default_factory=RelationshipGraph)

    # Training info
    tables_trained: List[str] = field(default_factory=list)
    sample_strategy: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Statistics from training data
    table_stats: Dict[str, TableStats] = field(default_factory=dict)

    # Learned models (SDV synthesizers, serialized)
    synthesizers: Dict[str, bytes] = field(default_factory=dict)

    # Encoders for categorical columns
    encoders: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Privacy policy applied
    privacy_policy: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save model pack to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        metadata = {
            "version": self.version,
            "created_at": self.created_at,
            "schema_name": self.schema_name,
            "tables_trained": self.tables_trained,
            "sample_strategy": self.sample_strategy,
            "training_config": self.training_config,
            "privacy_policy": self.privacy_policy,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save relationship graph
        with open(path / "relationship_graph.json", "w") as f:
            json.dump(self.relationship_graph.to_dict(), f, indent=2)

        # Save table stats
        stats_dict = {k: v.to_dict() for k, v in self.table_stats.items()}
        with open(path / "table_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)

        # Save synthesizers (pickle)
        with open(path / "synthesizers.pkl", "wb") as f:
            pickle.dump(self.synthesizers, f)

        # Save encoders
        with open(path / "encoders.json", "w") as f:
            json.dump(self.encoders, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ModelPack:
        """Load model pack from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load relationship graph
        with open(path / "relationship_graph.json", "r") as f:
            graph_data = json.load(f)

        # Load table stats
        with open(path / "table_stats.json", "r") as f:
            stats_data = json.load(f)

        # Load synthesizers
        with open(path / "synthesizers.pkl", "rb") as f:
            synthesizers = pickle.load(f)

        # Load encoders
        with open(path / "encoders.json", "r") as f:
            encoders = json.load(f)

        return cls(
            version=metadata.get("version", "1.0.0"),
            created_at=metadata.get("created_at", ""),
            schema_name=metadata.get("schema_name"),
            relationship_graph=RelationshipGraph.from_dict(graph_data),
            tables_trained=metadata.get("tables_trained", []),
            sample_strategy=metadata.get("sample_strategy"),
            training_config=metadata.get("training_config", {}),
            table_stats={k: TableStats.from_dict(v) for k, v in stats_data.items()},
            synthesizers=synthesizers,
            encoders=encoders,
            privacy_policy=metadata.get("privacy_policy", {}),
        )
