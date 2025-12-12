"""
Generator component for producing synthetic data from trained models.

Handles:
- Dependency resolution and closure calculation
- Topological ordering for generation sequence
- FK consistency via parent key sampling
- Privacy-aware column generation
"""

from __future__ import annotations

import logging
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    GenerationConfig,
    ModelPack,
    PrivacyLevel,
    Relationship,
    TableMetadata,
)

logger = logging.getLogger(__name__)


class Generator:
    """
    Generates synthetic data using trained model packs.

    The generator:
    1. Resolves dependency closure for target table
    2. Determines generation order via topological sort
    3. Generates parent tables first
    4. Ensures FK consistency by sampling from generated parent keys
    5. Outputs data in requested formats
    """

    def __init__(
        self,
        model_pack: ModelPack,
        config: GenerationConfig,
    ):
        """
        Initialize generator.

        Args:
            model_pack: Trained model pack with synthesizers
            config: Generation configuration
        """
        self.model_pack = model_pack
        self.config = config
        self.faker = Faker()

        # Set seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            Faker.seed(config.seed)

        # Generated data storage
        self._generated_data: Dict[str, pd.DataFrame] = {}

        # Parent key pools for FK sampling
        self._parent_keys: Dict[str, Dict[str, List[Any]]] = {}

    @classmethod
    def from_model_path(cls, model_path: Path, config: GenerationConfig) -> Generator:
        """Create generator from saved model pack path."""
        model_pack = ModelPack.load(model_path)
        return cls(model_pack, config)

    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for target table and dependencies.

        Returns:
            Dictionary mapping table names to generated DataFrames
        """
        logger.info(f"Generating data for target: {self.config.target_table}")

        # Get dependency closure and generation order
        closure, sorted_tables = self.model_pack.relationship_graph.get_dependency_closure(
            self.config.target_table,
            include_children=self.config.include_children,
        )

        logger.info(f"Dependency closure: {len(closure)} tables")
        logger.info(f"Generation order: {sorted_tables}")

        # Calculate row counts for each table
        row_counts = self._calculate_row_counts(sorted_tables)

        # Generate tables in order
        for table_name in tqdm(sorted_tables, desc="Generating tables"):
            table_upper = table_name.upper()
            target_rows = row_counts.get(table_upper, self.config.target_rows)

            logger.info(f"Generating {target_rows} rows for {table_name}")

            df = self._generate_table(table_upper, target_rows)
            self._generated_data[table_upper] = df

            # Extract keys for child FK sampling
            self._extract_parent_keys(table_upper, df)

        return self._generated_data

    def _calculate_row_counts(self, tables: List[str]) -> Dict[str, int]:
        """Calculate row counts for each table in generation order."""
        row_counts = {}
        target_upper = self.config.target_table.upper()

        for table_name in tables:
            table_upper = table_name.upper()

            # Check for explicit override
            if table_upper in self.config.table_row_counts:
                row_counts[table_upper] = self.config.table_row_counts[table_upper]
            elif table_upper == target_upper:
                row_counts[table_upper] = self.config.target_rows
            else:
                # Parent tables: scale down, but maintain minimum
                # Check if this is a reference/lookup table
                table_meta = self.model_pack.relationship_graph.get_table(table_name)
                if table_meta and table_meta.is_reference_table:
                    # Reference tables: use training size or explicit count
                    stats = self.model_pack.table_stats.get(table_upper)
                    if stats:
                        row_counts[table_upper] = min(stats.row_count, 1000)
                    else:
                        row_counts[table_upper] = 100
                else:
                    # Regular parent: scale based on factor
                    scaled = int(self.config.target_rows * self.config.parent_scale_factor)
                    row_counts[table_upper] = max(scaled, 10)  # Minimum 10 rows

        return row_counts

    def _generate_table(self, table_name: str, num_rows: int) -> pd.DataFrame:
        """Generate data for a single table."""
        table_meta = self.model_pack.relationship_graph.get_table(table_name)

        # Try SDV synthesizer first
        if table_name in self.model_pack.synthesizers:
            df = self._generate_with_synthesizer(table_name, num_rows)
        else:
            df = self._generate_fallback(table_name, num_rows, table_meta)

        # Apply FK constraints
        df = self._apply_fk_constraints(table_name, df)

        # Apply privacy transformations (generate format-only for sensitive)
        df = self._apply_privacy_generation(table_name, df, table_meta)

        # Ensure PK uniqueness
        df = self._ensure_pk_uniqueness(table_name, df, table_meta)

        return df

    def _generate_with_synthesizer(self, table_name: str, num_rows: int) -> pd.DataFrame:
        """Generate using trained SDV synthesizer."""
        try:
            synthesizer_bytes = self.model_pack.synthesizers[table_name]
            synthesizer = pickle.loads(synthesizer_bytes)

            # Check if it's a fallback synthesizer
            if isinstance(synthesizer, dict) and synthesizer.get("_type") == "fallback":
                return self._generate_from_fallback_model(synthesizer, num_rows)

            # SDV synthesizer
            df = synthesizer.sample(num_rows=num_rows)
            return df

        except Exception as e:
            logger.error(f"Error using synthesizer for {table_name}: {e}")
            table_meta = self.model_pack.relationship_graph.get_table(table_name)
            return self._generate_fallback(table_name, num_rows, table_meta)

    def _generate_from_fallback_model(
        self,
        model: Dict[str, Any],
        num_rows: int,
    ) -> pd.DataFrame:
        """Generate from fallback model (stored distributions)."""
        data = {}

        for col_name, col_info in model["columns"].items():
            null_fraction = col_info.get("null_fraction", 0)

            if "value_counts" in col_info:
                # Categorical: sample from distribution
                values = list(col_info["value_counts"].keys())
                probs = list(col_info["value_counts"].values())
                col_data = np.random.choice(values, size=num_rows, p=probs)
            elif "mean" in col_info:
                # Numeric: sample from normal distribution
                mean = col_info["mean"]
                std = col_info.get("std", 1.0) or 1.0
                min_val = col_info.get("min", mean - 3 * std)
                max_val = col_info.get("max", mean + 3 * std)

                col_data = np.random.normal(mean, std, num_rows)
                col_data = np.clip(col_data, min_val, max_val)

                # Convert to int if original was int
                if "int" in col_info.get("dtype", "").lower():
                    col_data = col_data.astype(int)
            elif "sample_values" in col_info:
                # String: sample from observed values
                sample_values = col_info["sample_values"]
                if sample_values:
                    col_data = np.random.choice(sample_values, size=num_rows)
                else:
                    col_data = [self.faker.text(max_nb_chars=50) for _ in range(num_rows)]
            else:
                # Default: generate random strings
                col_data = [self.faker.text(max_nb_chars=50) for _ in range(num_rows)]

            # Apply nulls
            if null_fraction > 0:
                null_mask = np.random.random(num_rows) < null_fraction
                col_data = np.where(null_mask, None, col_data)

            data[col_name] = col_data

        return pd.DataFrame(data)

    def _generate_fallback(
        self,
        table_name: str,
        num_rows: int,
        table_meta: Optional[TableMetadata],
    ) -> pd.DataFrame:
        """Generate data using metadata and Faker when no synthesizer available."""
        data = {}

        if not table_meta or not table_meta.columns:
            logger.warning(f"No metadata for {table_name}, generating minimal data")
            return pd.DataFrame({"ID": range(1, num_rows + 1)})

        for col_meta in table_meta.columns:
            col_name = col_meta.name.upper()
            col_data = self._generate_column(col_meta, num_rows)
            data[col_name] = col_data

        return pd.DataFrame(data)

    def _generate_column(self, col_meta: ColumnMetadata, num_rows: int) -> List[Any]:
        """Generate data for a single column based on metadata."""
        # Use Faker provider if specified
        if col_meta.faker_provider:
            return self._generate_with_faker(col_meta.faker_provider, num_rows)

        # Use allowed values if specified
        if col_meta.allowed_values:
            return list(np.random.choice(col_meta.allowed_values, size=num_rows))

        # Generate based on data type
        data_type = col_meta.data_type
        null_fraction = 0.05 if col_meta.nullable else 0

        if data_type in (DataType.INTEGER, DataType.BIGINT):
            if col_meta.value_range:
                min_val, max_val = col_meta.value_range
            else:
                min_val, max_val = 1, 1000000
            data = np.random.randint(min_val, max_val + 1, size=num_rows)

        elif data_type in (DataType.DECIMAL, DataType.FLOAT, DataType.DOUBLE):
            if col_meta.value_range:
                min_val, max_val = col_meta.value_range
            else:
                min_val, max_val = 0.0, 10000.0
            data = np.random.uniform(min_val, max_val, size=num_rows)
            if col_meta.scale:
                data = np.round(data, col_meta.scale)

        elif data_type == DataType.DATE:
            start = datetime.now() - timedelta(days=365 * 5)
            end = datetime.now()
            data = [
                start + timedelta(days=random.randint(0, (end - start).days))
                for _ in range(num_rows)
            ]
            data = [d.date() for d in data]

        elif data_type == DataType.TIMESTAMP:
            start = datetime.now() - timedelta(days=365 * 5)
            end = datetime.now()
            data = [
                start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
                for _ in range(num_rows)
            ]

        elif data_type == DataType.BOOLEAN:
            data = list(np.random.choice([True, False], size=num_rows))

        else:  # STRING and others
            max_len = col_meta.max_length or 50
            data = [self.faker.text(max_nb_chars=min(max_len, 200))[:max_len] for _ in range(num_rows)]

        # Apply nulls
        if null_fraction > 0:
            null_mask = np.random.random(num_rows) < null_fraction
            data = [None if null_mask[i] else data[i] for i in range(num_rows)]

        return data

    def _generate_with_faker(self, provider: str, num_rows: int) -> List[Any]:
        """Generate data using Faker provider."""
        try:
            faker_method = getattr(self.faker, provider)
            return [faker_method() for _ in range(num_rows)]
        except AttributeError:
            logger.warning(f"Unknown Faker provider: {provider}")
            return [self.faker.text(max_nb_chars=50) for _ in range(num_rows)]

    def _apply_fk_constraints(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply foreign key constraints by sampling from parent keys."""
        df = df.copy()

        # Get relationships where this table is the child
        for rel in self.model_pack.relationship_graph.relationships:
            if rel.child_table.upper() != table_name.upper():
                continue

            parent_upper = rel.parent_table.upper()
            if parent_upper not in self._parent_keys:
                logger.warning(f"Parent keys not available for {parent_upper}")
                continue

            # Sample FK values from parent keys
            for i, (child_col, parent_col) in enumerate(
                zip(rel.child_columns, rel.parent_columns)
            ):
                child_col_upper = child_col.upper()
                parent_col_upper = parent_col.upper()

                if parent_col_upper not in self._parent_keys.get(parent_upper, {}):
                    logger.warning(f"Parent column {parent_col_upper} not in keys for {parent_upper}")
                    continue

                parent_values = self._parent_keys[parent_upper][parent_col_upper]
                if not parent_values:
                    continue

                # Sample with replacement
                sampled_fks = np.random.choice(parent_values, size=len(df), replace=True)

                # Handle optional relationships (some nulls)
                if rel.is_optional:
                    null_mask = np.random.random(len(df)) < 0.05
                    sampled_fks = np.where(null_mask, None, sampled_fks)

                df[child_col_upper] = sampled_fks

        return df

    def _extract_parent_keys(self, table_name: str, df: pd.DataFrame) -> None:
        """Extract primary key values for FK sampling by child tables."""
        table_meta = self.model_pack.relationship_graph.get_table(table_name)
        if not table_meta:
            return

        self._parent_keys[table_name.upper()] = {}

        # Extract PK columns
        for pk_col in table_meta.primary_key:
            pk_col_upper = pk_col.upper()
            if pk_col_upper in df.columns:
                values = df[pk_col_upper].dropna().tolist()
                self._parent_keys[table_name.upper()][pk_col_upper] = values

        # Also extract columns that are referenced by FKs
        for rel in self.model_pack.relationship_graph.relationships:
            if rel.parent_table.upper() == table_name.upper():
                for parent_col in rel.parent_columns:
                    parent_col_upper = parent_col.upper()
                    if parent_col_upper in df.columns:
                        values = df[parent_col_upper].dropna().tolist()
                        self._parent_keys[table_name.upper()][parent_col_upper] = values

    def _apply_privacy_generation(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> pd.DataFrame:
        """Apply privacy-aware generation for sensitive columns."""
        df = df.copy()
        privacy_policy = self.model_pack.privacy_policy.get(table_name.lower(), {})

        for col_name in df.columns:
            col_meta = table_meta.get_column(col_name) if table_meta else None
            col_policy = privacy_policy.get(col_name.lower(), {})

            privacy_level = PrivacyLevel.PUBLIC
            format_pattern = None
            faker_provider = None

            # Get settings from column metadata
            if col_meta:
                privacy_level = col_meta.privacy_level
                format_pattern = col_meta.format_pattern
                faker_provider = col_meta.faker_provider

            # Override from policy
            if isinstance(col_policy, str):
                privacy_level = PrivacyLevel(col_policy.lower())
            elif isinstance(col_policy, dict):
                if "level" in col_policy:
                    privacy_level = PrivacyLevel(col_policy["level"].lower())
                if "format_pattern" in col_policy:
                    format_pattern = col_policy["format_pattern"]
                if "faker_provider" in col_policy:
                    faker_provider = col_policy["faker_provider"]

            # Apply based on privacy level
            if privacy_level == PrivacyLevel.RESTRICTED:
                df = df.drop(columns=[col_name])

            elif privacy_level == PrivacyLevel.PII:
                # Generate with Faker if provider specified, else null
                if faker_provider:
                    df[col_name] = [
                        getattr(self.faker, faker_provider)()
                        for _ in range(len(df))
                    ]
                else:
                    df[col_name] = None

            elif privacy_level == PrivacyLevel.SENSITIVE:
                # Generate format-preserving data
                if format_pattern:
                    df[col_name] = [
                        self._generate_from_pattern(format_pattern)
                        for _ in range(len(df))
                    ]
                elif faker_provider:
                    df[col_name] = [
                        getattr(self.faker, faker_provider)()
                        for _ in range(len(df))
                    ]

        return df

    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate string matching a pattern (simple implementation)."""
        result = []
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == "X":
                result.append(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            elif c == "x":
                result.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
            elif c == "9":
                result.append(random.choice("0123456789"))
            elif c == "#":
                result.append(random.choice("0123456789"))
            else:
                result.append(c)
            i += 1
        return "".join(result)

    def _ensure_pk_uniqueness(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> pd.DataFrame:
        """Ensure primary key columns have unique values."""
        if not table_meta or not table_meta.primary_key:
            return df

        df = df.copy()

        for pk_col in table_meta.primary_key:
            pk_col_upper = pk_col.upper()
            if pk_col_upper not in df.columns:
                continue

            # Check for duplicates
            if df[pk_col_upper].duplicated().any():
                col_meta = table_meta.get_column(pk_col)

                # Regenerate unique values
                if col_meta and col_meta.data_type in (DataType.INTEGER, DataType.BIGINT):
                    # Use sequential IDs
                    df[pk_col_upper] = range(1, len(df) + 1)
                else:
                    # Use UUIDs for strings
                    import uuid
                    df[pk_col_upper] = [str(uuid.uuid4()) for _ in range(len(df))]

        return df

    def get_generated_data(self) -> Dict[str, pd.DataFrame]:
        """Return all generated data."""
        return self._generated_data
