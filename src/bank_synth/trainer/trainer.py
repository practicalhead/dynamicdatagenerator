"""
Trainer component for learning data distributions from sample extracts.

Uses SDV (Synthetic Data Vault) under the hood for relational modeling,
with custom handling for banking-specific patterns and privacy constraints.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from bank_synth.models import (
    ColumnMetadata,
    ColumnStats,
    DataType,
    ModelPack,
    PrivacyLevel,
    RelationshipGraph,
    TableMetadata,
    TableStats,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trains synthetic data models from sample extracts.

    The trainer:
    1. Reads sample data for each table
    2. Computes statistics and distributions
    3. Trains SDV synthesizers per table
    4. Captures cross-table correlations via relationship graph
    5. Outputs a portable ModelPack artifact
    """

    def __init__(
        self,
        relationship_graph: RelationshipGraph,
        privacy_policy: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize trainer.

        Args:
            relationship_graph: Graph with table metadata and relationships
            privacy_policy: Per-table column privacy settings
        """
        self.graph = relationship_graph
        self.privacy_policy = privacy_policy or {}
        self.model_pack = ModelPack(
            relationship_graph=relationship_graph,
            privacy_policy=privacy_policy,
        )

        # Training state
        self._sample_data: Dict[str, pd.DataFrame] = {}
        self._synthesizers: Dict[str, Any] = {}

    def load_samples(
        self,
        sample_dir: Path,
        tables: Optional[List[str]] = None,
    ) -> None:
        """
        Load sample data files from directory.

        Args:
            sample_dir: Directory containing sample Parquet/CSV files
            tables: Optional list of specific tables to load
        """
        sample_dir = Path(sample_dir)
        logger.info(f"Loading samples from {sample_dir}")

        # Find all sample files
        parquet_files = {f.stem.upper(): f for f in sample_dir.glob("**/*.parquet")}
        csv_files = {f.stem.upper(): f for f in sample_dir.glob("**/*.csv")}

        all_files = {**csv_files, **parquet_files}  # Parquet takes precedence

        tables_to_load = tables or list(all_files.keys())

        for table_name in tables_to_load:
            table_upper = table_name.upper()
            if table_upper not in all_files:
                logger.warning(f"No sample file found for table: {table_name}")
                continue

            file_path = all_files[table_upper]
            logger.info(f"Loading sample for {table_name} from {file_path}")

            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            # Normalize column names to uppercase
            df.columns = [c.upper() for c in df.columns]

            self._sample_data[table_upper] = df

        logger.info(f"Loaded samples for {len(self._sample_data)} tables")

    def load_sample_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Load sample data directly from DataFrame.

        Args:
            table_name: Table name
            df: DataFrame with sample data
        """
        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        self._sample_data[table_name.upper()] = df

    def train(
        self,
        schema_name: Optional[str] = None,
        sample_strategy: str = "percent:1",
    ) -> ModelPack:
        """
        Train models on loaded sample data.

        Args:
            schema_name: Schema name to record in model pack
            sample_strategy: Strategy used to create samples (for reference)

        Returns:
            Trained ModelPack ready for generation
        """
        logger.info(f"Training on {len(self._sample_data)} tables")

        self.model_pack.schema_name = schema_name
        self.model_pack.sample_strategy = sample_strategy
        self.model_pack.tables_trained = list(self._sample_data.keys())

        # Process tables in topological order for consistency
        _, sorted_tables = self.graph.get_dependency_closure(
            list(self._sample_data.keys())[0],
            include_children=True,
        )

        # Filter to only tables with samples
        tables_to_train = [
            t for t in sorted_tables
            if t.upper() in self._sample_data
        ]

        # Add any remaining tables not in the dependency graph
        for t in self._sample_data.keys():
            if t.lower() not in [x.lower() for x in tables_to_train]:
                tables_to_train.append(t)

        for table_name in tqdm(tables_to_train, desc="Training tables"):
            table_upper = table_name.upper()
            df = self._sample_data.get(table_upper)
            if df is None:
                continue

            table_meta = self.graph.get_table(table_name)

            # Compute statistics
            stats = self._compute_statistics(table_upper, df, table_meta)
            self.model_pack.table_stats[table_upper] = stats

            # Apply privacy transformations
            df_processed = self._apply_privacy(table_upper, df, table_meta)

            # Train synthesizer
            synthesizer = self._train_synthesizer(table_upper, df_processed, table_meta)
            if synthesizer:
                # Serialize synthesizer
                self.model_pack.synthesizers[table_upper] = pickle.dumps(synthesizer)

            # Store encoders for categorical columns
            encoders = self._build_encoders(table_upper, df_processed, table_meta)
            if encoders:
                self.model_pack.encoders[table_upper] = encoders

        logger.info("Training complete")
        return self.model_pack

    def _compute_statistics(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> TableStats:
        """Compute statistical summary for a table."""
        stats = TableStats(
            name=table_name,
            row_count=len(df),
        )

        for col_name in df.columns:
            col_series = df[col_name]
            col_meta = table_meta.get_column(col_name) if table_meta else None
            data_type = col_meta.data_type if col_meta else self._infer_type(col_series)

            col_stats = ColumnStats(
                name=col_name,
                data_type=data_type,
                null_fraction=col_series.isna().mean(),
                distinct_count=col_series.nunique(),
                total_count=len(col_series),
            )

            # Numeric stats
            if data_type in (DataType.INTEGER, DataType.BIGINT, DataType.DECIMAL,
                            DataType.FLOAT, DataType.DOUBLE):
                numeric = pd.to_numeric(col_series, errors="coerce")
                col_stats.min_value = float(numeric.min()) if not numeric.isna().all() else None
                col_stats.max_value = float(numeric.max()) if not numeric.isna().all() else None
                col_stats.mean = float(numeric.mean()) if not numeric.isna().all() else None
                col_stats.std = float(numeric.std()) if not numeric.isna().all() else None

            # Date/timestamp stats
            elif data_type in (DataType.DATE, DataType.TIMESTAMP):
                try:
                    dates = pd.to_datetime(col_series, errors="coerce")
                    col_stats.min_value = str(dates.min()) if not dates.isna().all() else None
                    col_stats.max_value = str(dates.max()) if not dates.isna().all() else None
                except Exception:
                    pass

            # Categorical / low cardinality stats
            if col_stats.distinct_count <= 100:
                col_stats.value_frequencies = col_series.value_counts().head(100).to_dict()

            # Check uniqueness
            col_stats.is_unique = col_stats.distinct_count == col_stats.total_count

            stats.column_stats[col_name] = col_stats

        return stats

    def _apply_privacy(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> pd.DataFrame:
        """Apply privacy transformations before training."""
        df = df.copy()

        # Get table privacy policy
        table_policy = self.privacy_policy.get(table_name.lower(), {})

        for col_name in df.columns:
            col_meta = table_meta.get_column(col_name) if table_meta else None
            privacy_level = PrivacyLevel.PUBLIC

            # Check column metadata
            if col_meta and col_meta.privacy_level:
                privacy_level = col_meta.privacy_level

            # Check privacy policy override
            col_policy = table_policy.get(col_name.lower(), {})
            if isinstance(col_policy, str):
                privacy_level = PrivacyLevel(col_policy.lower())
            elif isinstance(col_policy, dict) and "level" in col_policy:
                privacy_level = PrivacyLevel(col_policy["level"].lower())

            # Apply transformation
            if privacy_level == PrivacyLevel.RESTRICTED:
                # Drop column entirely
                df = df.drop(columns=[col_name])
                logger.info(f"Dropped restricted column: {table_name}.{col_name}")

            elif privacy_level == PrivacyLevel.PII:
                # Replace with null for training
                df[col_name] = None
                logger.info(f"Nullified PII column: {table_name}.{col_name}")

            elif privacy_level == PrivacyLevel.SENSITIVE:
                # Keep format pattern only (e.g., SSN becomes XXX-XX-XXXX pattern)
                if df[col_name].dtype == object:
                    df[col_name] = df[col_name].apply(
                        lambda x: self._mask_sensitive(x) if pd.notna(x) else x
                    )

        return df

    def _mask_sensitive(self, value: Any) -> str:
        """Mask sensitive value while preserving format."""
        if not isinstance(value, str):
            value = str(value)

        # Preserve format with X placeholders
        result = []
        for c in value:
            if c.isalpha():
                result.append("X")
            elif c.isdigit():
                result.append("9")
            else:
                result.append(c)
        return "".join(result)

    def _train_synthesizer(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> Optional[Any]:
        """Train SDV synthesizer for a table."""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata

            if len(df) == 0:
                logger.warning(f"Empty DataFrame for {table_name}, skipping synthesizer")
                return None

            # Build SDV metadata
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(df)

            # Customize metadata based on our column metadata
            if table_meta:
                for col_meta in table_meta.columns:
                    col_name = col_meta.name.upper()
                    if col_name not in df.columns:
                        continue

                    # Set primary key
                    if col_meta.is_primary_key:
                        try:
                            sdv_metadata.set_primary_key(col_name)
                        except Exception:
                            pass  # May already be set

                    # Set column types
                    sdv_type = self._map_to_sdv_type(col_meta.data_type)
                    if sdv_type:
                        try:
                            sdv_metadata.update_column(col_name, sdtype=sdv_type)
                        except Exception:
                            pass

            # Create and train synthesizer
            synthesizer = GaussianCopulaSynthesizer(sdv_metadata)
            synthesizer.fit(df)

            logger.info(f"Trained synthesizer for {table_name}")
            return synthesizer

        except ImportError:
            logger.warning("SDV not available, using fallback training")
            return self._train_fallback_synthesizer(table_name, df, table_meta)
        except Exception as e:
            logger.error(f"Error training synthesizer for {table_name}: {e}")
            return self._train_fallback_synthesizer(table_name, df, table_meta)

    def _train_fallback_synthesizer(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> Dict[str, Any]:
        """
        Fallback synthesizer when SDV is not available.

        Stores distributions and statistics for simple generation.
        """
        fallback = {
            "_type": "fallback",
            "columns": {},
            "row_count": len(df),
        }

        for col_name in df.columns:
            col_series = df[col_name]
            col_info = {
                "dtype": str(col_series.dtype),
                "null_fraction": float(col_series.isna().mean()),
            }

            # Store value distribution for categorical
            if col_series.nunique() <= 100:
                col_info["value_counts"] = col_series.value_counts(normalize=True).to_dict()
            else:
                # Store numeric distribution
                if np.issubdtype(col_series.dtype, np.number):
                    col_info["mean"] = float(col_series.mean())
                    col_info["std"] = float(col_series.std())
                    col_info["min"] = float(col_series.min())
                    col_info["max"] = float(col_series.max())
                else:
                    # Sample values for string columns
                    col_info["sample_values"] = col_series.dropna().head(1000).tolist()

            fallback["columns"][col_name] = col_info

        return fallback

    def _build_encoders(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_meta: Optional[TableMetadata],
    ) -> Dict[str, Any]:
        """Build encoders for categorical columns."""
        encoders = {}

        for col_name in df.columns:
            col_series = df[col_name]

            # Only for low-cardinality string columns
            if col_series.dtype == object and col_series.nunique() <= 100:
                unique_values = col_series.dropna().unique().tolist()
                encoders[col_name] = {
                    "type": "categorical",
                    "values": unique_values,
                }

        return encoders

    def _infer_type(self, series: pd.Series) -> DataType:
        """Infer DataType from pandas Series."""
        dtype = str(series.dtype).lower()

        if "int64" in dtype:
            return DataType.BIGINT
        elif "int" in dtype:
            return DataType.INTEGER
        elif "float" in dtype:
            return DataType.DOUBLE
        elif "datetime" in dtype:
            return DataType.TIMESTAMP
        elif "bool" in dtype:
            return DataType.BOOLEAN
        else:
            return DataType.STRING

    def _map_to_sdv_type(self, data_type: DataType) -> Optional[str]:
        """Map our DataType to SDV sdtype."""
        mapping = {
            DataType.STRING: "categorical",
            DataType.INTEGER: "numerical",
            DataType.BIGINT: "numerical",
            DataType.DECIMAL: "numerical",
            DataType.FLOAT: "numerical",
            DataType.DOUBLE: "numerical",
            DataType.DATE: "datetime",
            DataType.TIMESTAMP: "datetime",
            DataType.BOOLEAN: "boolean",
        }
        return mapping.get(data_type)

    def save(self, output_path: Path) -> None:
        """Save the trained model pack to disk."""
        self.model_pack.save(output_path)
        logger.info(f"Model pack saved to {output_path}")
