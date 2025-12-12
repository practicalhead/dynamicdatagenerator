"""
Spark Session Manager for distributed data generation.

Manages SparkSession lifecycle and configuration for:
- Local mode (development/testing)
- YARN/Kubernetes cluster mode (production)
- Hive metastore integration
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global session instance
_spark_session = None


class SparkSessionManager:
    """
    Manages SparkSession configuration and lifecycle.

    Features:
    - Automatic cluster detection (local vs YARN vs Kubernetes)
    - Hive metastore integration
    - Memory and parallelism configuration
    - ORC and Parquet codec configuration
    """

    # Default configurations for different cluster modes
    DEFAULT_CONFIGS = {
        "local": {
            "spark.master": "local[*]",
            "spark.driver.memory": "4g",
            "spark.executor.memory": "4g",
            "spark.sql.shuffle.partitions": "200",
            "spark.default.parallelism": "200",
        },
        "yarn": {
            "spark.master": "yarn",
            "spark.submit.deployMode": "client",
            "spark.driver.memory": "8g",
            "spark.executor.memory": "8g",
            "spark.executor.instances": "10",
            "spark.executor.cores": "4",
            "spark.sql.shuffle.partitions": "1000",
            "spark.default.parallelism": "1000",
        },
        "kubernetes": {
            "spark.master": "k8s://",
            "spark.driver.memory": "8g",
            "spark.executor.memory": "8g",
            "spark.executor.instances": "10",
            "spark.executor.cores": "4",
            "spark.sql.shuffle.partitions": "1000",
            "spark.default.parallelism": "1000",
        },
    }

    # ORC and Parquet optimizations
    FILE_FORMAT_CONFIGS = {
        "spark.sql.orc.impl": "native",
        "spark.sql.orc.enableVectorizedReader": "true",
        "spark.sql.orc.filterPushdown": "true",
        "spark.sql.parquet.compression.codec": "snappy",
        "spark.sql.orc.compression.codec": "snappy",
    }

    # Memory and performance optimizations
    PERFORMANCE_CONFIGS = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.broadcastTimeout": "600",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    }

    def __init__(
        self,
        app_name: str = "BankSynth_DataGenerator",
        mode: str = "auto",
        hive_enabled: bool = True,
        warehouse_dir: Optional[str] = None,
        custom_configs: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize Spark session manager.

        Args:
            app_name: Spark application name
            mode: Cluster mode ("local", "yarn", "kubernetes", "auto")
            hive_enabled: Enable Hive metastore support
            warehouse_dir: Hive warehouse directory path
            custom_configs: Additional Spark configurations
        """
        self.app_name = app_name
        self.mode = mode if mode != "auto" else self._detect_mode()
        self.hive_enabled = hive_enabled
        self.warehouse_dir = warehouse_dir
        self.custom_configs = custom_configs or {}
        self._session = None

    def _detect_mode(self) -> str:
        """Auto-detect the appropriate Spark mode."""
        # Check for YARN
        if os.environ.get("HADOOP_CONF_DIR") or os.environ.get("YARN_CONF_DIR"):
            logger.info("Detected YARN environment")
            return "yarn"

        # Check for Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            logger.info("Detected Kubernetes environment")
            return "kubernetes"

        # Default to local
        logger.info("Using local mode")
        return "local"

    def get_session(self):
        """
        Get or create SparkSession.

        Returns:
            SparkSession instance
        """
        if self._session is not None:
            return self._session

        from pyspark.sql import SparkSession

        # Build configuration
        builder = SparkSession.builder.appName(self.app_name)

        # Apply mode-specific configs
        mode_configs = self.DEFAULT_CONFIGS.get(self.mode, self.DEFAULT_CONFIGS["local"])
        for key, value in mode_configs.items():
            builder = builder.config(key, value)

        # Apply file format optimizations
        for key, value in self.FILE_FORMAT_CONFIGS.items():
            builder = builder.config(key, value)

        # Apply performance optimizations
        for key, value in self.PERFORMANCE_CONFIGS.items():
            builder = builder.config(key, value)

        # Apply custom configs
        for key, value in self.custom_configs.items():
            builder = builder.config(key, value)

        # Hive support
        if self.hive_enabled:
            builder = builder.enableHiveSupport()
            if self.warehouse_dir:
                builder = builder.config("spark.sql.warehouse.dir", self.warehouse_dir)

        # Create session
        self._session = builder.getOrCreate()

        # Log configuration
        logger.info(f"SparkSession created: {self.app_name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Parallelism: {self._session.sparkContext.defaultParallelism}")

        return self._session

    def stop(self):
        """Stop the SparkSession."""
        if self._session is not None:
            self._session.stop()
            self._session = None
            logger.info("SparkSession stopped")


def get_spark_session(
    app_name: str = "BankSynth_DataGenerator",
    mode: str = "auto",
    hive_enabled: bool = True,
    **kwargs,
):
    """
    Get or create a global SparkSession.

    Convenience function for getting a SparkSession with sensible defaults.

    Args:
        app_name: Spark application name
        mode: Cluster mode ("local", "yarn", "kubernetes", "auto")
        hive_enabled: Enable Hive metastore support
        **kwargs: Additional configs passed to SparkSessionManager

    Returns:
        SparkSession instance

    Example:
        spark = get_spark_session()
        df = spark.read.parquet("/path/to/data")
    """
    global _spark_session

    if _spark_session is None:
        manager = SparkSessionManager(
            app_name=app_name,
            mode=mode,
            hive_enabled=hive_enabled,
            **kwargs,
        )
        _spark_session = manager.get_session()

    return _spark_session


def configure_for_large_scale(spark, target_rows: int):
    """
    Configure Spark for large-scale data generation.

    Automatically adjusts parallelism and memory based on target row count.

    Args:
        spark: SparkSession instance
        target_rows: Target number of rows to generate
    """
    # Calculate optimal partition count
    # Aim for ~1M rows per partition for efficiency
    rows_per_partition = 1_000_000
    optimal_partitions = max(200, target_rows // rows_per_partition)

    # Cap at reasonable maximum
    optimal_partitions = min(optimal_partitions, 10000)

    spark.conf.set("spark.sql.shuffle.partitions", str(optimal_partitions))
    spark.conf.set("spark.default.parallelism", str(optimal_partitions))

    logger.info(f"Configured for {target_rows:,} rows with {optimal_partitions} partitions")
