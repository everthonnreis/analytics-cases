import re
import logging
import unicodedata
from pyspark.sql.functions import col, udf
from pyspark.sql.column import Column
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparkTools:
    def _is_column(self, column: Dict[str, Column]) -> col:
        """Check if column is a string or pyspark column"""
        if isinstance(column, Column):
            return column
        elif isinstance(column, str):
            return col(column)
        else:
            raise TypeError(
                "'column' must be a string or "
                + f"pyspark column, not {type(column)}"
            )

    def _normalize_str(self, column: str) -> udf:
        """Normalize string"""

        def udf_function(str):
            return "".join(
                ch
                for ch in unicodedata.normalize("NFKD", str)
                if not unicodedata.combining(ch)
            )

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _remove_special_chars(self, column: str) -> udf:
        """Remove special characters from string"""

        def udf_function(str):
            str = re.sub(r"[^A-Za-z0-9 ]+", "", str)
            return re.sub(r"  +", " ", str)

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _upper_columns(self, column: str) -> udf:
        """Convert column to uppercase"""

        def udf_function(str):
            return str.upper()

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _convert_datatype(
        self, dataframe, columns: col, datatype_1: str, datatype_2: str) -> None:
        """Convert columns datatype_1 to datatype_2"""
        for column in dataframe.schema:
            if column.name in columns:
                if datatype_1 in str(column.dataType):
                    dataframe = dataframe.withColumn(
                        column.name, col(column.name).cast(datatype_2)
                    )
        return dataframe

    def _replace_comma(self, column: str) -> udf:
        """Convert comma to dot"""

        def udf_function(str):
            if str is not None and str != "":
                return str.replace(",", ".")

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _blank_as_null(self, column: str) -> None:
        """Convert blank to null"""
        from pyspark.sql.functions import when

        column = self._is_column(column)
        return when(column != "", column).otherwise(None)

    def _optimized_repartition(
        self,
        spark,
        df,
        partitionBy: list = [],
        minRecordsPerFile: int = 2**17,
        ratio: float = 1,
    ) -> None:
        from math import ceil

        try:
            _rows = df.count()
        except Exception:
            _rows = 0

        if _rows == 0:
            logger.warning("DataFrame is empty. No repartitioning needed")
            return df.repartition(1)

        _rdd_partitions = df.rdd.getNumPartitions()
        logger.info(f"Current number of partitions: {_rdd_partitions}")
        _parallelism = int(spark.sparkContext.defaultParallelism)
        logger.info("Spark default parallelism: " + str(_parallelism))

        _spark_perf = ceil(((_rdd_partitions + _parallelism) / 2) * ratio)
        if _rows:
            logger.info("DataFrame row count: " + str(_rows))
            _lake_perf = ceil(_rows / minRecordsPerFile)
        else:
            _lake_perf = None
        if (_lake_perf is None) or (_lake_perf > _spark_perf):
            _partitions = _spark_perf
        else:
            _partitions = ceil((_spark_perf + _lake_perf) / 2)
        logger.info(f"Optimized partitions: {_partitions}")

        if partitionBy:
            partitionBy = (
                [partitionBy] if isinstance(partitionBy, str) else partitionBy
            )
            df = df.repartition(_partitions, *partitionBy)
        elif _partitions == _rdd_partitions:
            logger.info("No repartitioning needed")
        elif _rdd_partitions > _partitions:
            df = df.coalesce(_partitions)
        else:
            logger.info("rdd already has less partitions than optimal")
        return df

    def _optimized_write(
        self,
        spark: SparkSession,
        df,
        path: str,
        format: str,
        mode: str,
        partitionBy: List = [],
        minRecordsPerFile: int = 2**17,
        ratio: float = 1,
        **kwargs,
    ) -> None:
        from time import perf_counter

        t0 = perf_counter()
        if isinstance(partitionBy, str):
            partitionBy = [partitionBy]

        logger.info(
            f"Writing DataFrame with mode '{mode}' and format '{format}' "
            + f"to path '{path}'"
        )
        _df = self._optimized_repartition(
            spark=spark, df=df, partitionBy=partitionBy, ratio=ratio
        )
        t1 = perf_counter()
        _df.write.format(format).options(**kwargs).mode(mode).partitionBy(
            partitionBy
        ).save(path)

        _df.unpersist()
        t2 = perf_counter()
        logger.info(
            f"DataFrame written, {str(int(t2-t0))}s elapsed "
            + f"(stage {str(int(t1-t0))}s + write {str(int(t2-t1))}s)."
        )

    def _load(
        self, spark: SparkSession, _format: str, _path: str, **kwargs
    ) -> None:
        if _format.lower() in ["parquet", "delta", "csv"]:
            _df = (
                spark.read.format(_format.lower()).options(**kwargs).load(_path)
            )
            return _df
        else:
            raise NotImplementedError(
                f'Input format "{_format}" not supported.'
            )
