# Databricks notebook source
# MAGIC %md
# MAGIC ### Mount S3 on /FileStore/tables/

# COMMAND ----------

import urllib

aws_key = (
        spark.read.format("csv")
                  .option("header", "true")
                  .option("delimiter", ",")
                  .load('dbfs:/FileStore/tables/aws_keys_csv.csv')
)

ACCESS_KEY = aws_key.select('Access key ID').collect()[0][0]
SECRET_KEY = aws_key.select('Secret access key').collect()[0][0]
ENCODED_SECRET_KEY = urllib.parse.quote(SECRET_KEY, "")
AWS_S3_BUCKET = 'superlogica-cases'
MOUNT_NAME = '/mnt/superlogica-cases'
SOURCE_URL = 's3n://{0}:{1}@{2}'.format(ACCESS_KEY, ENCODED_SECRET_KEY,AWS_S3_BUCKET)
dbutils.fs.mount(SOURCE_URL, MOUNT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkTools

# COMMAND ----------

from __future__ import annotations

import re
import urllib
import logging
import unicodedata
from typing import Dict, List
from copy import deepcopy
from types import SimpleNamespace
from pyspark.sql.functions import col, udf, when, lit, current_date, max
from pyspark.sql.column import Column
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from delta.tables import *
from pyspark.sql import functions as F

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

            if str is not None:
                return "".join(
                    ch
                    for ch in unicodedata.normalize("NFKD", str)
                    if not unicodedata.combining(ch)
                )

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))
    
    def _get_col_values_list(self, df: DataFrame, col_name: str) -> List:
        
        _rows = df.select(col_name).distinct().collect()
        values = [str(_row[col_name]) for _row in _rows]
        values.sort(reverse=True)
        return values    

    def _remove_special_chars(self, column: str) -> udf:
        """Remove special characters from string"""

        def udf_function(str):
            if str is not None:
                str = re.sub(r"[^A-Za-z0-9 ]+", "", str)
                return re.sub(r"  +", " ", str)

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _upper_columns(self, column: str) -> udf:
        """Convert column to uppercase"""

        def udf_function(str):
            if str is not None:
                return str.upper()

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _convert_datatype(
        self, df: DataFrame, columns: col, datatype_1: str, datatype_2: str
    ) -> DataFrame:
        """Convert columns datatype_1 to datatype_2"""
        for column in df.schema:
            if column.name in columns:
                if datatype_1 in str(column.dataType):
                    df = df.withColumn(
                        column.name, col(column.name).cast(datatype_2)
                    )
        return df

    def _replace_comma(self, column: str) -> udf:
        """Convert comma to dot"""

        def udf_function(str):
            if str is not None and str != "":
                return str.replace(",", ".")

        udf_call = udf(udf_function)

        return udf_call(self._is_column(column))

    def _blank_as_null(self, column: str) -> None:
        """Convert blank to null"""
        column = self._is_column(column)
        return when(column == "", lit(None)).otherwise(column)

    def _optimized_repartition(
        self,
        spark: SparkSession,
        df: DataFrame,
        partitionBy: list = [],
        minRecordsPerFile: int = 2**17,
        ratio: float = 1,
    ) -> DataFrame:
        """Optimizes partitions"""
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
        df: DataFrame,
        path: str,
        format: str,
        mode: str,
        partitionBy: List = [],
        minRecordsPerFile: int = 2**17,
        ratio: float = 1,
        **kwargs,
    ) -> None:
        """Optimize write"""
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

    def _load_to_namespace(
        self, spark: SparkSession, tables: dict, options: dict
    ) -> SimpleNamespace:
        """Reading dataframes with namespace"""
        tables = deepcopy(tables)
        default_options = deepcopy(options)

        table_dict = dict()
        for name, value in tables.items():
            if isinstance(value, str):
                _path = value
                _options = deepcopy(default_options)
            elif isinstance(value, dict):
                _path = value.pop("path")
                _options = deepcopy(default_options)
                _options.update(value.pop("options", dict()))

            _format = _options.pop("format")

            if _format.lower() in ["parquet", "delta", "csv"]:
                _df = (
                    spark.read.format(_format.lower())
                    .options(**_options)
                    .load(_path)
                )
            else:
                raise NotImplementedError(
                    f'Input format "{_format}" not supported.'
                )
            table_dict[name.lower()] = _df
        return SimpleNamespace(**table_dict)

    def _delta_merge(
                    self,
                    spark: SparkSession,
                    raw: DataFrame,
                    _path,
                    _merge_date_col,
                    _key_cols) -> None:
        """Write dataframe with upsert"""
        if not DeltaTable.isDeltaTable(spark, _path):
            logger.info("Building first DeltaTable")
            self._optimized_write(
                spark=spark,
                df=raw,
                path=_path,
                format="delta",
                mode="overwrite",
            )
            raw.select(_merge_date_col).distinct().orderBy(
                _merge_date_col, ascending=False
            ).show(30, False)
            raw.unpersist()
            logger.info("First DeltaTable built successfully.")
        else:
            deltaTable = DeltaTable.forPath(spark, _path)
            merge_keys = [f"STAGING.{_col} = RAW.{_col}" for _col in _key_cols]
            _merge_query = ""
            for i, _key in enumerate(merge_keys):
                _merge_query += _key + (
                    " AND " if i + 1 < len(merge_keys) else " "
                )
            merge_query = _merge_query + (
                f"AND STAGING.{_merge_date_col} < RAW.{_merge_date_col}"
            )
            logger.info(f"Merge query: {merge_query}")

            delta_data = deltaTable.toDF()

            max_update_date = delta_data.agg(
                max(col(_merge_date_col))
            ).collect()[0][0]
            raw = raw.filter(col(_merge_date_col) > max_update_date)

            if raw.count() > 0:
                logger.info(f"Updating {raw.count()} new rows.")
                raw = raw.withColumn("data_extraction", current_date())

                raw = self._optimized_repartition(spark, raw)
                (
                    deltaTable.alias("STAGING")
                    .merge(raw.alias("RAW"), merge_query)
                    .whenMatchedUpdateAll()
                    .whenNotMatchedInsertAll()
                    .execute()
                )
                logger.info("DeltaTable merge complete.")

                deltaTable.toDF().select(_merge_date_col).distinct().orderBy(
                    _merge_date_col, ascending=False
                ).show(30, False)
                deltaTable.toDF().unpersist()
            else:
                logger.info("No new data found.")

    def _check_history(
        self,
        spark: SparkSession,
        _path: str,
        _columns: List = ["version", "timestamp", "userName", "operation"],
    ) -> None:
        """Checks the history of changes in the dataframe"""
        if isinstance(_path, DeltaTable):
            delta_table.history().select(_columns).show(truncate=False)
        else:
            delta_table = DeltaTable.forPath(spark, _path)
            delta_table.history().select(_columns).show(truncate=False)

    def _revert_version(
        self, spark: SparkSession, _path: str, _version: int
    ) -> None:
        """Revert changes to the dataframe"""
        delta_table = DeltaTable.forPath(spark, _path)
        delta_table.restoreToVersion(_version)

        self._check_history(spark, _path)

    def _execute_etl(self,
                     spark: SparkSession, 
                     dfs: SimpleNamespace, 
                     _path: str,
                     _merge_date_col: str,
                     _key_cols: List) -> None:
  
        date_columns = ['DT_CRIACAO_COND'
                       ,'DT_DESATIVACAO_COND'
                       ,'DT_ATUALIZACAO_CAD']
        raw = (
            self._convert_datatype(dfs, date_columns, 'StringType', 'date')
        )
        raw = (
            raw.withColumn('QTD_UNIDADES', self._replace_comma('QTD_UNIDADES'))
               .withColumn('ST_CNPJ_CONDOMINIO', self._remove_special_chars('ST_CNPJ_CONDOMINIO'))
               .withColumn('ST_CEP_CONDOMINIO', self._remove_special_chars('ST_CEP_CONDOMINIO'))         
        )
        raw = (
                raw.withColumn('ST_NOME_CONDOMINIO', self._normalize_str('ST_NOME_CONDOMINIO'))
                   .withColumn('ST_NOME_CONDOMINIO', self._remove_special_chars('ST_NOME_CONDOMINIO'))
                   .withColumn('ST_NOME_CONDOMINIO', self._upper_columns('ST_NOME_CONDOMINIO'))
        )
        raw = (
            self._convert_datatype(raw, ['QTD_UNIDADES'], 'StringType', 'integer')
        )
        
        self._delta_merge(spark, raw, _path, _merge_date_col, _key_cols)
                                
    def _execute_cep_etl(self, 
                         spark: SparkSession, 
                         _df: DataFrame, 
                         _path: str) -> None:
        _df = _df.select(
            *[col(column).alias("ST_" + column.upper()) for column in _df.columns]
        )
        
        for column in _df.columns:
            _df = _df.withColumn(column, self._normalize_str(col(column)))
            _df = _df.withColumn(column, self._upper_columns(col(column)))
            _df = _df.withColumn(column, when((col(column) == 'null'), None).otherwise(col(column)))
        
        _df = (
            _df.select(['ST_CEP', 'ST_STATE'])
               .withColumnRenamed('ST_CEP', 'ST_CEP_CONDOMINIO') 
               .withColumnRenamed('ST_STATE', 'UF')
        )
        self._optimized_write(spark, 
                              _df, 
                              _path, 
                              format='delta', 
                              mode='overwrite')


# COMMAND ----------

display(dbutils.fs.ls('dbfs:/mnt/superlogica-cases/'))

# COMMAND ----------

spt = SparkTools()

# COMMAND ----------

# Read dataframes
_namespace = {
    "tables": {
          '_16': 'dbfs:/mnt/superlogica-cases/raw/condominios/2023-06-16', 
          '_17': 'dbfs:/mnt/superlogica-cases/raw/condominios/2023-06-17',
          '_18': 'dbfs:/mnt/superlogica-cases/raw/condominios/2023-06-18'
    },
    "options": {"format": "csv", 
                "header": True, 
                "demlimiter": ","},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Criar métodos em Python ou em PySpark para realizar o tratamento dos dados com os objetivos de:
# MAGIC
# MAGIC * Ajustar os tipos de dados (converter os campos ID para string, os campos DT para datetime, e otimizar os campos numéricos);
# MAGIC * Tratar os dados das colunas CNPJ e CEP de maneira a manter apenas os números (Ex: XX.XXX.XXX/YYYY-ZZ para XXXXXXXXYYYYZZ);
# MAGIC * Padronizar as colunas de texto, removendo acentos e mantendo todas as letras em maíusculo ou minúsculo;
# MAGIC * Tratar os dados faltantes, duplicados e/ou inconsistentes.

# COMMAND ----------

# day 16 dataframes ETL steps 
date_columns = ['DT_CRIACAO_COND', 'DT_DESATIVACAO_COND', 'DT_ATUALIZACAO_CAD']
raw = spt._convert_datatype(dfs._16, date_columns, 'StringType', 'date')
raw.select(date_columns).printSchema()

# COMMAND ----------

raw = raw.withColumn('QTD_UNIDADES', spt._replace_comma('QTD_UNIDADES'))
raw = spt._convert_datatype(raw, ['QTD_UNIDADES'], 'StringType', 'integer')
raw.show(100)
raw.select('QTD_UNIDADES').printSchema()

# COMMAND ----------

raw = (
    raw.withColumn('ST_CNPJ_CONDOMINIO', spt._remove_special_chars('ST_CNPJ_CONDOMINIO'))
       .withColumn('ST_CEP_CONDOMINIO', spt._remove_special_chars('ST_CEP_CONDOMINIO'))    
)

raw.show(100)

# COMMAND ----------

raw = (
        raw.withColumn('ST_NOME_CONDOMINIO', spt._normalize_str('ST_NOME_CONDOMINIO'))
           .withColumn('ST_NOME_CONDOMINIO', spt._remove_special_chars('ST_NOME_CONDOMINIO'))
           .withColumn('ST_NOME_CONDOMINIO', spt._upper_columns('ST_NOME_CONDOMINIO'))
)
raw.show(100)


# COMMAND ----------

raw = raw.dropDuplicates(['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND'])
raw.show(100)

# COMMAND ----------

# write dataframe day 16
_path = 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios'
_key_cols = ['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND', 'ST_CNPJ_CONDOMINIO']
_merge_date_col = 'DT_ATUALIZACAO_CAD'

spt._delta_merge(spark, 
                 raw, 
                 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 
                 'DT_ATUALIZACAO_CAD', 
                 ['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND', 'ST_CNPJ_CONDOMINIO'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Realizar uma carga completa com as extrações do dia 16/06 e incrementar os dados com as extrações dos dias 17/06 e 18/06, adicionando os novos cadastros e atualizando as informações que foram alteradas.

# COMMAND ----------

# Execute full ETL for day 17
spt._execute_etl(spark, 
                 dfs._17,                 
                 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 
                 'DT_ATUALIZACAO_CAD', 
                 ['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND', 'ST_CNPJ_CONDOMINIO'])

# COMMAND ----------

# Execute full ETL for day 18
spt._execute_etl(spark, 
                 dfs._18,                  
                 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 
                 'DT_ATUALIZACAO_CAD', 
                 ['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND', 'ST_CNPJ_CONDOMINIO'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Exportar o dataset gerado 'Cadastral condominios' para um arquivo tipo Apache Parquet.

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/mnt/superlogica-cases/staging/cadastral_condominios'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Ler o arquivo Parquet e exibir o dataset para validação.

# COMMAND ----------

# Load new dataframe
_namespace = {
    "tables": {
          'staging': 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 

    },
    "options": {"format": "delta"},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)
dfs.staging.orderBy(col('DT_ATUALIZACAO_CAD').desc()).show(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Além de possibilitar a:
# MAGIC * Identificação das alterações realizadas ao longo do tempo;
# MAGIC * Reversão das alterações realizadas.

# COMMAND ----------

# Verify changes:
spt._check_history(spark, 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios')

# COMMAND ----------

# Revert changes
spt._revert_version(spark, 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 1)

# COMMAND ----------

# Load revert version
_namespace = {
    "tables": {
          'staging': 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 

    },
    "options": {"format": "delta"},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)
dfs.staging.orderBy(col('DT_ATUALIZACAO_CAD').desc()).show(100)

# COMMAND ----------

# Read dataframes
_namespace = {
    "tables": {
          '_18': 'dbfs:/mnt/superlogica-cases/raw/condominios/2023-06-18'
    },
    "options": {"format": "csv", 
                "header": True, 
                "demlimiter": ","},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)

# COMMAND ----------

# return day 18
spt._execute_etl(spark, 
                 dfs._18,                  
                 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 
                 'DT_ATUALIZACAO_CAD', 
                 ['ID_ADMINISTRADORA', 'ID_CONDOMINIO_COND', 'ST_CNPJ_CONDOMINIO'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### CepAPI

# COMMAND ----------

import aiohttp
import asyncio
import logging
import nest_asyncio
from tqdm.notebook import tqdm
from time import sleep
from typing import List

nest_asyncio.apply()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CepAPI:
    def __init__(self, base_url: str, limit: int):
        self.base_url = base_url
        self.limit = limit
        self.rate_limit = {"max_retrys": 5, "sleep_requests": 30}

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        self.data = []

    def _client_conn(self, limit: int) -> None:
        return aiohttp.TCPConnector(limit=self.limit, ttl_dns_cache=300)

    def _get_url(self, cep: str) -> str:
        if (
            cep is not None
            and cep != "99999999"
            and cep != "00000000"
            and cep != ""
        ):
            url = self.base_url % cep
            if url is not None:
                return url
        else:
            logger.error("Invalid zip code")

    def _create_url(self, cep_list: List) -> List:
        return [self._get_url(cep) for cep in cep_list]

    async def _requests(self, session, url: str, retry: str = 0) -> None:
        try:
            async with session.get(url, headers=self.headers) as resp:
                resp.raise_for_status()
                if resp.status == 200:
                    if (resp.status == 429) and (
                        retry < self.rate_limit["max_retrys"]
                    ):
                        logger.error(
                            f"{resp.status} Too many requests. "
                            + f"sleep for {self.rate_limit['sleep_requests']}s and retry."
                        )
                        sleep(self.rate_limit["sleep_requests"])
                        return await self._requests(
                            session, url, retry=retry + 1
                        )
                    response = await resp.json()
                    if response is not None:
                        self.data.append(response)
                    else:
                        logger.error(
                            "The get() function no return a response object"
                        )
                else:
                    logger.error("Status: {}".format(resp.status))
                    response = dict()
                    self.data.append(response)
                return self.data
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP client reply error: {e}")

    async def _get_requests(self, urls: List) -> None:
        timeout = aiohttp.ClientTimeout(total=8 * 60)
        conn = self._client_conn(self.limit)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, connector=conn
            ) as session:
                urls = tqdm(urls)
                for url in urls:
                    if url is not None:
                        response = await self._requests(session, url, retry=0)
                        await asyncio.sleep(0.05)
                    else:
                        logger.error("Invalid URL")
                await conn.close()
                return response
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout during request: {e}")

    def _extract_cep(self, cep_list: List):
        loop = asyncio.get_event_loop()
        urls = self._create_url(cep_list)
        _results = loop.run_until_complete(self._get_requests(urls))
        _df = spark.read.json(spark.sparkContext.parallelize(_results))
        return _df    


# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Determinar o estado (UF) do condomínio, via API, a partir do CEP e criar uma tabela que correlacione CEP e UF. Depois realizar o join em SQL com a base de dados obtida nos passos anteriores (base resultante dessa etapa: Condominios).

# COMMAND ----------

# Load revert version
_namespace = {
    "tables": {
          'staging': 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios', 

    },
    "options": {"format": "delta"},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)

# COMMAND ----------

cep_list = spt._get_col_values_list(dfs.staging, 'ST_CEP_CONDOMINIO')

# COMMAND ----------

cep_api = CepAPI('https://brasilapi.com.br/api/cep/v1/%s', 20)

# COMMAND ----------

ceps = cep_api._extract_cep(cep_list)

# COMMAND ----------

ceps.show()

# COMMAND ----------

spt._execute_cep_etl(spark, ceps, _path='dbfs:/mnt/superlogica-cases/staging/ceps_uf' )    

# COMMAND ----------

# Load CEP dataframe
_namespace = {
    "tables": {
          'cadastral_condominios': 'dbfs:/mnt/superlogica-cases/staging/cadastral_condominios',
          'ceps_uf': 'dbfs:/mnt/superlogica-cases/staging/ceps_uf', 

    },
    "options": {"format": "delta"},
}
dfs = spt._load_to_namespace(
    spark, **_namespace
)

# COMMAND ----------

dfs.ceps_uf.show()

# COMMAND ----------

condominios = (
    dfs.cadastral_condominios.join(dfs.ceps_uf, 
                                   on=['ST_CEP_CONDOMINIO'],
                                   how='left')
)
condominios.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Criar uma tabela em SQL com as métricas da base construída no passo anterior (Condominios) por UF (Metricas por UF).

# COMMAND ----------

metricas_uf = (
    condominios.groupBy('UF')
               .agg(F.count('ID_CONDOMINIO_COND').alias('TOTAL_CONDOMINIOS'),
                    F.countDistinct('ID_ADMINISTRADORA').alias('TOTAL_ADMINISTRADORAS'),
                    F.sum('QTD_UNIDADES').alias('TOTAL_UNIDADES')) 
)
metricas_uf.show()

# COMMAND ----------

dfs.cadastral_condominios.createOrReplaceTempView("CADASTRAL_CONDOMINIOS")
dfs.ceps_uf.createOrReplaceTempView("CEPS_UF")


spark.sql("""
            DROP TABLE IF EXISTS CONDOMINIOS
          """)

spark.sql(
    """
        CREATE TABLE CONDOMINIOS AS
        SELECT 
             C.ID_ADMINISTRADORA
            ,C.ID_CONDOMINIO_COND
            ,C.ST_CNPJ_CONDOMINIO
            ,C.ST_NOME_CONDOMINIO
            ,C.ST_CEP_CONDOMINIO
            ,C.DT_CRIACAO_COND
            ,C.DT_DESATIVACAO_COND
            ,C.QTD_UNIDADES
            ,C.DT_ATUALIZACAO_CAD
            ,U.UF

        FROM CADASTRAL_CONDOMINIOS C
        LEFT JOIN CEPS_UF U
        ON C.ST_CEP_CONDOMINIO = U.ST_CEP_CONDOMINIO
    """
)

spark.sql(
    """
        SELECT * 
        FROM CONDOMINIOS
    """).show()

# COMMAND ----------

spark.sql("""
          DROP TABLE IF EXISTS METRICAS_UF
          """)

spark.sql(
    """
        CREATE TABLE METRICAS_UF AS
            SELECT
                UF,
                COUNT(ID_CONDOMINIO_COND) AS TOTAL_CONDOMINIOS,
                COUNT(DISTINCT ID_ADMINISTRADORA) AS TOTAL_ADMINISTRADORAS,
                SUM(QTD_UNIDADES) AS TOTAL_UNIDADES
            FROM
                CONDOMINIOS
            GROUP BY
                UF;   
    """
)

spark.sql(
    """
        SELECT * 
        FROM METRICAS_UF
    """).show()

# COMMAND ----------

condominios.show(300)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GoogleSheetExport

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ⚙️ Salvar as bases 'Condominios' e 'Metricas por UF' em uma planilha do Google Sheets (uma aba para cada), via API, e compartilhar a planilha com os e-mails: felipe.gonzaga@superlogica.com, ricardo.monteiro@superlogica.com e tifane.carvalho@superlogica.com.

# COMMAND ----------

display(dbutils.fs.ls('/FileStore/tables/'))

# COMMAND ----------

import math
import datetime 
import pyspark.pandas as ps
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class GoogleSheetExport:
    def __init__(self,
                 spreadsheet_id: str,
                 sheet_name: str,
                 google_token: str,
                 scopes: List          
                 ):
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name
        self.google_token = google_token
        self.scopes = scopes     

    def _service(self):
        credentials = Credentials.from_authorized_user_file(self.google_token, self.scopes)
        service = build('sheets', 'v4', credentials=credentials)
        return service


    def _get_sheet_ids(self) -> None:
        spreadsheet = self._service()
        sheet_id = None
        for sheet in spreadsheet['sheets']:
            if sheet['properties']['title'] == sheet_name:
                break
            logger.info(f"{sheet['properties']['title']} ID: {sheet['properties']['sheetId']}")


    def _create_new_tab(self, new_tab_name: str) -> None:
        service = self._service()
        request_body = {
            'requests': [
                {
                    'addSheet': {
                        'properties': {
                            'title': new_tab_name
                        }
                    }
                }
            ]
        }    
        response = service.spreadsheets().batchUpdate(
            spreadsheetId=self.spreadsheet_id,
            body=request_body
        ).execute()


    def _rename_tab(self, tab_id: int, new_tab_name: str) -> None:
        service = self._service()
        request_body = {
            'requests': [
                {
                    'updateSheetProperties': {
                        'properties': {
                            'sheetId': tab_id,
                            'title': new_tab_name
                        },
                        'fields': 'title'
                    }
                }
            ]
        }
        response = service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id, 
                body=request_body
        ).execute()

    def _export_data(self, df: DataFrame, tab_name: str) -> None:

        service = self._service()
        pandas_df = ps.DataFrame(df)
        data_str = [
                        [
                            cell.strftime("%Y-%m-%d") 
                            if isinstance(cell, datetime.date) 
                                else cell 
                            if not (isinstance(cell, float) and math.isnan(cell)) 
                                else None 
                        
                            for cell in row] 
            
                    for row in [pandas_df.columns.tolist()] + pandas_df.to_numpy().tolist()]
        
        service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=tab_name,
                valueInputOption='RAW',
                body={'values': data_str}
        ).execute()



# COMMAND ----------

gs = GoogleSheetExport(
    spreadsheet_id='1fLB4G_EElC_DwxF3P19rHaYgfWsYwIElA7_5Cf93xZA',
    sheet_name = 'Superlogica!1:1000',
    google_token = '/dbfs/FileStore/tables/token.json',
    scopes = ['https://www.googleapis.com/auth/spreadsheets'],
)

# COMMAND ----------

gs._rename_tab('1442465606', 'Condomínios')

# COMMAND ----------

gs._export_data(condominios, 'Condomínios')

# COMMAND ----------

gs._create_new_tab('Métricas por UF')

# COMMAND ----------

gs._export_data(metricas_uf, 'Métricas por UF')
