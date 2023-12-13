import os.path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql import functions as F
import sys


def shortest_paths_linear(edges):
    edges = edges.alias("edges").cache()
    paths = edges.alias("paths")

    while True:
        new_paths = paths.join(edges, col('paths.edge_2') == col('edges.edge_1'))\
            .select(col('paths.edge_1').alias('edge_1'), col('edges.edge_2').alias('edge_2'), (col('paths.length') + col('edges.length')).alias('length'))\
            .union(paths).groupBy("edge_1", "edge_2").agg(F.min("length").alias("length")).alias("new_paths").checkpoint()

        if paths.count() != new_paths.count():
            paths = new_paths.alias("paths")
        elif paths.join(new_paths, (col('paths.edge_1') == col('new_paths.edge_1')) & (col('paths.edge_2') == col('new_paths.edge_2')))\
                .filter('paths.length > new_paths.length')\
                .count() > 0:
            paths = new_paths.alias("paths")
        else:
            break

    return paths


def shortest_paths_doubling(edges):
    paths = edges.select("edge_1", "edge_2", "length").alias("paths")

    while True:
        new_paths = paths.alias("p1")\
            .join(paths.alias("p2"), col("p1.edge_2") == col("p2.edge_1"))\
            .select(col("p1.edge_1").alias("edge_1"), col("p2.edge_2").alias("edge_2"),
                    (col("p1.length") + col("p2.length")).alias("length"))\
            .union(paths)\
            .groupBy("edge_1", "edge_2")\
            .agg(F.min("length").alias("length"))\
            .alias("new_paths")\
            .checkpoint()

        if paths.count() != new_paths.count():
            paths = new_paths.alias("paths")
        elif paths\
                .join(new_paths, (col('paths.edge_1') == col('new_paths.edge_1')) & (col('paths.edge_2') == col('new_paths.edge_2'))) \
                .filter('paths.length > new_paths.length') \
                .count() > 0:
            paths = new_paths.alias("paths")
        else:
            break

    return paths


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {os.path.basename(__file__)} <linear|doubling> <input_file> <output_file>")
        sys.exit(1)

    algorithm = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    # .master("local[*]") \
    spark = SparkSession.builder \
        .master("spark://master:7077") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "1g") \
        .appName("ShortestPath") \
        .getOrCreate()

    #spark.sparkContext.setLogLevel("DEBUG")
    spark.sparkContext.setCheckpointDir('./tmp')

    schema = StructType([
        StructField("edge_1", IntegerType(), True),
        StructField("edge_2", IntegerType(), True),
        StructField("length", DoubleType(), True)
    ])

    edges = spark.read.csv(input_file, schema=schema, header=True)

    if algorithm == "linear":
        result = shortest_paths_linear(edges)
    elif algorithm == "doubling":
        result = shortest_paths_doubling(edges)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result.toPandas().to_csv(output_file, index=False)

    spark.stop()


if __name__ == "__main__":
    main()


