#pyspark
import argparse

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import array_contains
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import lit

def random_text_classifier(input_loc, output_loc):
    schema = StructType([
    StructField('cid', StringType(), True),   # Column name with String type
    StructField('review_str', StringType(), True),  # Column age with Integer type
    ])
    df_raw = spark.read.option('header', 'true') \
                   .schema(schema) \
                   .csv(input_loc)
    
    #df_raw = df_raw.withColumn("review_to_ken",lit(None))
    tokenizer = Tokenizer(inputCol = 'review_str', outputCol = 'review_token')
    df_tokens = tokenizer.transform(df_raw).select('cid', 'review_token')

    remove = StopWordsRemover(inputCol = 'review_token', outputCol = 'review_clean')
    df_clean = remove.transform(df_tokens).select('cid', 'review_clean')

    df_out = df_clean.select('cid', array_contains(df_clean.review_clean, "good").alias('positive_review'))
    df_out.write.mode("overwrite").parquet(output_loc)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='HDFS input', default='/movie')
    parser.add_argument('--output', type=str, help='HDFS output', default='/output')
    args = parser.parse_args()
    spark = SparkSession.builder.appName('Random text classifier').getOrCreate()
    random_text_classifier(input_loc=args.input, output_loc=args.output)