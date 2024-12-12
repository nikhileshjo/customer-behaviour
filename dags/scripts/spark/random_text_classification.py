#pyspark
import argparse

from pyspark.sql import SparkSession
from pyspark.ml import Tokenizer, StopWordsRemover
from pyspark.sql.functions import array_contains

def random_text_classifier(input_loc, output_loc):
    df_raw = spark.read.option("header", True).csv(input_loc)

    tokenizer = Tokenizer(inputCol = 'review_str', outputCol = 'review_to_ken')
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