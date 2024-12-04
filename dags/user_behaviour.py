from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.operators import PythonOperator
import os

unload_user_purchase = './scripts/sql/filter_unload_user_purchase.sql'
temp_filtered_user_purchase = '/temp/temp_filtered_user_purchase.csv'

movie_review_local = '/data/movie_review/movie_review.csv'
movie_review_load = 'movie_review/load/movie.csv'

BUCKET_NAME = 'customer-behaviour'
temp_filtered_user_purchase_key = 'user_purchase/stage/{{ds}}/temp_filtered_user_purchase.csv'

def _local_to_s3(filename, key, bucket_name = BUCKET_NAME) :
    s3 = S3Hook('s3_conn')
    s3.load_file(filename = filename, bucket_name = bucket_name, replace = True, key = key)

def remove_local_file(filelocation):
    if os.path.isfile(filelocation):
        os.remove(filelocation)
    else:
        logging.info(f'File {filelocation} not found')

default_args = {
    "owner" : "airflow",
    "depends_on_past" : True,
    "wait_for_downstream" : True,
    "start_date" : datetime(2010,12,1),
    "email" : ["airflow@airflow.com"],
    "email_on_failure" : False,
    "email_on_retry" : False,
    "retries" : 1,
    "retry_delay" : timedelta(minutes=5)
}

dag = DAG("user_behaviour", default_args=default_args, schedule_interval="0 0 * * *", max_active_runs=1)

end_of_data_pipeline = DummyOperator(task_id='end_of_data_pipeline', dag=dag)

#loads postgres data to a local location in the computer filtered by date
pg_unload = PostgresOperator(
    dag  = dag,
    task_id = 'pg_unload',
    sql = unload_user_purchase,
    postgres_conn_id = 'postgres_default',
    params = {'temp_filtered_user_purchase': temp_filtered_user_purchase},
    depends_on_past = True,
    wait_for_downstream = True
)

#Uploads the extracted CSV to S3
user_purchase_to_s3_stage = PythonOperator(
    dag = dag,
    task_id = 'user_purchase_to_s3_stage',
    python_callable = _local_to_s3,
    op_kwargs = {
        'filename' : temp_filtered_user_purchase,
        'key' : temp_filtered_user_purchase_key
    }
)

#removes the file extracted from PostgreSQL
remove_local_user_purchase_file = PythonOperator(
    dag = dag,
    task_id = 'remove_local_user_purchase_file',
    python_callable = remove_local_file,
    op_kwargs = {
        'filelocation' : temp_filtered_user_purchase
    }
)

#upload the local file to AWS S3(dummy data source dropped into an SFTP server)
movie_review_to_s3_stage = PythonOperator(
    dag = dag,
    task_id = 'movie_review_to_s3_stage',
    python_callable = _local_to_s3,
    op_kwargs = {
        'filename' : movie_review_local,
        'key' : movie_review_load
    }
)

#PostgreSQL to Redshift
pg_unload>>user_purchase_to_s3_stage>>remove_local_user_purchase_file>>end_of_data_pipeline

#CSV dropped by client to Redshift
movie_review_to_s3_stage