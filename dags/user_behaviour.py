from datetime import datetime, timedelta
import json
import os

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.operators import PythonOperator
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor
from airflow.models import taskinstance

#imports for Redshift
from airflow.hooks.postgres_hook import PostgresHook
import psycopg2

unload_user_purchase = './scripts/sql/filter_unload_user_purchase.sql'
temp_filtered_user_purchase = '/temp/temp_filtered_user_purchase.csv'

movie_review_local = '/data/movie_review/movie_review.csv'
movie_review_load = 'movie_review/load/movie.csv'

BUCKET_NAME = 'customer-behaviour'
temp_filtered_user_purchase_key = 'user_purchase/stage/{{ds}}/temp_filtered_user_purchase.csv'

movie_clean_emr_steps = './dags/scripts/sql/emr/clean_movie_review.json'
movie_text_classification_script = './dags/scripts/spark/random_text_classification.py'

EMR_ID = 'j-3R6VF9G727AQI'
movie_review_load_folder = 'movie_review/load/'
movie_review_stage = 'movie_review/stage/'
text_classifier_script = 'scripts/random_text_classifier.py'

get_user_behaviour = 'scripts/sql/get_user_behavior_metrics.sql'

def _local_to_s3(filename, key, bucket_name = BUCKET_NAME) :
    s3 = S3Hook('s3_conn')
    s3.load_file(filename = filename, bucket_name = bucket_name, replace = True, key = key)

def remove_local_file(filelocation):
    if os.path.isfile(filelocation):
        os.remove(filelocation)
    else:
        logging.info(f'File {filelocation} not found')

def run_redshift_external_query(qry):
    rs_hook = PostgresHook(postgres_conn_id='redshift')
    rs_conn = rs_hook.get_conn()
    rs_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    rs_cursor = rs_conn.cursor()
    rs_cursor.execute(qry)
    rs_cursor.close()
    rs_conn.commit()

def fail_on_zero(**kwargs):
    ti = kwargs['ti']
    count = ti.xcom_pull(task_ids='get_count')
    if count == 0:
        raise ValueError("Count is 0. Failing the task.")

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

#copy EMR script to S3
emr_script_to_s3 = PythonOperator(
    dag = dag,
    task_id = 'emr_script_to_s3',
    python_callable = _local_to_s3,
    op_kwargs = {
        'filename' : movie_text_classification_script,
        'key': 'scripts/random_text_classification.py'
    }
)

with open(movie_clean_emr_steps) as json_file:
    emr_steps = json.load(json_file)

add_emr_steps = EmrAddStepsOperator(
    dag = dag,
    task_id = 'add_emr_steps',
    job_flow_id = EMR_ID,
    aws_conn_id = 'aws_default',
    steps = emr_steps,
    params = {
        'BUCKET_NAME' : BUCKET_NAME,
        'movie_review_load' : movie_review_load_folder,
        'text_classifier_script' : text_classifier_script,
        'movie_review_stage' : movie_review_stage
    },
    depends_on_past = True
)

last_step = len(emr_steps) - 1

clean_movie_review_data = EmrStepSensor(
    dag=dag,
    task_id='clean_movie_review_data',
    job_flow_id=EMR_ID,
    step_id='{{ task_instance.xcom_pull("add_emr_steps", key="return_value")[' + str(
        last_step) + '] }}',
    depends_on_past=True
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


# insert data in Redshift
user_purchase_to_rs_stage = PythonOperator(
    dag = dag,
    task_id = 'user_purchase_to_rs_stage',
    python_callable = run_redshift_external_query,
    op_kwargs = {
        'qry' : "alter table spectrum.user_purchase_staging add partition(insert_date='{{ ds }}') \
            location 's3://customer-behaviour/user_purchase/stage/{{ ds }}'"
    },
)

get_user_behaviour = PostgresOperator(
    dag = dag,
    task_id = 'get_user_behaviour',
    sql = get_user_behaviour,
    postgres_conn_id = 'redshift'
)

get_count = PostgresOperator(
        dag = dag,
        task_id='get_count',
        postgres_conn_id='redshift',
        sql="SELECT COUNT(*) FROM public.user_behavior_metric WHERE insert_date = '{{ ds }}';",
        do_xcom_push=True,
    )

validate_count = PythonOperator(
        dag = dag,
        task_id='validate_count',
        python_callable=fail_on_zero,
        provide_context=True
    )

pg_unload >> user_purchase_to_s3_stage >> remove_local_user_purchase_file >> user_purchase_to_rs_stage
[movie_review_to_s3_stage, emr_script_to_s3] >> add_emr_steps >> clean_movie_review_data
[user_purchase_to_rs_stage, clean_movie_review_data] >> get_user_behaviour >> get_count >> validate_count >> end_of_data_pipeline