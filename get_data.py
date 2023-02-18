from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, BooleanType, DoubleType
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as F
from google.cloud import storage
import os

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("PySpark Read JSON") \
    .getOrCreate()

# Reading JSON file into dataframe    
dataframe = spark.read.json("metadata.json")


@udf
def extract_prefix(s):
    split = s.split('/')

    if len(split) == 1:
        return 'arxiv'
    else:
        return split[0]

@udf
def extract_date(s):
    split = s.split('/')
    if len(split) == 1:
        return split[0][:4]
    else:
        return split[1][:4]

@udf
def extract_suffix(s):
    split = s.split('/')
    if len(split) == 1:
        return split[0]
    else:
        return split[1]
    

ids = dataframe\
    .withColumn('abstract_length', F.length('abstract'))\
    .where('abstract_length > 500') \
    .select('id') \
    .withColumn('prefix',extract_prefix(dataframe.id)) \
    .withColumn('date', extract_date(dataframe.id)) \
    .withColumn('suffix', extract_suffix(dataframe.id))

def download_public_file(bucket_name, source_blob_name, destination_file_name):


    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )


def spark_download_public_file(x):
    
    bucket_name = 'arxiv-dataset'
    prefix = x.prefix
    date = x.date
    suffix = x.suffix

    source_blob_name = 'arxiv/'+prefix+'/pdf/'+date+'/'+suffix+'v1.pdf'


    if not os.path.exists('/pine/scr/z/g/zgordo/articles'):
        os.mkdir('/pine/scr/z/g/zgordo/articles')
    if not os.path.exists('articles/'+prefix):
        os.mkdir('articles/'+prefix)
    if not os.path.exists('/pine/scr/z/g/zgordo/articles/'+prefix+'/'+date):
        os.mkdir('/pine/scr/z/g/zgordo/articles/'+prefix+'/'+date)

    destination = '/pine/scr/z/g/zgordo/articles/'+prefix+'/'+date+'/'+suffix+'v1.pdf'

    download_public_file(bucket_name, source_blob_name, destination)


    
ids.foreach(spark_download_public_file)