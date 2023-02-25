from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, BooleanType, DoubleType
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as F
from google.cloud import storage
import PyPDF2
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

def pdf_2_txt(src,dest):
    reader = PyPDF2.PdfReader(src)
    txt = ''
    for page in reader.pages:
        txt += page.extract_text()

    with open(dest, 'w') as f:
        f.write(txt)


def spark_download_public_file(x):
    
    bucket_name = 'arxiv-dataset'
    prefix = x.prefix
    date = x.date
    suffix = x.suffix

    source_blob_name = 'arxiv/'+prefix+'/pdf/'+date+'/'+suffix+'v1.pdf'


    if not os.path.exists('articles'):
        os.mkdir('articles')
    if not os.path.exists('articles/'+prefix):
        os.mkdir('articles/'+prefix)
    if not os.path.exists('articles/'+prefix+'/'+date):
        os.mkdir('articles/'+prefix+'/'+date)

    destination = 'articles/'+prefix+'/'+date+'/'+suffix+'v1.pdf'
    src = destination
    dest = src[:-3] + 'txt'

    if not os.path.exists(dest):
        try:
            download_public_file(bucket_name, source_blob_name, destination)

            pdf_2_txt(src=src, dest=dest)

            os.remove(src)
        except:
            print(suffix)

    
ids.foreach(spark_download_public_file)