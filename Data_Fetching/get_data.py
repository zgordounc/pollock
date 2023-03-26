from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, BooleanType, DoubleType
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as F
from google.cloud import storage
import PyPDF2
import os
import pandas as pd

# starts the local spark server
spark = SparkSession.builder \
    .appName("PySpark Read JSON") \
    .getOrCreate()

# Reading JSON file into dataframe    
# update with your path to metadata.json
dataframe = spark.read.json("Data_Fetching/metadata.json")


legend = pd.DataFrame({
    'abstract_path' : [],
    'raw_path' : [],
})

# extracts the prefix which is a folder name
@udf
def extract_prefix(s):
    split = s.split('/')

    if len(split) == 1:
        return 'arxiv'
    else:
        return split[0]

# extracts the date which is a folder name
@udf
def extract_date(s):
    split = s.split('/')
    if len(split) == 1:
        return split[0][:4]
    else:
        return split[1][:4]

# extracts the suffix name which is a part of the file name
@udf
def extract_suffix(s):
    split = s.split('/')
    if len(split) == 1:
        return split[0]
    else:
        return split[1]
    

# use spark to efficiently apply the above functions to extract prefix, data, and sufffix
ids = dataframe\
    .withColumn('abstract_length', F.length('abstract'))\
    .where('abstract_length > 500') \
    .select('id','abstract') \
    .withColumn('prefix',extract_prefix(dataframe.id)) \
    .withColumn('date', extract_date(dataframe.id)) \
    .withColumn('suffix', extract_suffix(dataframe.id))

# a function which downloads from GCS and writes the file
def download_public_file(bucket_name, source_blob_name, destination_file_name):
    '''
    ARGUMENTS:
        bucket_name = the name of the bucket in our case it will be 'arxiv-dataset'
        source_blob_name = the file path to the article we want to download within the bucket
        destination_file_name = the name of the file that the function rights

    OUTPUT:
        No output but the file written is a pdf
    '''

    # spin of client to access GCS
    storage_client = storage.Client.create_anonymous_client()

    # grab the correct bucket from GCS
    bucket = storage_client.bucket(bucket_name)

    # grab blob from bucket
    blob = bucket.blob(source_blob_name)

    # download the blob to destination
    blob.download_to_filename(destination_file_name)

    # print statement to show download happening
    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )

# converts a local pdf file to txt
def pdf_2_txt(src,dest):
    reader = PyPDF2.PdfReader(src)
    txt = ''
    for page in reader.pages:
        txt += page.extract_text()

    with open(dest, 'w') as f:
        f.write(txt)

# This is a wrapper function for download_public_file that reads from our dataframe and then converts the downloaded pdf to a txt 
def spark_download_public_file(x):
    '''
    ARGUMENTS:
        x = Spark dataframe containing columns ['prefix', 'data', 'suffix']
    
    OUTPUT:
        no output but converts pdf to txt
    '''

    # extract information from the columns in x
    bucket_name = 'arxiv-dataset'
    prefix = x.prefix
    date = x.date
    suffix = x.suffix

    # construct the sourc_blob_name argument
    source_blob_name = 'arxiv/'+prefix+'/pdf/'+date+'/'+suffix+'v1.pdf'


    # check if folders exist
    # If the folders do not exist make them
    if not os.path.exists('articles'):
        os.mkdir('articles')
    if not os.path.exists('articles/'+prefix):
        os.mkdir('articles/'+prefix)
    if not os.path.exists('articles/'+prefix+'/'+date):
        os.mkdir('articles/'+prefix+'/'+date)

    if not os.path.exists('abstracts'):
        os.mkdir('abstracts')
    if not os.path.exists('abstracts/'+prefix):
        os.mkdir('abstracts/'+prefix)
    if not os.path.exists('abstracts/'+prefix+'/'+date):
        os.mkdir('abstracts/'+prefix+'/'+date)
    

    # construct destination file path
    destination = 'Data_Fetching/articles/'+prefix+'/'+date+'/'+suffix+'v1.pdf'
    abs_dest = 'Data_Fetching/abstracts/'+prefix+'/'+date+'/'+suffix+'v1_abstract.txt'


    # making a copy of destination to be the source file fro converting the pdf to txt
    src = destination

    # create destination for txt file by replacing .pdf with .txt
    dest = src[:-3] + 'txt'

    # check if file exists so we don't download the same file twice
    if not os.path.exists(dest):
        # try block is used to keep the program running if it hits an error while downloading
        try:
            download_public_file(bucket_name, source_blob_name, destination)

            pdf_2_txt(src=src, dest=dest)

            os.remove(src)
        except:
            print(suffix)

    if not os.path.exists(abs_dest):
        with open(abs_dest, 'w') as f:
            f.write(x.abstract)

    if os.path.exists(dest) and os.path.exists(abs_dest):
        row = [dest, abs_dest]
        legend.loc[len(legend)] = row


        legend.to_csv('Data_Fetching/legend.csv', index=False)

# iterates through df to apply spark_download_public_file
ids.foreach(spark_download_public_file)

