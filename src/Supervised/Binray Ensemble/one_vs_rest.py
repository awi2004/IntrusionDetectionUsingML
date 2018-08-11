from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext, StorageLevel
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import when
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import DecisionTreeClassifier
import os
from time import time
"""
This script is about creating one vs rest model using Random forest classifier
"""

##########################################
## Author:  Kumar Awanish ###
##########################################
memory = '4g'
pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

SparkContext.setSystemProperty('spark.executor.memory', '4g')
sc = SparkContext("local", "App Name")
spark = SparkSession(sc)

datapath = '/Users/akumarmandapati/git/iosl/CSVs/Allcsv/final.csv'
datapath_DosGE = '/Users/akumarmandapati/git/Iosl/CSVs/Wednesday-workingHours.pcap_ISCX.csv'

# Load and parse the data file, converting it to a DataFrame.
try:
    print("Reading Dataset")
    dataset = spark.read.csv(datapath, header=True)
except Exception as IOError:
    print("Error parsing file, check the path")

print("Input N for not including Benign data")
choice = input()

if(choice == 'N'):

    dataset = dataset.withColumnRenamed(" Label", "Label")  # Rename Columns
    dataset = dataset.filter(dataset.Label.isNotNull())
    # selecting rows other than BENIGN
    dataset = dataset.filter(dataset.Label.isin('BENIGN') == False)
    print("Processing without Benign data done!")

elif(choice == 'Y'):
    dataset = dataset.withColumnRenamed(" Label", "Label")  # Rename Columns
    dataset = dataset.filter(dataset.Label.isNotNull())
    print("Utilizing Complete Dataset\n")

# Feature columns for various attacks

cols_select_Dos_GE = [' Bwd Packet Length Std', ' Flow IAT Min', ' Fwd IAT Min', ' Flow IAT Mean', 'Label']
cols_select_PortScan = ['Init_Win_bytes_forward', ' Bwd Packets/s', ' PSH Flag Count', 'Label']
cols_select_Heartbleed = [' Bwd Packet Length Std', ' Subflow Fwd Bytes', ' Flow Duration',
                          'Total Length of Fwd Packets', 'Label']
cols_select_Doshulk = [' Bwd Packet Length Std', ' Flow Duration', ' Flow IAT Std', 'Label']
cols_select_Doshttp = [' Flow Duration', 'Active Mean', ' Active Min', ' Flow IAT Std', 'Label']
cols_select_slowloris = [' Flow Duration', ' Flow IAT Mean', ' Flow IAT Min', ' Bwd IAT Mean']
cols_select_sshpat = ['Init_Win_bytes_forward', ' Subflow Fwd Bytes', 'Total Length of Fwd Packets', ' ACK Flag Count', 'Label']
cols_select_ftppat = ['Init_Win_bytes_forward', 'Fwd PSH Flags', ' SYN Flag Count', 'Fwd Packets/s', 'Label']
cols_select_Webattk = ['Init_Win_bytes_forward', ' Subflow Fwd Bytes', ' Init_Win_bytes_backward',
                       'Total Length of Fwd Packets', 'Label']
cols_select_Infiltrt = [' Subflow Fwd Bytes', 'Total Length of Fwd Packets', ' Flow Duration', 'Active Mean', 'Label']
cols_select_bot = [' Subflow Fwd Bytes', 'Total Length of Fwd Packets', ' Fwd Packet Length Mean', ' Bwd Packets/s', 'Label']
cols_select_Ddos = [' Bwd Packet Length Std', ' Average Packet Size', ' Flow Duration', ' Flow IAT Std', 'Label']



def dos_ge(dataset):
    dataset = dataset.select(' Bwd Packet Length Std', ' Flow IAT Min', ' Fwd IAT Min', ' Flow IAT Mean', 'Label').dropDuplicates()
    #dataset = dataset.filter(dataset.Label.isNotNull())
    dataset_final = dataset.filter(dataset.Label.isin('BENIGN') == False)  # selecting rows other than BENIGN
    print("Count of Attacks post filtering:\n", dataset_final.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "PortScan", "DDoS"]

    # DoS GoldenEye vs the rest of attacks
    dataset_dos_ge = dataset_final.withColumn('Label', when(dataset.Label.isin(labels), 0).otherwise(1))
    print(" DDOS_GE vs ALL_ATTACKS:", dataset_dos_ge.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_dos_ge.select(*(col(c).cast("float").alias(c) for c in dataset_dos_ge.columns))
    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)
    # Merge features to one feature column
    print("Number of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_dge = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))
        # Returns trained model and test data
        global model, data
        model_dge = rfModel_dge
        data_dge = testData
        return model_dge, data_dge

    except Exception as Error:
        print("Error Training Model")

#Function call

#dos_ge(dataset)



def port_scan(dataset):
    dataset = dataset.select('Init_Win_bytes_forward', ' Bwd Packets/s', ' PSH Flag Count', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "DDoS"]

    # Port-Scan vs the rest of attacks
    dataset_port_scan = dataset.withColumn('Label', when(dataset.Label.isin(labels), 0).otherwise(1))
    print(" PortScan vs ALL_ATTACKS:", dataset_port_scan.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_port_scan.select(*(col(c).cast("float").alias(c) for c in dataset_port_scan.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("Number of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_ps = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        #Returns trained model and test data
        global model,data
        model_ps = rfModel_ps
        data_ps = testData
        return model_ps,data_ps

    except Exception as Error:
        print("Error Training Model")


#port_scan(dataset)


def ddos(dataset):
    dataset = dataset.select(' Bwd Packet Length Std', ' Average Packet Size', ' Flow Duration', ' Flow IAT Std', 'Label').dropDuplicates()

    print("Count of Attacks post filtering:\n", dataset.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # DDOS vs the rest of attacks
    dataset_ddos = dataset.withColumn('Label', when(dataset.Label.isin(labels), 0).otherwise(1))
    print(" DDOS vs ALL_ATTACKS:", dataset_ddos.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_ddos.select(*(col(c).cast("float").alias(c) for c in dataset_ddos.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("Number of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_ddos = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_ddos, data_ddos
        model_ddos = rfModel_ddos
        data_ddos = testData
        return model_ddos, data_ddos

    except Exception as Error:
        print("Error Training Model")

#ddos(dataset)


def sshpat(dataset):

    dataset_sshpat = dataset.select('Init_Win_bytes_forward', ' Subflow Fwd Bytes', 'Total Length of Fwd Packets', ' ACK Flag Count', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_sshpat.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "Heartbleed", "DoS Hulk", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # SSHpatator vs the rest of attacks
    dataset_sshpat = dataset.withColumn('Label', when(dataset.Label.isin(labels), 0).otherwise(1))
    print(" SSH Patator vs ALL_ATTACKS:", dataset_sshpat.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_sshpat.select(*(col(c).cast("float").alias(c) for c in dataset_sshpat.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("Number of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_ddos = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_sshpat, data_sshpat
        model_sshpat = rfModel_ddos
        data_sshpat = testData
        return model_sshpat, data_sshpat

    except Exception as Error:
        print("Error Training Model")

#sshpat(dataset)


def heartbleed(dataset):

    dataset_heart_bleed = dataset.select(' Bwd Packet Length Std', ' Subflow Fwd Bytes', ' Flow Duration',
                          'Total Length of Fwd Packets', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_heart_bleed.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "DoS Hulk", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Heartbleed vs the rest of attacks
    dataset_heart_bleed = dataset_heart_bleed.withColumn('Label', when(dataset_heart_bleed.Label.isin(labels), 0).otherwise(1))
    print("\nHeart-Bleed vs ALL_ATTACKS:", dataset_heart_bleed.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_heart_bleed.select(*(col(c).cast("float").alias(c) for c in dataset_heart_bleed.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_heart_bleed = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_heart_bleed, data_heart_bleed
        model_heart_bleed = rfModel_heart_bleed
        data_heart_bleed = testData
        return model_heart_bleed, data_heart_bleed

    except Exception as Error:
        print("Error Training Model")


#heartbleed(dataset)


def doshulk(dataset):

    dataset_doshulk = dataset.select(' Bwd Packet Length Std', ' Flow Duration', ' Flow IAT Std', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_doshulk.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Heartbleed vs the rest of attacks
    dataset_doshulk = dataset_doshulk.withColumn('Label', when(dataset_doshulk.Label.isin(labels), 0).otherwise(1))
    print("\nDos Hulk vs ALL_ATTACKS:", dataset_doshulk.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_doshulk.select(*(col(c).cast("float").alias(c) for c in dataset_doshulk.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_dos_hulk = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_dos_hulk, data_dos_hulk
        model_dos_hulk = rfModel_dos_hulk
        data_dos_hulk = testData
        return model_dos_hulk, data_dos_hulk

    except Exception as Error:
        print("Error Training Model")


#doshulk(dataset)


def doshttp(dataset):

    dataset_doshttp = dataset.select(' Flow Duration', 'Active Mean', ' Active Min', ' Flow IAT Std', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_doshttp.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS slowloris", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Heartbleed vs the rest of attacks
    dataset_doshttp = dataset_doshttp.withColumn('Label', when(dataset_doshttp.Label.isin(labels), 0).otherwise(1))
    print("\nDos Http vs ALL_ATTACKS:", dataset_doshttp.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_doshttp.select(*(col(c).cast("float").alias(c) for c in dataset_doshttp.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_dos_http = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_dos_http, data_dos_http
        model_dos_http = rfModel_dos_http
        data_dos_http = testData
        return model_dos_http, data_dos_http

    except Exception as Error:
        print("Error Training Model")

#doshttp(dataset)

def slowloris(dataset):

    dataset_slowloris = dataset.select(' Flow Duration', ' Flow IAT Mean', ' Flow IAT Min', ' Bwd IAT Mean').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_slowloris.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS Hulk", "FTP-Patator", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Heartbleed vs the rest of attacks
    dataset_slowloris = dataset_slowloris.withColumn('Label', when(dataset_slowloris.Label.isin(labels), 0).otherwise(1))
    print("\nSlowloris vs ALL_ATTACKS:", dataset_slowloris.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_slowloris.select(*(col(c).cast("float").alias(c) for c in dataset_slowloris.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_slowloris = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_slowloris, data_slowloris
        model_slowloris = rfModel_slowloris
        data_slowloris = testData
        return model_slowloris, data_slowloris

    except Exception as Error:
        print("Error Training Model")

#slowloris(dataset)


def ftppat(dataset):

    dataset_ftppat = dataset.select('Init_Win_bytes_forward', 'Fwd PSH Flags', ' SYN Flag Count', 'Fwd Packets/s', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_ftppat.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Heartbleed vs the rest of attacks
    dataset_ftppat = dataset_ftppat.withColumn('Label', when(dataset_ftppat.Label.isin(labels), 0).otherwise(1))
    print("\nFTP Patator vs ALL_ATTACKS:", dataset_ftppat.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_ftppat.select(*(col(c).cast("float").alias(c) for c in dataset_ftppat.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_ftppat = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_ftppat, data_ftppat
        model_ftppat = rfModel_ftppat
        data_ftppat = testData
        return model_ftppat, data_ftppat

    except Exception as Error:
        print("Error Training Model")

#ftppat(dataset)

def webattck(dataset):

    dataset_webattck = dataset.select(' Flow Duration', 'Active Mean', ' Active Min', ' Flow IAT Std', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_webattck.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'FTP-Patator',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "Infiltration", "Bot",
              "DoS GoldenEye", "PortScan"]

    # Webattack vs the rest of attacks
    dataset_webattck = dataset_webattck.withColumn('Label', when(dataset_webattck.Label.isin(labels), 0).otherwise(1))
    print("\nWeb Attack vs ALL_ATTACKS:", dataset_webattck.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_webattck.select(*(col(c).cast("float").alias(c) for c in dataset_webattck.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_webattck = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_webattck, data_webattck
        model_webattck = rfModel_webattck
        data_webattck = testData
        return model_webattck, data_webattck

    except Exception as Error:
        print("Error Training Model")


#webattck(dataset)


def infiltration(dataset):

    dataset_infiltration = dataset.select(' Flow Duration', 'Active Mean', ' Active Min', ' Flow IAT Std', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_infiltration.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "Bot",
              "DoS GoldenEye", "PortScan","FTP-Patator"]

    # Webattack vs the rest of attacks
    dataset_infiltration = dataset_infiltration.withColumn('Label', when(dataset_infiltration.Label.isin(labels), 0).otherwise(1))
    print("\nInfiltration vs ALL_ATTACKS:", dataset_infiltration.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_infiltration.select(*(col(c).cast("float").alias(c) for c in dataset_infiltration.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_infiltration = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_infiltration, data_infiltration
        model_infiltration = rfModel_infiltration
        data_infiltration = testData
        return model_infiltration, data_infiltration

    except Exception as Error:
        print("Error Training Model")

#infiltration(dataset)


def bot(dataset):

    dataset_bot = dataset.select(' Flow Duration', 'Active Mean', ' Active Min', ' Flow IAT Std', 'Label').dropDuplicates()
    print("Count of Attacks post filtering:\n", dataset_bot.groupBy('Label').count().collect())

    # Attack Labels
    labels = ["DoS Slowhttptest", 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS',
              "DDoS", "SSH-Patator", "Heartbleed", "DoS Hulk", "DoS slowloris", "Infiltration",
              "DoS GoldenEye", "PortScan","FTP-Patator"]

    # Webattack vs the rest of attacks
    dataset_bot = dataset_bot.withColumn('Label', when(dataset_bot.Label.isin(labels), 0).otherwise(1))
    print("\nBot vs ALL_ATTACKS:", dataset_bot.groupBy('Label').count().collect())

    # Converting to float values
    final_data = dataset_bot.select(*(col(c).cast("float").alias(c) for c in dataset_bot.columns))

    # Fill Null values
    final_data = final_data.filter(final_data.Label.isNotNull())
    final_data = final_data.na.fill(0.0)

    # Merge features to one feature column
    print("\nNumber of records vs all Attacks", final_data.groupBy('Label').count().collect())
    stages = []  # stages in our Pipeline
    assemblerInputs = final_data.columns[0:-1]  # Specify the columns which are featuers
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # assembler.transform(final_data)
    assembler.transform(final_data.na.drop())
    stages += [assembler]

    cols = final_data.columns
    # Create a Pipeline.
    pipeline = Pipeline(stages=stages)

    # Run the feature transformations.
    #  - fit() computes feature statistics as needed.
    #  - transform() actually transforms the features.

    pipelineModel = pipeline.fit(final_data)
    final_data = pipelineModel.transform(final_data)

    # Keep relevant columns
    selectedcols = ["features"] + cols
    final_data = final_data.select(selectedcols)
    # display(final_data)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=200)
    print("\nNumber of records for training: " + str(trainingData.count()))
    print("\nNumber of records for evaluation: " + str(testData.count()))

    # Create a RandomForest model.
    rf = RandomForestClassifier(labelCol="Label", featuresCol="features")
    try:
        # Train model with Training Data
        print("Training Model with RandomForest:")
        start = time()
        rfModel_bot = rf.fit(trainingData)
        print("RF training took %.2f seconds" % ((time() - start)))

        # Returns trained model and test data
        global model_bot, data_bot
        model_bot = rfModel_bot
        data_bot = testData
        return model_bot, data_bot

    except Exception as Error:
        print("Error Training Model")

def predictor(model,testdata):
    print("Running Predict Function")
    print("Predicting Model with RF Model:")
    predictions = model.transform(testdata)
    evaluator = BinaryClassificationEvaluator(labelCol="Label")

    print("Evaluation Metric", evaluator.evaluate(predictions) * 100)
    print("Metric Used", evaluator.getMetricName())

bot(dataset)


predictor(model_bot,data_bot)
