
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
import numpy as np
import random
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.executor.cores",'3')
conf.set("spark.executor.instances",'1')
conf.set("spark.executor.memory", "1g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)
sqlcontext = SQLContext(sc)

def shuffle_csv(csv_file):
    lines = open(csv_file).readlines()
    random.shuffle(lines)
    open(csv_file, 'w').writelines(lines)

def load_data_frame(csv_file, shuffle=True, train=True):
    if shuffle:
        shuffle_csv(csv_file)
    data = sc.textFile('/home/minglu/dist_spark/data/' + csv_file) # This is an RDD, which will later be transformed to a data frame
    data = data.filter(lambda x:x.split(',')[0] != 'label').map(lambda line: line.split(','))
    if train:
        data = data.map(
            lambda line: (Vectors.dense(np.asarray(line[1:]).astype(np.float32)),
                          'class_'+str(line[0]),int(line[0])) )
    else:
        # Test data gets dummy labels. We need the same structure as in Train data
        data = data.map( lambda line: (Vectors.dense(np.asarray(line[1:]).astype(np.float32)),'class_'+str(line[0]),int(line[0])) ) 
    return sqlcontext.createDataFrame(data, ['features', 'category','label'])

train_df = load_data_frame("train.csv")
test_df = load_data_frame("test.csv", shuffle=False, train=False)
from pyspark.ml.feature import StringIndexer

string_indexer = StringIndexer(inputCol="category", outputCol="index_category")
fitted_indexer = string_indexer.fit(train_df)
indexed_df = fitted_indexer.transform(train_df)

from distkeras.transformers import *
from pyspark.ml.feature import OneHotEncoder
####OneHot
nb_classes = 9
encoder = OneHotTransformer(nb_classes, input_col='label', output_col="label_encoded")
dataset_train = encoder.transform(indexed_df)
dataset_test = encoder.transform(test_df)

###encoder
from pyspark.ml.feature import MinMaxScaler
transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, \
                                o_min=0.0, o_max=250.0, \
                                input_col="features", \
                                output_col="features_normalized")
# Transform the dataset
dataset_train = transformer.transform(dataset_train)
dataset_test = transformer.transform(dataset_test)

# vectors using Spark.
reshape_transformer = ReshapeTransformer("features_normalized", "matrix", (64, 64, 3))
dataset_train = reshape_transformer.transform(dataset_train)
dataset_test = reshape_transformer.transform(dataset_test)



from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

#scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
#fitted_scaler = scaler.fit(indexed_df)
#scaled_df = fitted_scaler.transform(indexed_df)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, generic_utils
from keras.optimizers import *
from keras.layers.core import *
from keras.layers.convolutional import *

from keras.layers.noise import GaussianDropout
nb_classes = train_df.select("category").distinct().count()
input_dim = len(train_df.select("features").first()[0])

####model
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(2,2),
                 padding='same',
                 input_shape=(64,64,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=64,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=128,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=256,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=512,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=1024,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))
model.add(Flatten())

model.add(Dense(1000,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(500,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(nb_classes,activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam')


#################################################################分散模型前處理
# However, for this we need to specify a procedure how to do this.
def evaluate_accuracy(model, test_set, features="matrix"):
    evaluator = AccuracyEvaluator(prediction_col="prediction_index", label_col="label")
    predictor = ModelPredictor(keras_model=model, features_col=features)
    transformer = LabelIndexTransformer(output_dim=nb_classes)
    test_set = test_set.select(features, "label")
    test_set = predictor.predict(test_set)
    test_set = transformer.transform(test_set)
    score = evaluator.evaluate(test_set)
    return score

# Select the desired columns, this will reduce network usage.
dataset_train1 = dataset_train.select("features_normalized", "matrix","label", "label_encoded")
dataset_test1 = dataset_test.select("features_normalized", "matrix","label", "label_encoded")
# Keras expects DenseVectors.
dense_transformer = DenseTransformer(input_col="features_normalized", output_col="features_normalized_dense")
dataset_train1 = dense_transformer.transform(dataset_train)
dataset_test1 = dense_transformer.transform(dataset_test)
dataset_train1.repartition(1)
dataset_test1.repartition(1)
# Assing the training and test set.
training_set = dataset_train.repartition(1)
test_set = dataset_test.repartition(1)
# Cache them.
training_set.cache()
test_set.cache()
# Precache the trainingset on the nodes using a simple count.
print(training_set.count())

###################
from distkeras.trainers import *
trainer = ADAG(keras_model=model, worker_optimizer='adam', loss='categorical_crossentropy',
               num_workers=1, batch_size=100, communication_window=5, num_epoch=50,
               features_col="matrix", label_col="label_encoded"
               )
trained_model = trainer.train(training_set)
from distkeras.predictors import *
from distkeras.transformers import *
from distkeras.evaluators import *
from distkeras.utils import *

print("Training time: " + str(trainer.get_training_time()))
print("Accuracy: " + str(evaluate_accuracy(trained_model, test_set)))
print("Number of parameter server updates: " + str(trainer.parameter_server.num_updates))

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[string_indexer, scaler, trainer_model])

from pyspark.mllib.evaluation import MulticlassMetrics

fitted_pipeline = pipeline.fit(dataset_train) # Fit model to data

prediction = fitted_pipeline.transform(dataset_train) # Evaluate on train data.
# prediction = fitted_pipeline.transform(test_df) # <-- The same code evaluates test data.
pnl = prediction.select("index_category", "prediction")
pnl.show(100)

prediction_and_label = pnl.map(lambda row: (row.index_category, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print(metrics.precision())
