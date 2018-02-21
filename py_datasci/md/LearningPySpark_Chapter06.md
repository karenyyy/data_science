
## Predict chances of infant survival with ML


```python
spark
```




    <pyspark.sql.session.SparkSession at 0x7f475960bc88>




```python
# prepare data
import pyspark.sql.types as typ
import pyspark.sql.functions as func
labels = [
    ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.IntegerType()),
    ('DIABETES_GEST', typ.IntegerType()),
    ('HYP_TENS_PRE', typ.IntegerType()),
    ('HYP_TENS_GEST', typ.IntegerType()),
    ('PREV_BIRTH_PRETERM', typ.IntegerType())
]

schema = typ.StructType([
    typ.StructField(e[0], e[1], False) for e in labels
])

births = spark.read.csv('data/births_transformed.csv.gz', 
                        header=True, 
                        schema=schema)
births = births.withColumn(
    'INFANT_ALIVE_AT_REPORT', 
    func.col('INFANT_ALIVE_AT_REPORT').cast(typ.DoubleType())
)
```

### Create transformers


```python
births = births.withColumn('BIRTH_PLACE_INT', births['BIRTH_PLACE'].cast(
    typ.IntegerType()))
births
```




    DataFrame[INFANT_ALIVE_AT_REPORT: double, BIRTH_PLACE: string, MOTHER_AGE_YEARS: int, FATHER_COMBINED_AGE: int, CIG_BEFORE: int, CIG_1_TRI: int, CIG_2_TRI: int, CIG_3_TRI: int, MOTHER_HEIGHT_IN: int, MOTHER_PRE_WEIGHT: int, MOTHER_DELIVERY_WEIGHT: int, MOTHER_WEIGHT_GAIN: int, DIABETES_PRE: int, DIABETES_GEST: int, HYP_TENS_PRE: int, HYP_TENS_GEST: int, PREV_BIRTH_PRETERM: int, BIRTH_PLACE_INT: int]



Create `Transformer`.


```python
# one hot encoder
import pyspark.ml.feature as ft

encoder = ft.OneHotEncoder(
    inputCol='BIRTH_PLACE_INT', 
    outputCol='BIRTH_PLACE_VEC')

```

Now we create a single column with all the features collated together. 


```python
# vector assembler
featuresCreator = ft.VectorAssembler(
    inputCols=[col[0] for col in labels[2:]] + [encoder.getOutputCol()],
    outputCol='features')

```

### Create an estimator


```python
import pyspark.ml.classification as cl
```


```python
logistic = cl.LogisticRegression(
    maxIter=10, 
    regParam=0.01, 
    labelCol='INFANT_ALIVE_AT_REPORT')
```

### Create a pipeline


```python
# pipeline
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
        encoder, 
        featuresCreator, 
        logistic
    ])
```

### Fit the model


```python
births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)
for i in births_train.take(10):
    print(i)
    
```

    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=12, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=60, MOTHER_PRE_WEIGHT=154, MOTHER_DELIVERY_WEIGHT=154, MOTHER_WEIGHT_GAIN=0, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=12, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=62, MOTHER_PRE_WEIGHT=145, MOTHER_DELIVERY_WEIGHT=152, MOTHER_WEIGHT_GAIN=7, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=57, MOTHER_PRE_WEIGHT=100, MOTHER_DELIVERY_WEIGHT=108, MOTHER_WEIGHT_GAIN=8, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=62, MOTHER_PRE_WEIGHT=218, MOTHER_DELIVERY_WEIGHT=240, MOTHER_WEIGHT_GAIN=22, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=64, MOTHER_PRE_WEIGHT=125, MOTHER_DELIVERY_WEIGHT=143, MOTHER_WEIGHT_GAIN=18, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=67, MOTHER_PRE_WEIGHT=194, MOTHER_DELIVERY_WEIGHT=196, MOTHER_WEIGHT_GAIN=2, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=1, CIG_1_TRI=1, CIG_2_TRI=1, CIG_3_TRI=1, MOTHER_HEIGHT_IN=61, MOTHER_PRE_WEIGHT=115, MOTHER_DELIVERY_WEIGHT=130, MOTHER_WEIGHT_GAIN=15, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=14, FATHER_COMBINED_AGE=15, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=61, MOTHER_PRE_WEIGHT=128, MOTHER_DELIVERY_WEIGHT=127, MOTHER_WEIGHT_GAIN=0, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=14, FATHER_COMBINED_AGE=16, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=63, MOTHER_PRE_WEIGHT=180, MOTHER_DELIVERY_WEIGHT=206, MOTHER_WEIGHT_GAIN=26, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=14, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=57, MOTHER_PRE_WEIGHT=100, MOTHER_DELIVERY_WEIGHT=134, MOTHER_WEIGHT_GAIN=34, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)


Now run our `pipeline` and estimate our model.


```python
model = pipeline.fit(births_train)
test_model = model.transform(births_test)
```

Here's what the `test_model` looks like.


```python
test_model.take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=66, MOTHER_PRE_WEIGHT=133, MOTHER_DELIVERY_WEIGHT=135, MOTHER_WEIGHT_GAIN=2, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1, BIRTH_PLACE_VEC=SparseVector(9, {1: 1.0}), features=SparseVector(24, {0: 13.0, 1: 99.0, 6: 66.0, 7: 133.0, 8: 135.0, 9: 2.0, 16: 1.0}), rawPrediction=DenseVector([1.0573, -1.0573]), probability=DenseVector([0.7422, 0.2578]), prediction=0.0)]



### Model performance

Obviously, we would like to now test how well our model did.


```python
# performance
import pyspark.ml.evaluation as ev

evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='INFANT_ALIVE_AT_REPORT')

print(evaluator.evaluate(test_model, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))
```

    0.7401301847095617
    0.7139354342365674


### Saving the model

PySpark allows saving  the `Pipeline` definition for later use.


```python
pipelinePath = 'data/oneHotEncoder_Logistic_Pipeline'
pipeline.write().overwrite().save(pipelinePath)
```

So, you can load it up later and use straight away to `.fit(...)` and predict.


```python
loadedPipeline = Pipeline.load(pipelinePath)
loadedPipeline \
    .fit(births_train)\
    .transform(births_test)\
    .take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=66, MOTHER_PRE_WEIGHT=133, MOTHER_DELIVERY_WEIGHT=135, MOTHER_WEIGHT_GAIN=2, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1, BIRTH_PLACE_VEC=SparseVector(9, {1: 1.0}), features=SparseVector(24, {0: 13.0, 1: 99.0, 6: 66.0, 7: 133.0, 8: 135.0, 9: 2.0, 16: 1.0}), rawPrediction=DenseVector([1.0573, -1.0573]), probability=DenseVector([0.7422, 0.2578]), prediction=0.0)]



We can also save the whole model


```python
from pyspark.ml import PipelineModel

modelPath = 'data/oneHotEncoder_Logistic_PipelineModel'
model.write().overwrite().save(modelPath)

loadedPipelineModel = PipelineModel.load(modelPath)
test_loadedModel = loadedPipelineModel.transform(births_test)
```

## Parameter hyper-tuning

### Grid search

Load the `.tuning` part of the package.


```python
import pyspark.ml.tuning as tune
```

Next let's specify the model and the list of hyperparameters we want to loop through.


```python
logistic = cl.LogisticRegression(
    labelCol='INFANT_ALIVE_AT_REPORT')

grid = tune.ParamGridBuilder() \
    .addGrid(logistic.maxIter,  
             [2, 10, 50]) \
    .addGrid(logistic.regParam, 
             [0.01, 0.05, 0.3]) \
    .build()
```

Next, we need some way of comparing the models.


```python
import pyspark.ml.evaluation as ev
evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='INFANT_ALIVE_AT_REPORT')
```

Create the logic that will do the validation work for us.


```python
pipeline = Pipeline(stages=
                    [encoder,
                     featuresCreator])
cv = tune.CrossValidator(
    estimator=logistic, 
    estimatorParamMaps=grid, 
    evaluator=evaluator
)
data_transformer = pipeline.fit(births_train)
```


```python
cvModel = cv.fit(data_transformer.transform(births_train))
```

The `cvModel` will return the best model estimated. We can now use it to see if it performed better than our previous model.


```python
# evaluation
data_train = data_transformer.transform(births_test)
results = cvModel.transform(data_train)

print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderPR'}))
```

    0.7404304424804281
    0.7156729757616691


What parameters has the best model? The answer is a little bit convoluted but here's how you can extract it.


```python
results = [
    (
        [
            {key.name: paramValue} 
            for key, paramValue 
            in zip(
                params.keys(), 
                params.values())
        ], metric
    ) 
    for params, metric 
    in zip(
        cvModel.getEstimatorParamMaps(), 
        cvModel.avgMetrics
    )
]

sorted(results, 
       key=lambda el: el[1], 
       reverse=True)[0]
```




    ([{'maxIter': 50}, {'regParam': 0.01}], 0.7386395855167077)



### Train-Validation splitting

Use the `ChiSqSelector` to select only top 5 features, thus limiting the complexity of our model.


```python
selector = ft.ChiSqSelector(
    numTopFeatures=5, 
    featuresCol=featuresCreator.getOutputCol(), 
    outputCol='selectedFeatures',
    labelCol='INFANT_ALIVE_AT_REPORT'
)

logistic = cl.LogisticRegression(
    labelCol='INFANT_ALIVE_AT_REPORT',
    featuresCol='selectedFeatures'
)

pipeline = Pipeline(stages=[encoder,
                            featuresCreator,
                            selector])
data_transformer = pipeline.fit(births_train)
```

The `TrainValidationSplit` object gets created in the same fashion as the `CrossValidator` model.


```python
tvs = tune.TrainValidationSplit(
    estimator=logistic, 
    estimatorParamMaps=grid, 
    evaluator=evaluator
)
```


```python
#evaluation
tvsModel = tvs.fit(
    data_transformer \
        .transform(births_train)
)

data_train = data_transformer \
    .transform(births_test)
results = tvsModel.transform(data_train)

print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderPR'}))
```

    0.7334857800726642
    0.7071651608758281


## Other features of PySpark ML in action

### Feature extraction

#### NLP related feature extractors

Simple dataset.


```python
text_data = spark.createDataFrame([
    ['''Machine learning can be applied to a wide variety 
        of data types, such as vectors, text, images, and 
        structured data. This API adopts the DataFrame from 
        Spark SQL in order to support a variety of data types.'''],
    ['''DataFrame supports many basic and structured types; 
        see the Spark SQL datatype reference for a list of 
        supported types. In addition to the types listed in 
        the Spark SQL guide, DataFrame can use ML Vector types.'''],
    ['''A DataFrame can be created either implicitly or 
        explicitly from a regular RDD. See the code examples 
        below and the Spark SQL programming guide for examples.'''],
    ['''Columns in a DataFrame are named. The code examples 
        below use names such as "text," "features," and "label."''']
], ['input'])
```

First, we need to tokenize this text.


```python
import pyspark.ml.feature as ft

tokenizer = ft.RegexTokenizer(
    inputCol='input', 
    outputCol='input_arr', 
    pattern='\s+|[,.\"]')
```

The output of the tokenizer looks similar to this.


```python
tok = tokenizer \
    .transform(text_data) \
    .select('input_arr') 

tok.take(1)
```




    [Row(input_arr=['machine', 'learning', 'can', 'be', 'applied', 'to', 'a', 'wide', 'variety', 'of', 'data', 'types', 'such', 'as', 'vectors', 'text', 'images', 'and', 'structured', 'data', 'this', 'api', 'adopts', 'the', 'dataframe', 'from', 'spark', 'sql', 'in', 'order', 'to', 'support', 'a', 'variety', 'of', 'data', 'types'])]



Use the `StopWordsRemover(...)`.


```python
stopwords = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(), 
    outputCol='input_stop')
```

The output of the method looks as follows


```python
stopwords.transform(tok).select('input_stop').take(1)
```




    [Row(input_stop=['machine', 'learning', 'applied', 'wide', 'variety', 'data', 'types', 'vectors', 'text', 'images', 'structured', 'data', 'api', 'adopts', 'dataframe', 'spark', 'sql', 'order', 'support', 'variety', 'data', 'types'])]



Build `NGram` model and the `Pipeline`.


```python
ngram = ft.NGram(n=2, 
    inputCol=stopwords.getOutputCol(), 
    outputCol="nGrams")

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords, ngram])
```

Now that we have the `pipeline` we follow in the very similar fashion as before.


```python
data_ngram = pipeline \
    .fit(text_data) \
    .transform(text_data)
    
data_ngram.select('nGrams').take(1)
```




    [Row(nGrams=['machine learning', 'learning applied', 'applied wide', 'wide variety', 'variety data', 'data types', 'types vectors', 'vectors text', 'text images', 'images structured', 'structured data', 'data api', 'api adopts', 'adopts dataframe', 'dataframe spark', 'spark sql', 'sql order', 'order support', 'support variety', 'variety data', 'data types'])]



That's it. We got our n-grams and we can then use them in further NLP processing.

#### Discretize continuous variables

It is sometimes useful to *band* the values into discrete buckets.


```python
import numpy as np

x = np.arange(0, 100)
x = x / 100.0 * np.pi * 4
y = x * np.sin(x / 1.764) + 20.1234

schema = typ.StructType([
    typ.StructField('continuous_var', 
                    typ.DoubleType(), 
                    False
   )
])

data = spark.createDataFrame([[float(e), ] for e in y], schema=schema)
data.show()
```

    +------------------+
    |    continuous_var|
    +------------------+
    |           20.1234|
    |20.132344452369832|
    |20.159087064491775|
    |20.203356291885854|
    | 20.26470185735763|
    |20.342498180090526|
    |  20.4359491438498|
    |20.544094172020312|
    |20.665815568330437|
    |20.799847073505322|
    |  20.9447835797997|
    | 21.09909193743627|
    |21.261122779470593|
    | 21.42912328456607|
    | 21.60125079063745|
    |21.775587166351258|
    |21.950153842094366|
    |22.122927397273514|
    |22.291855596719525|
    |22.454873765567744|
    +------------------+
    only showing top 20 rows
    


Use the `QuantileDiscretizer` model to split our continuous variable into 5 buckets (see the `numBuckets` parameter).


```python
discretizer = ft.QuantileDiscretizer(
    numBuckets=5, 
    inputCol='continuous_var', 
    outputCol='discretized')
```

Let's see what we got.


```python
data_discretized = discretizer.fit(data).transform(data)
data_discretized.show(50)
```

    +------------------+-----------+
    |    continuous_var|discretized|
    +------------------+-----------+
    |           20.1234|        2.0|
    |20.132344452369832|        2.0|
    |20.159087064491775|        2.0|
    |20.203356291885854|        2.0|
    | 20.26470185735763|        2.0|
    |20.342498180090526|        2.0|
    |  20.4359491438498|        2.0|
    |20.544094172020312|        2.0|
    |20.665815568330437|        2.0|
    |20.799847073505322|        2.0|
    |  20.9447835797997|        2.0|
    | 21.09909193743627|        2.0|
    |21.261122779470593|        3.0|
    | 21.42912328456607|        3.0|
    | 21.60125079063745|        3.0|
    |21.775587166351258|        3.0|
    |21.950153842094366|        3.0|
    |22.122927397273514|        3.0|
    |22.291855596719525|        3.0|
    |22.454873765567744|        3.0|
    |22.609921389293767|        3.0|
    |22.754958823619244|        3.0|
    |22.887983997780925|        4.0|
    | 23.00704899418632|        4.0|
    | 23.11027638776662|        4.0|
    |23.195875229382086|        4.0|
    | 23.26215655943226|        4.0|
    |23.307548340364534|        4.0|
    | 23.33060970004563|        4.0|
    |23.330044381943413|        4.0|
    | 23.30471330273859|        4.0|
    |23.253646123319946|        4.0|
    |23.176051745081992|        4.0|
    | 23.07132765000503|        4.0|
    |22.939068010115477|        4.0|
    |22.779070499557093|        3.0|
    | 22.59134175060465|        3.0|
    | 22.37610140347311|        3.0|
    |22.133784708664646|        3.0|
    |21.865043649799823|        3.0|
    | 21.57074656434146|        3.0|
    |21.251976249282055|        3.0|
    |20.910026548668988|        2.0|
    | 20.54639742972568|        2.0|
    |20.162788564229796|        2.0|
    |19.761091441670477|        2.0|
    | 19.34338005046327|        2.0|
    | 18.91190017309325|        2.0|
    |18.469057350422936|        1.0|
    |18.017403579482767|        1.0|
    +------------------+-----------+
    only showing top 50 rows
    



```python
data_discretized \
    .groupby('discretized')\
    .mean('continuous_var')\
    .sort('discretized')\
    .show()
```

    +-----------+-------------------+
    |discretized|avg(continuous_var)|
    +-----------+-------------------+
    |        0.0| 12.314360733007915|
    |        1.0| 16.046244793347466|
    |        2.0|  20.25079947835259|
    |        3.0| 22.040988218437327|
    |        4.0| 24.264824657002865|
    +-----------+-------------------+
    


#### Standardizing continuous variables

Create a vector representation of our continuous variable (as it is only a single float)



```python
vectorizer = ft.VectorAssembler(
    inputCols=['continuous_var'], 
    outputCol= 'continuous_vec')
```

Build a `normalizer` and a `pipeline`.


```python
normalizer = ft.StandardScaler(
    inputCol=vectorizer.getOutputCol(), 
    outputCol='normalized', 
    withMean=True,
    withStd=True
)

pipeline = Pipeline(stages=[vectorizer, normalizer])
data_standardized = pipeline.fit(data).transform(data)
data_standardized.show()
```

    +------------------+--------------------+--------------------+
    |    continuous_var|      continuous_vec|          normalized|
    +------------------+--------------------+--------------------+
    |           20.1234|           [20.1234]|[0.2342913955450245]|
    |20.132344452369832|[20.132344452369832]|[0.23630959828688...|
    |20.159087064491775|[20.159087064491775]|[0.2423437310517903]|
    |20.203356291885854|[20.203356291885854]|[0.2523325232564444]|
    | 20.26470185735763| [20.26470185735763]|[0.26617437553725...|
    |20.342498180090526|[20.342498180090526]|[0.2837281334817449]|
    |  20.4359491438498|  [20.4359491438498]|[0.3048141635135432]|
    |20.544094172020312|[20.544094172020312]|[0.32921572364798...|
    |20.665815568330437|[20.665815568330437]|[0.3566806198337413]|
    |20.799847073505322|[20.799847073505322]|[0.3869231366536311]|
    |  20.9447835797997|  [20.9447835797997]|[0.4196262292862515]|
    | 21.09909193743627| [21.09909193743627]|[0.4544439618423727]|
    |21.261122779470593|[21.261122779470593]|[0.49100417549639...|
    | 21.42912328456607| [21.42912328456607]| [0.528911368245372]|
    | 21.60125079063745| [21.60125079063745]| [0.567749766655784]|
    |21.775587166351258|[21.775587166351258]|[0.6070865686111453]|
    |21.950153842094366|[21.950153842094366]|[0.6464753348602796]|
    |22.122927397273514|[22.122927397273514]|[0.6854595060946458]|
    |22.291855596719525|[22.291855596719525]|[0.7235760213604683]|
    |22.454873765567744|[22.454873765567744]|[0.7603590128437548]|
    +------------------+--------------------+--------------------+
    only showing top 20 rows
    


### Classification

Now try to use the `RandomForestClassfier` to model the chances of survival for an infant.

First, we need to cast the label feature to `DoubleType`.


```python
import pyspark.sql.functions as func

births = births.withColumn(
    'INFANT_ALIVE_AT_REPORT', 
    func.col('INFANT_ALIVE_AT_REPORT').cast(typ.DoubleType())
)

births_train, births_test = births \
    .randomSplit([0.7, 0.3], seed=666)
    
for i in births_train.take(3):
    print(i)
```

    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=12, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=60, MOTHER_PRE_WEIGHT=154, MOTHER_DELIVERY_WEIGHT=154, MOTHER_WEIGHT_GAIN=0, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=12, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=62, MOTHER_PRE_WEIGHT=145, MOTHER_DELIVERY_WEIGHT=152, MOTHER_WEIGHT_GAIN=7, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)
    Row(INFANT_ALIVE_AT_REPORT=0.0, BIRTH_PLACE='1', MOTHER_AGE_YEARS=13, FATHER_COMBINED_AGE=99, CIG_BEFORE=0, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=57, MOTHER_PRE_WEIGHT=100, MOTHER_DELIVERY_WEIGHT=108, MOTHER_WEIGHT_GAIN=8, DIABETES_PRE=0, DIABETES_GEST=0, HYP_TENS_PRE=0, HYP_TENS_GEST=0, PREV_BIRTH_PRETERM=0, BIRTH_PLACE_INT=1)


We are ready to build our model.


```python
classifier = cl.RandomForestClassifier(
    numTrees=50, 
    maxDepth=5, 
    labelCol='INFANT_ALIVE_AT_REPORT')

pipeline = Pipeline(
    stages=[
        encoder,
        featuresCreator, 
        classifier])

model = pipeline.fit(births_train)
test = model.transform(births_test)
```

Now see how the `RandomForestClassifier` model performs compared to the `LogisticRegression`.


```python
evaluator = ev.BinaryClassificationEvaluator(
    labelCol='INFANT_ALIVE_AT_REPORT')
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderROC"}))
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderPR"}))
```

    0.778265377684722
    0.7562887533811923


slightly better than logistic.

Let's test how well would one tree do, then.


```python
# default single decision tree classifier
classifier = cl.DecisionTreeClassifier(
    maxDepth=5, 
    labelCol='INFANT_ALIVE_AT_REPORT')
pipeline = Pipeline(stages=[
        encoder,
        featuresCreator, 
        classifier]
)

model = pipeline.fit(births_train)
test = model.transform(births_test)

evaluator = ev.BinaryClassificationEvaluator(
    labelCol='INFANT_ALIVE_AT_REPORT')
print(evaluator.evaluate(test, 
     {evaluator.metricName: "areaUnderROC"}))
print(evaluator.evaluate(test, 
     {evaluator.metricName: "areaUnderPR"}))
```

    0.7582781726635287
    0.7787580540118526


Not much difference.

### Clustering

In this example we will use k-means model to find similarities in the births data.


```python
import pyspark.ml.clustering as cluster

kmeans = cluster.KMeans(k = 5, 
    featuresCol='features')

pipeline = Pipeline(stages=[
        encoder,
        featuresCreator, 
        kmeans]
)

model = pipeline.fit(births_train)
```


```python
test = model.transform(births_test)
test.select(['BIRTH_PLACE_VEC','features', 'prediction']).show()
```

    +---------------+--------------------+----------+
    |BIRTH_PLACE_VEC|            features|prediction|
    +---------------+--------------------+----------+
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,16...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,16...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         4|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         4|
    |  (9,[1],[1.0])|(24,[0,1,2,3,4,5,...|         4|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         4|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         0|
    |  (9,[1],[1.0])|(24,[0,1,6,7,8,9,...|         1|
    +---------------+--------------------+----------+
    only showing top 20 rows
    



```python
import pyspark.sql.functions as func
from pyspark.sql.functions import col

test.groupBy('prediction').agg({
    #         func.count('*').alias('count'),
    #         *[func.mean(c).alias(c)
    #          for c in test.columns if c=='MOTHER_HEIGHT_IN']
            '*': 'count',
            'MOTHER_HEIGHT_IN': 'avg'
                })\
            .withColumnRenamed("count(1)", "count")\
            .sort(col('count').desc())\
            .show()
```

    +----------+---------------------+-----+
    |prediction|avg(MOTHER_HEIGHT_IN)|count|
    +----------+---------------------+-----+
    |         4|    64.31597357170618|10292|
    |         0|    64.43472584856397| 2298|
    |         2|    67.69473684210526|  475|
    |         1|    83.91154791154791|  407|
    |         3|    66.64658634538152|  249|
    +----------+---------------------+-----+
    


In the field of NLP, problems such as topic extract rely on clustering to detect documents with __similar topics__. 


```python
text_data = spark.createDataFrame([
    ['''To make a computer do anything, you have to write a 
    computer program. To write a computer program, you have 
    to tell the computer, step by step, exactly what you want 
    it to do. The computer then "executes" the program, 
    following each step mechanically, to accomplish the end 
    goal. When you are telling the computer what to do, you 
    also get to choose how it's going to do it. That's where 
    computer algorithms come in. The algorithm is the basic 
    technique used to get the job done. Let's follow an 
    example to help get an understanding of the algorithm 
    concept.'''],
    ['''Laptop computers use batteries to run while not 
    connected to mains. When we overcharge or overheat 
    lithium ion batteries, the materials inside start to 
    break down and produce bubbles of oxygen, carbon dioxide, 
    and other gases. Pressure builds up, and the hot battery 
    swells from a rectangle into a pillow shape. Sometimes 
    the phone involved will operate afterwards. Other times 
    it will die. And occasionally—kapow! To see what's 
    happening inside the battery when it swells, the CLS team 
    used an x-ray technology called computed tomography.'''],
    ['''This technology describes a technique where touch 
    sensors can be placed around any side of a device 
    allowing for new input sources. The patent also notes 
    that physical buttons (such as the volume controls) could 
    be replaced by these embedded touch sensors. In essence 
    Apple could drop the current buttons and move towards 
    touch-enabled areas on the device for the existing UI. It 
    could also open up areas for new UI paradigms, such as 
    using the back of the smartphone for quick scrolling or 
    page turning.'''],
    ['''The National Park Service is a proud protector of 
    America’s lands. Preserving our land not only safeguards 
    the natural environment, but it also protects the 
    stories, cultures, and histories of our ancestors. As we 
    face the increasingly dire consequences of climate 
    change, it is imperative that we continue to expand 
    America’s protected lands under the oversight of the 
    National Park Service. Doing so combats climate change 
    and allows all American’s to visit, explore, and learn 
    from these treasured places for generations to come. It 
    is critical that President Obama acts swiftly to preserve 
    land that is at risk of external threats before the end 
    of his term as it has become blatantly clear that the 
    next administration will not hold the same value for our 
    environment over the next four years.'''],
    ['''The National Park Foundation, the official charitable 
    partner of the National Park Service, enriches America’s 
    national parks and programs through the support of 
    private citizens, park lovers, stewards of nature, 
    history enthusiasts, and wilderness adventurers. 
    Chartered by Congress in 1967, the Foundation grew out of 
    a legacy of park protection that began over a century 
    ago, when ordinary citizens took action to establish and 
    protect our national parks. Today, the National Park 
    Foundation carries on the tradition of early park 
    advocates, big thinkers, doers and dreamers—from John 
    Muir and Ansel Adams to President Theodore Roosevelt.'''],
    ['''Australia has over 500 national parks. Over 28 
    million hectares of land is designated as national 
    parkland, accounting for almost four per cent of 
    Australia's land areas. In addition, a further six per 
    cent of Australia is protected and includes state 
    forests, nature parks and conservation reserves.National 
    parks are usually large areas of land that are protected 
    because they have unspoilt landscapes and a diverse 
    number of native plants and animals. This means that 
    commercial activities such as farming are prohibited and 
    human activity is strictly monitored.''']
], ['documents'])
```

First, we will once again use the `RegexTokenizer` and the `StopWordsRemover` models.


```python
tokenizer = ft.RegexTokenizer(
    inputCol='documents', 
    outputCol='input_arr', 
    pattern='\s+|[,.\"]')

# By default, the parameter “pattern” (regex, default: "\\s+") is used as delimiters 
# to split the input text

stopwords_removed = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(), 
    outputCol='input_stop')

```


```python
stringIndexer = ft.CountVectorizer(
    inputCol=stopwords_removed.getOutputCol(), 
    outputCol="input_indexed")

tokenized = stopwords_removed \
    .transform(
        tokenizer\
            .transform(text_data)
    )
    
input_indexed=stringIndexer \
            .fit(tokenized)\
            .transform(tokenized)\
            .select('input_indexed')\
            .take(2)
for i in input_indexed:
    print(i)
```

    Row(input_indexed=SparseVector(262, {2: 7.0, 5: 1.0, 9: 3.0, 10: 3.0, 14: 3.0, 15: 1.0, 21: 1.0, 23: 2.0, 24: 2.0, 32: 1.0, 42: 1.0, 55: 1.0, 57: 1.0, 59: 1.0, 78: 1.0, 80: 1.0, 83: 1.0, 89: 1.0, 93: 1.0, 95: 1.0, 97: 1.0, 119: 1.0, 120: 1.0, 127: 1.0, 136: 1.0, 141: 1.0, 153: 1.0, 157: 1.0, 162: 1.0, 201: 1.0, 204: 1.0, 209: 1.0, 216: 1.0, 252: 1.0, 253: 1.0, 259: 1.0}))
    Row(input_indexed=SparseVector(262, {15: 1.0, 19: 2.0, 28: 1.0, 33: 2.0, 37: 2.0, 40: 2.0, 49: 1.0, 52: 1.0, 58: 1.0, 63: 1.0, 67: 1.0, 69: 1.0, 70: 1.0, 77: 1.0, 86: 1.0, 87: 1.0, 91: 1.0, 94: 1.0, 108: 1.0, 111: 1.0, 115: 1.0, 125: 1.0, 146: 1.0, 147: 1.0, 152: 1.0, 156: 1.0, 164: 1.0, 169: 1.0, 170: 1.0, 174: 1.0, 179: 1.0, 182: 1.0, 185: 1.0, 189: 1.0, 195: 1.0, 198: 1.0, 208: 1.0, 214: 1.0, 219: 1.0, 220: 1.0, 224: 1.0, 227: 1.0, 228: 1.0, 233: 1.0, 236: 1.0, 244: 1.0, 248: 1.0, 249: 1.0}))



```python
text_transformed=stringIndexer.fit(tokenized).transform(tokenized).select('input_indexed')
for i in text_transformed.take(5):
    print(i)
    print('\n')
```

    Row(input_indexed=SparseVector(262, {2: 7.0, 6: 1.0, 10: 3.0, 13: 3.0, 14: 3.0, 19: 1.0, 24: 1.0, 30: 1.0, 34: 1.0, 40: 2.0, 41: 2.0, 44: 1.0, 50: 1.0, 60: 1.0, 65: 1.0, 92: 1.0, 96: 1.0, 104: 1.0, 113: 1.0, 117: 1.0, 121: 1.0, 143: 1.0, 145: 1.0, 147: 1.0, 149: 1.0, 164: 1.0, 182: 1.0, 183: 1.0, 188: 1.0, 191: 1.0, 223: 1.0, 224: 1.0, 232: 1.0, 247: 1.0, 250: 1.0, 256: 1.0}))
    
    
    Row(input_indexed=SparseVector(262, {17: 2.0, 21: 2.0, 24: 1.0, 29: 2.0, 36: 1.0, 38: 2.0, 48: 1.0, 49: 1.0, 63: 1.0, 69: 1.0, 70: 1.0, 74: 1.0, 75: 1.0, 77: 1.0, 78: 1.0, 79: 1.0, 88: 1.0, 89: 1.0, 93: 1.0, 94: 1.0, 99: 1.0, 102: 1.0, 110: 1.0, 111: 1.0, 119: 1.0, 126: 1.0, 135: 1.0, 138: 1.0, 141: 1.0, 144: 1.0, 155: 1.0, 163: 1.0, 173: 1.0, 178: 1.0, 179: 1.0, 192: 1.0, 196: 1.0, 199: 1.0, 203: 1.0, 206: 1.0, 209: 1.0, 213: 1.0, 235: 1.0, 239: 1.0, 243: 1.0, 244: 1.0, 255: 1.0, 260: 1.0}))
    
    
    Row(input_indexed=SparseVector(262, {5: 2.0, 6: 2.0, 12: 3.0, 25: 2.0, 30: 1.0, 31: 2.0, 32: 2.0, 33: 2.0, 36: 1.0, 39: 2.0, 43: 2.0, 61: 1.0, 67: 1.0, 68: 1.0, 71: 1.0, 73: 1.0, 83: 1.0, 85: 1.0, 87: 1.0, 101: 1.0, 107: 1.0, 120: 1.0, 123: 1.0, 125: 1.0, 128: 1.0, 129: 1.0, 154: 1.0, 157: 1.0, 165: 1.0, 174: 1.0, 198: 1.0, 200: 1.0, 207: 1.0, 210: 1.0, 217: 1.0, 225: 1.0, 233: 1.0, 234: 1.0, 240: 1.0, 241: 1.0, 248: 1.0, 253: 1.0, 259: 1.0}))
    
    
    Row(input_indexed=SparseVector(262, {0: 2.0, 1: 2.0, 3: 2.0, 6: 1.0, 8: 1.0, 9: 2.0, 11: 2.0, 15: 2.0, 16: 1.0, 18: 2.0, 19: 1.0, 22: 2.0, 23: 2.0, 27: 2.0, 28: 1.0, 34: 1.0, 45: 1.0, 55: 1.0, 59: 1.0, 62: 1.0, 81: 1.0, 86: 1.0, 91: 1.0, 103: 1.0, 105: 1.0, 108: 1.0, 109: 1.0, 114: 1.0, 115: 1.0, 118: 1.0, 124: 1.0, 127: 1.0, 132: 1.0, 133: 1.0, 136: 1.0, 140: 1.0, 142: 1.0, 148: 1.0, 150: 1.0, 151: 1.0, 152: 1.0, 153: 1.0, 156: 1.0, 169: 1.0, 175: 1.0, 177: 1.0, 190: 1.0, 197: 1.0, 205: 1.0, 208: 1.0, 219: 1.0, 222: 1.0, 227: 1.0, 236: 1.0, 237: 1.0, 238: 1.0, 242: 1.0, 254: 1.0, 261: 1.0}))
    
    
    Row(input_indexed=SparseVector(262, {0: 5.0, 1: 6.0, 4: 2.0, 7: 3.0, 9: 1.0, 11: 1.0, 20: 2.0, 28: 1.0, 42: 1.0, 54: 1.0, 58: 1.0, 66: 1.0, 80: 1.0, 82: 1.0, 95: 1.0, 97: 1.0, 98: 1.0, 106: 1.0, 112: 1.0, 130: 1.0, 131: 1.0, 159: 1.0, 160: 1.0, 161: 1.0, 162: 1.0, 166: 1.0, 168: 1.0, 170: 1.0, 171: 1.0, 172: 1.0, 180: 1.0, 181: 1.0, 184: 1.0, 185: 1.0, 186: 1.0, 187: 1.0, 189: 1.0, 202: 1.0, 212: 1.0, 214: 1.0, 215: 1.0, 216: 1.0, 218: 1.0, 221: 1.0, 228: 1.0, 229: 1.0, 230: 1.0, 231: 1.0, 245: 1.0, 251: 1.0, 257: 1.0}))
    
    


> how does CountVectorizer works?


```python
from pyspark.ml.feature import CountVectorizer

# Input data: Each row is a bag of words with a ID.
df = spark.createDataFrame([
    (0, "a b c d d d e".split(" ")),
    (1, "a b b c a u o r".split(" "))
], ["id", "words"])

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=10, minDF=1.0)

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)
```

    +---+------------------------+-------------------------------------------+
    |id |words                   |features                                   |
    +---+------------------------+-------------------------------------------+
    |0  |[a, b, c, d, d, d, e]   |(8,[0,1,2,3,4],[3.0,1.0,1.0,1.0,1.0])      |
    |1  |[a, b, b, c, a, u, o, r]|(8,[1,2,3,5,6,7],[2.0,2.0,1.0,1.0,1.0,1.0])|
    +---+------------------------+-------------------------------------------+
    


feature 1:
- a : 0 appears once
- b : 1 appears once
- d : 2 appears 3 times
- c : 3 appears once
- e : 6 appears once

similarly with feature 2

`LDA` model - the Latent Dirichlet Allocation model - to extract the topics.


```python
clustering = cluster.LDA(k=2, 
                         optimizer='online', 
                         featuresCol=stringIndexer.getOutputCol())

```


```python
pipeline = Pipeline(stages=[
        tokenizer, 
        stopwords_removed,
        stringIndexer, 
        clustering]
)
```

Let's see if we have properly uncovered the topics.


```python
topics = pipeline \
    .fit(text_data) \
    .transform(text_data)
topics.show()
```

    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |           documents|           input_arr|          input_stop|       input_indexed|   topicDistribution|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |To make a compute...|[to, make, a, com...|[make, computer, ...|(262,[2,5,8,11,13...|[0.03498938221884...|
    |Laptop computers ...|[laptop, computer...|[laptop, computer...|(262,[24,28,32,37...|[0.02270266261248...|
    |This technology d...|[this, technology...|[technology, desc...|(262,[5,6,7,17,18...|[0.02514214805729...|
    |The National Park...|[the, national, p...|[national, park, ...|(262,[0,1,4,5,9,1...|[0.38569930399852...|
    |The National Park...|[the, national, p...|[national, park, ...|(262,[0,1,3,9,12,...|[0.99174634339828...|
    |Australia has ove...|[australia, has, ...|[australia, 500, ...|(262,[0,3,4,6,10,...|[0.98909763313790...|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    


### Regression

here we try to predict the `MOTHER_WEIGHT_GAIN`.


```python
features = ['MOTHER_AGE_YEARS','MOTHER_HEIGHT_IN',
            'MOTHER_PRE_WEIGHT','DIABETES_PRE',
            'DIABETES_GEST','HYP_TENS_PRE', 
            'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM',
            'CIG_BEFORE','CIG_1_TRI', 'CIG_2_TRI', 
            'CIG_3_TRI'
           ]
```

First, we will collate all the features together and use the `ChiSqSelector` to select only the top 6 most important features.


```python
featuresCreator = ft.VectorAssembler(
    inputCols=[col for col in features], outputCol='features')

selector = ft.ChiSqSelector(
    numTopFeatures=6,
    outputCol="selectedFeatures",
    labelCol='MOTHER_WEIGHT_GAIN')

births_train.select('MOTHER_WEIGHT_GAIN').show()
set(
    births_train.select('MOTHER_WEIGHT_GAIN').rdd.map(
        lambda x: isinstance(x[0], int)).collect()
    )
```

    +------------------+
    |MOTHER_WEIGHT_GAIN|
    +------------------+
    |                 0|
    |                 7|
    |                 8|
    |                22|
    |                18|
    |                 2|
    |                15|
    |                 0|
    |                26|
    |                34|
    |                 9|
    |                 7|
    |                 0|
    |                 2|
    |                 0|
    |                 0|
    |                14|
    |                 4|
    |                 7|
    |                 6|
    +------------------+
    only showing top 20 rows
    





    {True}



In order to predict the weight gain we will use the gradient boosted trees regressor.


```python
import pyspark.ml.regression as reg

regressor = reg.GBTRegressor(
    maxIter=15, 
    maxDepth=3,
    labelCol='MOTHER_WEIGHT_GAIN')
```

Finally, again, we put it all together into a `Pipeline`.


```python
pipeline = Pipeline(stages=[
        featuresCreator, 
        selector,
        regressor])

weightGain = pipeline.fit(births_train)
```

Having created the `weightGain` model, let's see if it performs well on our testing data.


```python
evaluator = ev.RegressionEvaluator(
    labelCol='MOTHER_WEIGHT_GAIN',
    predictionCol="prediction")

print(evaluator.evaluate(
     weightGain.transform(births_test), 
    {evaluator.metricName: 'r2'}))
```

    0.4885528891241073



```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

data = spark.read.format("libsvm").load(
    "/home/karen/Downloads/data/sample_libsvm_data.txt")

featureIndexer = VectorIndexer(
    inputCol="features", 
    outputCol="indexedFeatures",
    maxCategories=4
    ).fit(data)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

pipeline = Pipeline(stages=[featureIndexer, gbt])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.select("prediction", "label", "features").show(10)

predictions.rdd.saveAsTextFile("predictions_file")
```

    +----------+-----+--------------------+
    |prediction|label|            features|
    +----------+-----+--------------------+
    |       0.0|  0.0|(692,[122,123,148...|
    |       0.0|  0.0|(692,[123,124,125...|
    |       0.0|  0.0|(692,[123,124,125...|
    |       0.0|  0.0|(692,[124,125,126...|
    |       0.0|  0.0|(692,[124,125,126...|
    |       0.0|  0.0|(692,[124,125,126...|
    |       0.0|  0.0|(692,[124,125,126...|
    |       0.0|  0.0|(692,[124,125,126...|
    |       0.0|  0.0|(692,[126,127,128...|
    |       0.0|  0.0|(692,[126,127,128...|
    +----------+-----+--------------------+
    only showing top 10 rows
    



```python
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R square on test data = %g" % r2)

gbtModel = model.stages
print(gbtModel)
```

    R square on test data = 1
    [VectorIndexer_48829a94bfb47a09e61f, GBTRegressionModel (uid=GBTRegressor_49d09a8accd090f5c740) with 10 trees]

