
# Prepare and understand data for modeling


```python
spark
```




    <pyspark.sql.session.SparkSession object at 0x7fdc21129908>




```python
df = spark.createDataFrame([
        (1, 144.5, 5.9, 33, 'M'),
        (2, 167.2, 5.4, 45, 'M'),
        (3, 124.1, 5.2, 23, 'F'),
        (4, 144.5, 5.9, 33, 'M'),
        (5, 133.2, 5.7, 54, 'F'),
        (3, 124.1, 5.2, 23, 'F'),
        (5, 129.2, 5.3, 42, 'M'),
    ], ['id', 'weight', 'height', 'age', 'gender'])
```

Check for duplicates.


```python
df.count()
```




    7




```python
df.distinct().count()
```




    6




```python
df = df.dropDuplicates()
df.show()
```

    +---+------+------+---+------+
    | id|weight|height|age|gender|
    +---+------+------+---+------+
    |  5| 133.2|   5.7| 54|     F|
    |  5| 129.2|   5.3| 42|     M|
    |  1| 144.5|   5.9| 33|     M|
    |  4| 144.5|   5.9| 33|     M|
    |  2| 167.2|   5.4| 45|     M|
    |  3| 124.1|   5.2| 23|     F|
    +---+------+------+---+------+
    


Let's confirm.


```python
print('Count of ids: {0}'.format(df.count()))
print('Count of distinct ids: {0}'.format(df.select([c for c in df.columns if c != 'id']).distinct().count()))
```

    Count of ids: 6
    Count of distinct ids: 5


We still have one more duplicate. We will use the `.dropDuplicates(...)` but add the `subset` parameter.


```python
df = df.dropDuplicates(subset=[c for c in df.columns if c != 'id'])
df.show()
```

    +---+------+------+---+------+
    | id|weight|height|age|gender|
    +---+------+------+---+------+
    |  5| 133.2|   5.7| 54|     F|
    |  1| 144.5|   5.9| 33|     M|
    |  2| 167.2|   5.4| 45|     M|
    |  3| 124.1|   5.2| 23|     F|
    |  5| 129.2|   5.3| 42|     M|
    +---+------+------+---+------+
    


To calculate the total and distinct number of IDs in one step we can use the `.agg(...)` method.


```python
import pyspark.sql.functions as fn

df.agg(
    fn.count('id').alias('count'),
    fn.countDistinct('id').alias('distinct')
).show()
```

    +-----+--------+
    |count|distinct|
    +-----+--------+
    |    5|       4|
    +-----+--------+
    


Give each row a unique ID. 


```python
df.withColumn('new_id', fn.monotonically_increasing_id()).show()
```

    +---+------+------+---+------+-------------+
    | id|weight|height|age|gender|       new_id|
    +---+------+------+---+------+-------------+
    |  5| 133.2|   5.7| 54|     F|  25769803776|
    |  1| 144.5|   5.9| 33|     M| 171798691840|
    |  2| 167.2|   5.4| 45|     M| 592705486848|
    |  3| 124.1|   5.2| 23|     F|1236950581248|
    |  5| 129.2|   5.3| 42|     M|1365799600128|
    +---+------+------+---+------+-------------+
    


### Missing observations


```python
df_miss = spark.createDataFrame([
        (1, 143.5, 5.6, 28,   'M',  100000),
        (2, 167.2, 5.4, 45,   'M',  None),
        (3, None , 5.2, None, None, None),
        (4, 144.5, 5.9, 33,   'M',  None),
        (5, 133.2, 5.7, 54,   'F',  None),
        (6, 124.1, 5.2, None, 'F',  None),
        (7, 129.2, 5.3, 42,   'M',  76000),
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
```


```python
df_miss.rdd.map(
    lambda row: (row['id'], sum([c == None for c in row]))
).collect()
```




    [(1, 0), (2, 1), (3, 4), (4, 1), (5, 1), (6, 2), (7, 0)]



> To drop the observation altogether or impute some of the observations?


```python
df_miss.where('id == 3').show()
```

    +---+------+------+----+------+------+
    | id|weight|height| age|gender|income|
    +---+------+------+----+------+------+
    |  3|  null|   5.2|null|  null|  null|
    +---+------+------+----+------+------+
    


What is the percentage of missing observations we see in each column?


```python
df_miss.agg(*[
    (1 - (fn.count(c) / fn.count('*'))).alias(c + '_missing')
    for c in df_miss.columns
]).show()
```

    +----------+------------------+--------------+------------------+------------------+------------------+
    |id_missing|    weight_missing|height_missing|       age_missing|    gender_missing|    income_missing|
    +----------+------------------+--------------+------------------+------------------+------------------+
    |       0.0|0.1428571428571429|           0.0|0.2857142857142857|0.1428571428571429|0.7142857142857143|
    +----------+------------------+--------------+------------------+------------------+------------------+
    


We will drop the `'income'` feature as most of its values are missing.


```python
df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])
df_miss_no_income.show()
```

    +---+------+------+----+------+
    | id|weight|height| age|gender|
    +---+------+------+----+------+
    |  1| 143.5|   5.6|  28|     M|
    |  2| 167.2|   5.4|  45|     M|
    |  3|  null|   5.2|null|  null|
    |  4| 144.5|   5.9|  33|     M|
    |  5| 133.2|   5.7|  54|     F|
    |  6| 124.1|   5.2|null|     F|
    |  7| 129.2|   5.3|  42|     M|
    +---+------+------+----+------+
    



```python
df_miss_no_income.dropna(thresh=3).show()
```

    +---+------+------+----+------+
    | id|weight|height| age|gender|
    +---+------+------+----+------+
    |  1| 143.5|   5.6|  28|     M|
    |  2| 167.2|   5.4|  45|     M|
    |  4| 144.5|   5.9|  33|     M|
    |  5| 133.2|   5.7|  54|     F|
    |  6| 124.1|   5.2|null|     F|
    |  7| 129.2|   5.3|  42|     M|
    +---+------+------+----+------+
    


To impute a mean, median or other *calculated* value you need to first calculate the value, create a dict with such values, and then pass it to the `.fillna(...)` method.


```python
means = df_miss_no_income.agg(
    *[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']
).toPandas().to_dict('records')[0]

means
```




    {'id': 4.0, 'height': 5.471428571428572, 'age': 40.4, 'weight': 140.28333333333333}




```python
means['gender'] = 'missing' # fill the null space in 'gender' as 'missing'
df_miss_no_income.fillna(means).show()
```

    +---+------------------+------+---+-------+
    | id|            weight|height|age| gender|
    +---+------------------+------+---+-------+
    |  1|             143.5|   5.6| 28|      M|
    |  2|             167.2|   5.4| 45|      M|
    |  3|140.28333333333333|   5.2| 40|missing|
    |  4|             144.5|   5.9| 33|      M|
    |  5|             133.2|   5.7| 54|      F|
    |  6|             124.1|   5.2| 40|      F|
    |  7|             129.2|   5.3| 42|      M|
    +---+------------------+------+---+-------+
    


### Outliers

Consider another simple example.


```python
df_outliers = spark.createDataFrame([
        (1, 143.5, 5.3, 28),
        (2, 154.2, 5.5, 45),
        (3, 342.3, 5.1, 99),
        (4, 144.5, 5.5, 33),
        (5, 133.2, 5.4, 54),
        (6, 124.1, 5.1, 21),
        (7, 129.2, 5.3, 42),
    ], ['id', 'weight', 'height', 'age'])
```

First, we calculate the lower and upper *cut off* points for each feature.


```python
cols = ['weight', 'height', 'age']
bounds = {}

for col in cols:
    quantiles = df_outliers.approxQuantile(col, [0.25, 0.75], 0.05)
    IQR = quantiles[1] - quantiles[0]
    bounds[col] = [quantiles[0] - 1.5 * IQR, quantiles[1] + 1.5 * IQR]
```

The `bounds` dictionary holds the lower and upper bounds for each feature. 


```python
bounds
```




    {'age': [-11.0, 93.0], 'height': [4.499999999999999, 6.1000000000000005], 'weight': [91.69999999999999, 191.7]}



To flag our outliers.


```python
outliers = df_outliers.select(*['id'] + [
    (
        (df_outliers[c] < bounds[c][0]) | 
        (df_outliers[c] > bounds[c][1])
    ).alias(c + '_outliers') for c in cols
])
outliers.show()
```

    +---+---------------+---------------+------------+
    | id|weight_outliers|height_outliers|age_outliers|
    +---+---------------+---------------+------------+
    |  1|          false|          false|       false|
    |  2|          false|          false|       false|
    |  3|           true|          false|        true|
    |  4|          false|          false|       false|
    |  5|          false|          false|       false|
    |  6|          false|          false|       false|
    |  7|          false|          false|       false|
    +---+---------------+---------------+------------+
    


Now join `outliers` and `df_outliers` table to pick out those outlier numbers


```python
df_outliers = df_outliers.join(outliers, on='id')
df_outliers.filter('weight_outliers').select('id', 'weight').show()
df_outliers.filter('age_outliers').select('id', 'age').show()
```

    +---+------+
    | id|weight|
    +---+------+
    |  3| 342.3|
    +---+------+
    
    +---+---+
    | id|age|
    +---+---+
    |  3| 99|
    +---+---+
    


## Understand your data

### Descriptive statistics


```python
import pyspark.sql.types as typ
```


```python
fraud = sc.textFile('/home/karen/Downloads/data/ccFraud.csv')
header = fraud.first()

fraud = fraud \
    .filter(lambda row: row != header) \
    .map(lambda row: [int(elem) for elem in row.split(',')])
fraud.take(1)
```




    [[1, 1, 35, 1, 3000, 4, 14, 2, 0]]




```python
fields = [
    *[
        typ.StructField(h[1:-1], typ.IntegerType(), True)
        for h in header.split(',')
    ]
]

schema = typ.StructType(fields)
```


```python
fraud_df = spark.createDataFrame(fraud, schema)
```


```python
fraud_df.printSchema()
```

    root
     |-- custID: integer (nullable = true)
     |-- gender: integer (nullable = true)
     |-- state: integer (nullable = true)
     |-- cardholder: integer (nullable = true)
     |-- balance: integer (nullable = true)
     |-- numTrans: integer (nullable = true)
     |-- numIntlTrans: integer (nullable = true)
     |-- creditLine: integer (nullable = true)
     |-- fraudRisk: integer (nullable = true)
    


#### For categorical columns we will count the frequencies of their values using `.groupby(...)` method.


```python
fraud_df.groupby('gender').count().show()
```

    +------+-------+
    |gender|  count|
    +------+-------+
    |     1|6178231|
    |     2|3821769|
    +------+-------+
    


#### For the truly numerical features we can use the `.describe()` method.


```python
numerical = ['balance', 'numTrans', 'numIntlTrans']
```


```python
desc = fraud_df.describe(numerical)
desc.show()
```

    +-------+-----------------+------------------+-----------------+
    |summary|          balance|          numTrans|     numIntlTrans|
    +-------+-----------------+------------------+-----------------+
    |  count|         10000000|          10000000|         10000000|
    |   mean|     4109.9199193|        28.9351871|        4.0471899|
    | stddev|3996.847309737258|26.553781024523122|8.602970115863904|
    |    min|                0|                 0|                0|
    |    max|            41485|               100|               60|
    +-------+-----------------+------------------+-----------------+
    



```python
fraud_df.agg({'balance': 'skewness'}).show()
```

    +------------------+
    | skewness(balance)|
    +------------------+
    |1.1818315552993839|
    +------------------+
    


### Correlations


```python
fraud_df.corr('balance', 'numTrans')
```




    0.0004452314017265387




```python
n_numerical = len(numerical)

corr = []

for i in range(0, n_numerical):
    temp = [None] * i
    
    for j in range(i, n_numerical):
        temp.append(fraud_df.corr(numerical[i], numerical[j]))
    corr.append(temp)
    
corr
```




    [[1.0, 0.0004452314017265387, 0.00027139913398178744], [None, 1.0, -0.00028057128198165544], [None, None, 1.0]]



### Visualization


```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

### Histograms

Aggreagate the data in workers and return aggregated list of cut-off points and counts in each bin of the histogram to the driver.


```python
hists = fraud_df.select('balance').rdd.flatMap(lambda row: row).histogram(20)
```


```python
data = {
    'bins': hists[0][:-1],
    'freq': hists[1]
}

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1)
ax.bar(data['bins'], data['freq'], width=2000)
ax.set_title('Histogram of \'balance\'')

plt.savefig('balance.png', dpi=300)
```


![png](output_57_0.png)



```python
data_driver = {'obs': fraud_df.select('balance').rdd.flatMap(lambda row: row).collect()}
```


```python
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1)

ax.hist(data_driver['obs'], bins=20)
ax.set_title('Histogram of \'balance\' using .hist()')

plt.savefig('balance2.png', dpi=300)
```


![png](output_59_0.png)


### Interactions between features

In this example we will sample our fraud dataset at 1% given gender as strata.


```python
data_sample = fraud_df.sampleBy('gender', {1: 0.0002, 2: 0.0002}).select(numerical)
data_sample.show()
```

    +-------+--------+------------+
    |balance|numTrans|numIntlTrans|
    +-------+--------+------------+
    |      0|      21|           3|
    |      0|      19|          23|
    |   3000|       8|          17|
    |    939|       9|           5|
    |   4000|      19|          10|
    |   5999|     100|           5|
    |  16000|      14|           0|
    |   6556|      79|           0|
    |   1458|      14|           0|
    |   6073|       3|          37|
    |   4000|      10|           0|
    |   6000|      12|           0|
    |      0|      10|           0|
    |   2766|      99|           0|
    |   4000|      37|           0|
    |    157|       8|           8|
    |   5000|      62|           3|
    |      0|     100|           0|
    |  10076|      72|           0|
    |  11675|      36|           4|
    +-------+--------+------------+
    only showing top 20 rows
    

