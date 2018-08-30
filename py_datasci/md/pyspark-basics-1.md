
_Main operations:_ `sc`, `spark`, `write.csv`, `read.csv`, `select`, `count`, `dtypes`, `schema`/`inferSchema`, `take`, `show`, `withColumnRenamed`, `columns`, `describe`, `coalesce`


```python
sc
```




    <pyspark.context.SparkContext object at 0x7f12ae6bfe50>




```python
spark
```




    <pyspark.sql.session.SparkSession object at 0x7f12ae649ad0>



# Loading From CSV



```python
df = spark.read.csv('/home/karen/workspace/sparkPacktReading/data/u.item', header=False, inferSchema=True, sep='|')
```


```python
df.show()
```

    +---+--------------------+-----------+----+--------------------+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
    |_c0|                 _c1|        _c2| _c3|                 _c4|_c5|_c6|_c7|_c8|_c9|_c10|_c11|_c12|_c13|_c14|_c15|_c16|_c17|_c18|_c19|_c20|_c21|_c22|_c23|
    +---+--------------------+-----------+----+--------------------+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
    |  1|    Toy Story (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  1|  1|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    |  2|    GoldenEye (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  1|  1|  0|  0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|
    |  3|   Four Rooms (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|
    |  4|   Get Shorty (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  1|  0|  0|  0|   1|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    |  5|      Copycat (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   1|   0|   1|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|
    |  6|Shanghai Triad (Y...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    |  7|Twelve Monkeys (1...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   1|   0|   0|   0|
    |  8|         Babe (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  1|   1|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    |  9|Dead Man Walking ...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    | 10|  Richard III (1995)|22-Jan-1996|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|
    | 11|Seven (Se7en) (1995)|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|
    | 12|Usual Suspects, T...|14-Aug-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|
    | 13|Mighty Aphrodite ...|30-Oct-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    | 14|  Postino, Il (1994)|01-Jan-1994|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   1|   0|   0|   0|   0|
    | 15|Mr. Holland's Opu...|29-Jan-1996|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    | 16|French Twist (Gaz...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   1|   0|   0|   0|   0|
    | 17|From Dusk Till Da...|05-Feb-1996|null|http://us.imdb.co...|  0|  1|  0|  0|  0|   1|   1|   0|   0|   0|   0|   1|   0|   0|   0|   0|   1|   0|   0|
    | 18|White Balloon, Th...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    | 19|Antonia's Line (1...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
    | 20|Angels and Insect...|01-Jan-1995|null|http://us.imdb.co...|  0|  0|  0|  0|  0|   0|   0|   0|   1|   0|   0|   0|   0|   0|   1|   0|   0|   0|   0|
    +---+--------------------+-----------+----+--------------------+---+---+---+---+---+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
    only showing top 20 rows
    



```python
from pyspark.sql.types import DateType, TimestampType, IntegerType, FloatType, LongType, DoubleType
from pyspark.sql.types import StructType, StructField

custom_schema = StructType([StructField('_c0', IntegerType(), True),
                           StructField('_c1', StringType(), True),
                           StructField('_c2', StringType(), True),
                           StructField('_c3', StringType(), True),
                           StructField('_c4', StringType(), True),
                           StructField('_c5', IntegerType(), True),
                           ...
                           StructField('_c23', IntegerType(), True)])
                           
df = spark.read.csv('/home/karen/workspace/sparkPacktReading/data/u.item', header=False, schema=custom_schema, sep='|')

```




    
    from pyspark.sql.types import DateType, TimestampType, IntegerType, FloatType, LongType, DoubleType
    from pyspark.sql.types import StructType, StructField
    custom_schema = StructType([StructField('_c0', DateType(), True),
                               StructField('_c1', StringType(), True),
                               StructField('_c2', DoubleType(), True),
                               StructField('_c3', DoubleType(), True),
                               StructField('_c4', DoubleType(), True),
                               StructField('_c5', IntegerType(), True),
                               ...
                               StructField('_c27', StringType(), True)])
    df = spark.read.csv('s3://ui-spark-social-science-public/data/Performance_2015Q1.txt', header=False, schema=custom_schema, sep='|')





```python
df.count()
```




    1682




```python
df.dtypes
```




    [('_c0', 'int'), ('_c1', 'string'), ('_c2', 'string'), ('_c3', 'string'), ('_c4', 'string'), ('_c5', 'int'), ('_c6', 'int'), ('_c7', 'int'), ('_c8', 'int'), ('_c9', 'int'), ('_c10', 'int'), ('_c11', 'int'), ('_c12', 'int'), ('_c13', 'int'), ('_c14', 'int'), ('_c15', 'int'), ('_c16', 'int'), ('_c17', 'int'), ('_c18', 'int'), ('_c19', 'int'), ('_c20', 'int'), ('_c21', 'int'), ('_c22', 'int'), ('_c23', 'int')]



For each pairing (a `tuple` object in Python, denoted by the parentheses), the first entry is the column name and the second is the dtype.  Notice that this data has no headers with it (we specified `headers=False` when we loaded it), so Spark used its default naming convention of `_c0, _c1, ... _cn`.  We'll makes some changes to that in a minute.

Take a peak at five rows:


```python
df.take(5)
```




    [Row(_c0=1, _c1='Toy Story (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)', _c5=0, _c6=0, _c7=0, _c8=1, _c9=1, _c10=1, _c11=0, _c12=0, _c13=0, _c14=0, _c15=0, _c16=0, _c17=0, _c18=0, _c19=0, _c20=0, _c21=0, _c22=0, _c23=0), Row(_c0=2, _c1='GoldenEye (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?GoldenEye%20(1995)', _c5=0, _c6=1, _c7=1, _c8=0, _c9=0, _c10=0, _c11=0, _c12=0, _c13=0, _c14=0, _c15=0, _c16=0, _c17=0, _c18=0, _c19=0, _c20=0, _c21=1, _c22=0, _c23=0), Row(_c0=3, _c1='Four Rooms (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)', _c5=0, _c6=0, _c7=0, _c8=0, _c9=0, _c10=0, _c11=0, _c12=0, _c13=0, _c14=0, _c15=0, _c16=0, _c17=0, _c18=0, _c19=0, _c20=0, _c21=1, _c22=0, _c23=0), Row(_c0=4, _c1='Get Shorty (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995)', _c5=0, _c6=1, _c7=0, _c8=0, _c9=0, _c10=1, _c11=0, _c12=0, _c13=1, _c14=0, _c15=0, _c16=0, _c17=0, _c18=0, _c19=0, _c20=0, _c21=0, _c22=0, _c23=0), Row(_c0=5, _c1='Copycat (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?Copycat%20(1995)', _c5=0, _c6=0, _c7=0, _c8=0, _c9=0, _c10=0, _c11=1, _c12=0, _c13=1, _c14=0, _c15=0, _c16=0, _c17=0, _c18=0, _c19=0, _c20=0, _c21=1, _c22=0, _c23=0)]




# Selecting & Renaming Columns


```python
df_lim = df.select('_c0','_c1','_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13')
df_lim.take(1)
```




    [Row(_c0=1, _c1='Toy Story (1995)', _c2='01-Jan-1995', _c3=None, _c4='http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)', _c5=0, _c6=0, _c7=0, _c8=1, _c9=1, _c10=1, _c11=0, _c12=0, _c13=0)]



rename columns one at a time, or several at a time:


```python
df_lim = df_lim.withColumnRenamed('_c2','release_date').withColumnRenamed('_c4','imdb').withColumnRenamed('_c1', 'movie_name')
df_lim
```




    DataFrame[_c0: int, movie_name: string, release_date: string, _c3: string, imdb: string, _c5: int, _c6: int, _c7: int, _c8: int, _c9: int, _c10: int, _c11: int, _c12: int, _c13: int]




```python
df_lim.take(1)
```




    [Row(_c0=1, movie_name='Toy Story (1995)', release_date='01-Jan-1995', _c3=None, imdb='http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)', _c5=0, _c6=0, _c7=0, _c8=1, _c9=1, _c10=1, _c11=0, _c12=0, _c13=0)]




```python
df_lim.columns
```




    ['_c0', 'movie_name', 'release_date', '_c3', 'imdb', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13']



# Describe


```python
df_described = df_lim.select('movie_name', 'release_date', 'imdb')
df_described.show()
```

    +--------------------+------------+--------------------+
    |          movie_name|release_date|                imdb|
    +--------------------+------------+--------------------+
    |    Toy Story (1995)| 01-Jan-1995|http://us.imdb.co...|
    |    GoldenEye (1995)| 01-Jan-1995|http://us.imdb.co...|
    |   Four Rooms (1995)| 01-Jan-1995|http://us.imdb.co...|
    |   Get Shorty (1995)| 01-Jan-1995|http://us.imdb.co...|
    |      Copycat (1995)| 01-Jan-1995|http://us.imdb.co...|
    |Shanghai Triad (Y...| 01-Jan-1995|http://us.imdb.co...|
    |Twelve Monkeys (1...| 01-Jan-1995|http://us.imdb.co...|
    |         Babe (1995)| 01-Jan-1995|http://us.imdb.co...|
    |Dead Man Walking ...| 01-Jan-1995|http://us.imdb.co...|
    |  Richard III (1995)| 22-Jan-1996|http://us.imdb.co...|
    |Seven (Se7en) (1995)| 01-Jan-1995|http://us.imdb.co...|
    |Usual Suspects, T...| 14-Aug-1995|http://us.imdb.co...|
    |Mighty Aphrodite ...| 30-Oct-1995|http://us.imdb.co...|
    |  Postino, Il (1994)| 01-Jan-1994|http://us.imdb.co...|
    |Mr. Holland's Opu...| 29-Jan-1996|http://us.imdb.co...|
    |French Twist (Gaz...| 01-Jan-1995|http://us.imdb.co...|
    |From Dusk Till Da...| 05-Feb-1996|http://us.imdb.co...|
    |White Balloon, Th...| 01-Jan-1995|http://us.imdb.co...|
    |Antonia's Line (1...| 01-Jan-1995|http://us.imdb.co...|
    |Angels and Insect...| 01-Jan-1995|http://us.imdb.co...|
    +--------------------+------------+--------------------+
    only showing top 20 rows
    


# Writing to S3


```python
df_described.write
    .format('<local file path>')
    .option("header", "true")
    .save('s3://<file path on S3>')
```


```python
df_described.where( df_described['release_date'].isNull() ).count()
```




    1




```python
def count_nulls(df):
    null_counts = []          
    # here: 
    # col[0]: cname
    # col[1]: dtype
    for col in df.dtypes:
        if col[1] != 'string':
            null_counts.append(tuple([col[0], df.where(df[col[0]].isNull()).count()]))
    return null_counts

null_counts = count_nulls(df)
null_counts
```




    [('_c0', 0), ('_c5', 0), ('_c6', 0), ('_c7', 0), ('_c8', 0), ('_c9', 0), ('_c10', 0), ('_c11', 0), ('_c12', 0), ('_c13', 0), ('_c14', 0), ('_c15', 0), ('_c16', 0), ('_c17', 0), ('_c18', 0), ('_c19', 0), ('_c20', 0), ('_c21', 0), ('_c22', 0), ('_c23', 0)]




```python
df_drops = df.dropna(how='all', subset=['_c5', '_c12', '_c23'])
df_drops.count()
```




    1682




```python
df.count()
```




    1682




```python
df_drops2 = df.dropna(thresh=2, subset=['_c4', '_c12', '_c23'])
```


```python
df_fill = df.fillna(0, subset=['_c12'])
```

### Moving Average Imputation


```python
df_diamonds = spark.read.csv('/home/karen/Downloads/data/diamonds.csv', 
                    inferSchema=True, header=True, sep=',', nullValue='')
df_diamonds.show()
```

    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    |carat|      cut|color|clarity|depth|table|price|   x|   y|   z|
    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    | 0.23|    Ideal|    E|    SI2| 61.5| 55.0|  326|3.95|3.98|2.43|
    | 0.21|  Premium|    E|    SI1| 59.8| 61.0|  326|3.89|3.84|2.31|
    | 0.23|     Good|    E|    VS1| 56.9| 65.0|  327|4.05|4.07|2.31|
    | 0.29|  Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|
    | 0.31|     Good|    J|    SI2| 63.3| 58.0|  335|4.34|4.35|2.75|
    | 0.24|Very Good|    J|   VVS2| 62.8| 57.0|  336|3.94|3.96|2.48|
    | 0.24|Very Good|    I|   VVS1| 62.3| 57.0|  336|3.95|3.98|2.47|
    | 0.26|Very Good|    H|    SI1| 61.9| 55.0|  337|4.07|4.11|2.53|
    | 0.22|     Fair|    E|    VS2| 65.1| 61.0|  337|3.87|3.78|2.49|
    | 0.23|Very Good|    H|    VS1| 59.4| 61.0|  338| 4.0|4.05|2.39|
    |  0.3|     Good|    J|    SI1| 64.0| 55.0|  339|4.25|4.28|2.73|
    | 0.23|    Ideal|    J|    VS1| 62.8| 56.0|  340|3.93| 3.9|2.46|
    | 0.22|  Premium|    F|    SI1| 60.4| 61.0|  342|3.88|3.84|2.33|
    | 0.31|    Ideal|    J|    SI2| 62.2| 54.0|  344|4.35|4.37|2.71|
    |  0.2|  Premium|    E|    SI2| 60.2| 62.0|  345|3.79|3.75|2.27|
    | 0.32|  Premium|    E|     I1| 60.9| 58.0|  345|4.38|4.42|2.68|
    |  0.3|    Ideal|    I|    SI2| 62.0| 54.0|  348|4.31|4.34|2.68|
    |  0.3|     Good|    J|    SI1| 63.4| 54.0|  351|4.23|4.29| 2.7|
    |  0.3|     Good|    J|    SI1| 63.8| 56.0|  351|4.23|4.26|2.71|
    |  0.3|Very Good|    J|    SI1| 62.7| 59.0|  351|4.21|4.27|2.66|
    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    only showing top 20 rows
    



```python
df_diamonds.count()==df_diamonds.where(df_diamonds['price']<400).count()
```




    False




```python
df_diamonds.where(df_diamonds['price']<400).show(50)
```

    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    |carat|      cut|color|clarity|depth|table|price|   x|   y|   z|
    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    | 0.23|    Ideal|    E|    SI2| 61.5| 55.0|  326|3.95|3.98|2.43|
    | 0.21|  Premium|    E|    SI1| 59.8| 61.0|  326|3.89|3.84|2.31|
    | 0.23|     Good|    E|    VS1| 56.9| 65.0|  327|4.05|4.07|2.31|
    | 0.29|  Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|
    | 0.31|     Good|    J|    SI2| 63.3| 58.0|  335|4.34|4.35|2.75|
    | 0.24|Very Good|    J|   VVS2| 62.8| 57.0|  336|3.94|3.96|2.48|
    | 0.24|Very Good|    I|   VVS1| 62.3| 57.0|  336|3.95|3.98|2.47|
    | 0.26|Very Good|    H|    SI1| 61.9| 55.0|  337|4.07|4.11|2.53|
    | 0.22|     Fair|    E|    VS2| 65.1| 61.0|  337|3.87|3.78|2.49|
    | 0.23|Very Good|    H|    VS1| 59.4| 61.0|  338| 4.0|4.05|2.39|
    |  0.3|     Good|    J|    SI1| 64.0| 55.0|  339|4.25|4.28|2.73|
    | 0.23|    Ideal|    J|    VS1| 62.8| 56.0|  340|3.93| 3.9|2.46|
    | 0.22|  Premium|    F|    SI1| 60.4| 61.0|  342|3.88|3.84|2.33|
    | 0.31|    Ideal|    J|    SI2| 62.2| 54.0|  344|4.35|4.37|2.71|
    |  0.2|  Premium|    E|    SI2| 60.2| 62.0|  345|3.79|3.75|2.27|
    | 0.32|  Premium|    E|     I1| 60.9| 58.0|  345|4.38|4.42|2.68|
    |  0.3|    Ideal|    I|    SI2| 62.0| 54.0|  348|4.31|4.34|2.68|
    |  0.3|     Good|    J|    SI1| 63.4| 54.0|  351|4.23|4.29| 2.7|
    |  0.3|     Good|    J|    SI1| 63.8| 56.0|  351|4.23|4.26|2.71|
    |  0.3|Very Good|    J|    SI1| 62.7| 59.0|  351|4.21|4.27|2.66|
    |  0.3|     Good|    I|    SI2| 63.3| 56.0|  351|4.26| 4.3|2.71|
    | 0.23|Very Good|    E|    VS2| 63.8| 55.0|  352|3.85|3.92|2.48|
    | 0.23|Very Good|    H|    VS1| 61.0| 57.0|  353|3.94|3.96|2.41|
    | 0.31|Very Good|    J|    SI1| 59.4| 62.0|  353|4.39|4.43|2.62|
    | 0.31|Very Good|    J|    SI1| 58.1| 62.0|  353|4.44|4.47|2.59|
    | 0.23|Very Good|    G|   VVS2| 60.4| 58.0|  354|3.97|4.01|2.41|
    | 0.24|  Premium|    I|    VS1| 62.5| 57.0|  355|3.97|3.94|2.47|
    |  0.3|Very Good|    J|    VS2| 62.2| 57.0|  357|4.28| 4.3|2.67|
    | 0.23|Very Good|    D|    VS2| 60.5| 61.0|  357|3.96|3.97| 2.4|
    | 0.23|Very Good|    F|    VS1| 60.9| 57.0|  357|3.96|3.99|2.42|
    | 0.24|Very Good|    E|    VS1| 61.5| 57.0|  357|3.99|4.07|2.48|
    | 0.23|Very Good|    D|    VS1| 61.8| 57.0|  357| 3.9|3.93|2.42|
    | 0.25|    Ideal|    H|    SI1| 62.8| 54.0|  357|4.05|4.07|2.55|
    | 0.23|     Good|    E|    VS2| 61.8| 63.0|  357|3.88|3.89| 2.4|
    | 0.23|     Good|    F|    VS2| 63.8| 57.0|  357|3.93|3.84|2.48|
    | 0.26|    Ideal|    I|    VS1| 61.9| 56.0|  358|4.08|4.16|2.55|
    | 0.28|Very Good|    H|    SI1| 61.5| 56.0|  360|4.21|4.24| 2.6|
    |  0.3|    Ideal|    I|    SI2| 62.0| 55.0|  360|4.32|4.33|2.68|
    | 0.27|    Ideal|    G|    SI2| 62.3| 55.0|  361|4.14|4.18|2.59|
    | 0.25|     Good|    E|    VS1| 63.3| 60.0|  361|3.99|4.04|2.54|
    | 0.25|     Fair|    E|    VS1| 55.2| 64.0|  361|4.21|4.23|2.33|
    | 0.32|     Good|    D|     I1| 64.0| 54.0|  361|4.33|4.36|2.78|
    | 0.23|Very Good|    D|    VS2| 62.7| 58.0|  362|3.86|3.89|2.43|
    | 0.24|Very Good|    E|    VS2| 64.1| 59.0|  362|3.88|3.92| 2.5|
    | 0.24|Very Good|    E|    VS2| 60.8| 56.0|  362|4.02|4.04|2.45|
    | 0.26|    Ideal|    H|    SI2| 62.5| 53.0|  362|4.09|4.13|2.57|
    | 0.25|    Ideal|    G|    SI1| 62.3| 53.0|  363|4.06|4.09|2.54|
    | 0.31|Very Good|    J|    SI1| 61.9| 59.0|  363|4.28|4.32|2.66|
    | 0.31|Very Good|    J|    SI1| 62.7| 59.0|  363|4.29|4.32| 2.7|
    | 0.31|  Premium|    J|    SI1| 60.9| 60.0|  363|4.36|4.38|2.66|
    +-----+---------+-----+-------+-----+-----+-----+----+----+----+
    only showing top 50 rows
    



```python
from pyspark.sql import Window
window = Window.partitionBy('cut', 'clarity').orderBy('price').rowsBetween(-3, 3)
window
```




    <pyspark.sql.window.WindowSpec object at 0x7fb326228320>




```python
from pyspark.sql.functions import mean
moving_avg = mean(df_diamonds['price']).over(window)
moving_avg
```




    Column<b'avg(price) OVER (PARTITION BY cut, clarity ORDER BY price ASC NULLS FIRST ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)'>




```python
df_diamonds = df_diamonds.withColumn('moving_avg', moving_avg)
df_diamonds.show()
```

    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+
    |carat|    cut|color|clarity|depth|table|price|   x|   y|   z|        moving_avg|
    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+
    | 0.29|Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|            358.75|
    |  0.2|Premium|    E|    VS2| 59.8| 62.0|  367|3.79|3.77|2.26|             360.4|
    |  0.2|Premium|    E|    VS2| 59.0| 60.0|  367|3.81|3.78|2.24|             361.5|
    |  0.2|Premium|    E|    VS2| 61.1| 59.0|  367|3.81|3.78|2.32| 362.2857142857143|
    |  0.2|Premium|    E|    VS2| 59.7| 62.0|  367|3.84| 3.8|2.28|             367.0|
    |  0.2|Premium|    F|    VS2| 62.6| 59.0|  367|3.73|3.71|2.33|367.14285714285717|
    |  0.2|Premium|    D|    VS2| 62.3| 60.0|  367|3.73|3.68|2.31| 367.2857142857143|
    |  0.2|Premium|    D|    VS2| 61.7| 60.0|  367|3.77|3.72|2.31|369.14285714285717|
    |  0.3|Premium|    J|    VS2| 62.2| 58.0|  368|4.28| 4.3|2.67|             371.0|
    |  0.3|Premium|    J|    VS2| 60.6| 59.0|  368|4.34|4.38|2.64| 373.7142857142857|
    | 0.31|Premium|    J|    VS2| 62.5| 60.0|  380|4.31|4.36|2.71|376.42857142857144|
    | 0.31|Premium|    J|    VS2| 62.4| 60.0|  380|4.29|4.33|2.69|379.14285714285717|
    | 0.21|Premium|    E|    VS2| 60.5| 59.0|  386|3.87|3.83|2.33| 381.7142857142857|
    | 0.21|Premium|    E|    VS2| 59.6| 56.0|  386|3.93|3.89|2.33| 384.2857142857143|
    | 0.21|Premium|    D|    VS2| 61.6| 59.0|  386|3.82|3.78|2.34|385.14285714285717|
    | 0.21|Premium|    D|    VS2| 60.6| 60.0|  386|3.85|3.81|2.32|             387.0|
    | 0.21|Premium|    D|    VS2| 59.1| 62.0|  386|3.89|3.86|2.29|             388.0|
    | 0.21|Premium|    D|    VS2| 58.3| 59.0|  386|3.96|3.93| 2.3|389.57142857142856|
    | 0.32|Premium|    J|    VS2| 61.9| 58.0|  393|4.35|4.38| 2.7|392.14285714285717|
    | 0.32|Premium|    J|    VS2| 61.9| 59.0|  393|4.35|4.38| 2.7| 394.7142857142857|
    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+
    only showing top 20 rows
    



```python
from pyspark.sql.functions import when, col

def replace(query, ma):
    return when(query<350, ma).otherwise(query)
```


```python
df_new = df_diamonds.withColumn('imputed', 
                       replace(col('price'), col('moving_avg')))
df_new.show()
```

    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+-------+
    |carat|    cut|color|clarity|depth|table|price|   x|   y|   z|        moving_avg|imputed|
    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+-------+
    | 0.29|Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|            358.75| 358.75|
    |  0.2|Premium|    E|    VS2| 59.8| 62.0|  367|3.79|3.77|2.26|             360.4|  367.0|
    |  0.2|Premium|    E|    VS2| 59.0| 60.0|  367|3.81|3.78|2.24|             361.5|  367.0|
    |  0.2|Premium|    E|    VS2| 61.1| 59.0|  367|3.81|3.78|2.32| 362.2857142857143|  367.0|
    |  0.2|Premium|    E|    VS2| 59.7| 62.0|  367|3.84| 3.8|2.28|             367.0|  367.0|
    |  0.2|Premium|    F|    VS2| 62.6| 59.0|  367|3.73|3.71|2.33|367.14285714285717|  367.0|
    |  0.2|Premium|    D|    VS2| 62.3| 60.0|  367|3.73|3.68|2.31| 367.2857142857143|  367.0|
    |  0.2|Premium|    D|    VS2| 61.7| 60.0|  367|3.77|3.72|2.31|369.14285714285717|  367.0|
    |  0.3|Premium|    J|    VS2| 62.2| 58.0|  368|4.28| 4.3|2.67|             371.0|  368.0|
    |  0.3|Premium|    J|    VS2| 60.6| 59.0|  368|4.34|4.38|2.64| 373.7142857142857|  368.0|
    | 0.31|Premium|    J|    VS2| 62.5| 60.0|  380|4.31|4.36|2.71|376.42857142857144|  380.0|
    | 0.31|Premium|    J|    VS2| 62.4| 60.0|  380|4.29|4.33|2.69|379.14285714285717|  380.0|
    | 0.21|Premium|    E|    VS2| 60.5| 59.0|  386|3.87|3.83|2.33| 381.7142857142857|  386.0|
    | 0.21|Premium|    E|    VS2| 59.6| 56.0|  386|3.93|3.89|2.33| 384.2857142857143|  386.0|
    | 0.21|Premium|    D|    VS2| 61.6| 59.0|  386|3.82|3.78|2.34|385.14285714285717|  386.0|
    | 0.21|Premium|    D|    VS2| 60.6| 60.0|  386|3.85|3.81|2.32|             387.0|  386.0|
    | 0.21|Premium|    D|    VS2| 59.1| 62.0|  386|3.89|3.86|2.29|             388.0|  386.0|
    | 0.21|Premium|    D|    VS2| 58.3| 59.0|  386|3.96|3.93| 2.3|389.57142857142856|  386.0|
    | 0.32|Premium|    J|    VS2| 61.9| 58.0|  393|4.35|4.38| 2.7|392.14285714285717|  393.0|
    | 0.32|Premium|    J|    VS2| 61.9| 59.0|  393|4.35|4.38| 2.7| 394.7142857142857|  393.0|
    +-----+-------+-----+-------+-----+-----+-----+----+----+----+------------------+-------+
    only showing top 20 rows
    


### Graphing


```python
df_diamonds.describe(['carat', 'depth', 'table', price]).show()
```

    +-------+------------------+------------------+------------------+-----------------+
    |summary|             carat|             depth|             table|            price|
    +-------+------------------+------------------+------------------+-----------------+
    |  count|             53940|             53940|             53940|            53940|
    |   mean|0.7979397478679852| 61.74940489432624| 57.45718390804603|3932.799721913237|
    | stddev|0.4740112444054196|1.4326213188336525|2.2344905628213247|3989.439738146397|
    |    min|               0.2|              43.0|              43.0|              326|
    |    max|              5.01|              79.0|              95.0|            18823|
    +-------+------------------+------------------+------------------+-----------------+
    



```python
import matplotlib
import matplotlib.pyplot as plt

carat = df_diamonds[['carat']].collect()
price = df_diamonds[['price']].collect()

plt.plot(carat, price, 'go', alpha=0.1)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Diamonds')
plt.show()
```

## To Pandas


```python
pandas_df = df_diamonds.toPandas()
pandas_df.describe()
```




                  carat         depth         table         price             x  \
    count  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000   
    mean       0.797940     61.749405     57.457184   3932.799722      5.731157   
    std        0.474011      1.432621      2.234491   3989.439738      1.121761   
    min        0.200000     43.000000     43.000000    326.000000      0.000000   
    25%        0.400000     61.000000     56.000000    950.000000      4.710000   
    50%        0.700000     61.800000     57.000000   2401.000000      5.700000   
    75%        1.040000     62.500000     59.000000   5324.250000      6.540000   
    max        5.010000     79.000000     95.000000  18823.000000     10.740000   
    
                      y             z    moving_avg  
    count  53940.000000  53940.000000  53940.000000  
    mean       5.734526      3.538734   3932.462711  
    std        1.142135      0.705699   3987.139854  
    min        0.000000      0.000000    344.500000  
    25%        4.720000      2.910000    949.714286  
    50%        5.710000      3.530000   2404.571429  
    75%        6.540000      4.040000   5328.285714  
    max       58.900000     31.800000  18790.250000  




```python
grouped = pandas_df.groupby('clarity').agg({'price':'mean', 'table':'min', 'depth':'max'}).reset_index()
grouped
```




      clarity  depth  table        price
    0      I1   78.2   52.0  3924.168691
    1      IF   65.6   52.0  2864.839106
    2     SI1   72.9   49.0  3996.001148
    3     SI2   72.2   50.1  5063.028606
    4     VS1   71.8   43.0  3839.455391
    5     VS2   79.0   51.0  3924.989395
    6    VVS1   67.6   52.0  2523.114637
    7    VVS2   67.6   51.0  3283.737071




```python
#in order from greatest clarity to least:
clarity_order = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'I2', 'I3']
mapping = {day: i for i, day in enumerate(clarity_order)}
key = grouped['clarity'].map(mapping)
grouped = grouped.iloc[key.argsort()]
grouped
```




      clarity  depth  table        price
    1      IF   65.6   52.0  2864.839106
    6    VVS1   67.6   52.0  2523.114637
    7    VVS2   67.6   51.0  3283.737071
    4     VS1   71.8   43.0  3839.455391
    5     VS2   79.0   51.0  3924.989395
    2     SI1   72.9   49.0  3996.001148
    3     SI2   72.2   50.1  5063.028606
    0      I1   78.2   52.0  3924.168691




```python
grouped.plot(kind='bar', x='clarity', legend=False)
```




    AxesSubplot(0.125,0.11;0.775x0.77)



### try glm


```python
df = df_diamonds[['carat', 'clarity', 'price']]

from pyspark.sql.functions import log
df = df.withColumn('lprice', log('price'))
df.show()
```

    +-----+-------+-----+------------------+
    |carat|clarity|price|            lprice|
    +-----+-------+-----+------------------+
    | 0.23|    SI2|  326| 5.786897381366708|
    | 0.21|    SI1|  326| 5.786897381366708|
    | 0.23|    VS1|  327|5.7899601708972535|
    | 0.29|    VS2|  334| 5.811140992976701|
    | 0.31|    SI2|  335| 5.814130531825066|
    | 0.24|   VVS2|  336| 5.817111159963204|
    | 0.24|   VVS1|  336| 5.817111159963204|
    | 0.26|    SI1|  337| 5.820082930352362|
    | 0.22|    VS2|  337| 5.820082930352362|
    | 0.23|    VS1|  338| 5.823045895483019|
    |  0.3|    SI1|  339|  5.82600010738045|
    | 0.23|    VS1|  340|5.8289456176102075|
    | 0.22|    SI1|  342| 5.834810737062605|
    | 0.31|    SI2|  344| 5.840641657373398|
    |  0.2|    SI2|  345|  5.84354441703136|
    | 0.32|     I1|  345|  5.84354441703136|
    |  0.3|    SI2|  348|5.8522024797744745|
    |  0.3|    SI1|  351| 5.860786223465865|
    |  0.3|    SI1|  351| 5.860786223465865|
    |  0.3|    SI1|  351| 5.860786223465865|
    +-----+-------+-----+------------------+
    only showing top 20 rows
    


[code here](https://github.com/UrbanInstitute/pyspark-tutorials/blob/master/indep_vars/build_indep_vars.py)


```python
import pyspark
def build_indep_vars(df, independent_vars, categorical_vars=None, keep_intermediate=False, summarizer=True):

    """
    Data verification
    df               : DataFrame
    independent_vars : List of column names
    categorical_vars : None or list of column names, e.g. ['col1', 'col2']
    """
    assert(type(df) is pyspark.sql.dataframe.DataFrame), 'pypark_glm: A pySpark dataframe is required as the first argument.'
    assert(type(independent_vars) is list), 'pyspark_glm: List of independent variable column names must be the third argument.'
    for iv in independent_vars:
        assert(type(iv) is str), 'pyspark_glm: Independent variables must be column name strings.'
        assert(iv in df.columns), 'pyspark_glm: Independent variable name is not a dataframe column.'
    if categorical_vars:
        for cv in categorical_vars:
            assert(type(cv) is str), 'pyspark_glm: Categorical variables must be column name strings.'
            assert(cv in df.columns), 'pyspark_glm: Categorical variable name is not a dataframe column.'
            assert(cv in independent_vars), 'pyspark_glm: Categorical variables must be independent variables.'

    """
    Code
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.regression import GeneralizedLinearRegression

    if categorical_vars:
        string_indexer = [StringIndexer(inputCol=x, 
                                        outputCol='{}_index'.format(x))
                          for x in categorical_vars]

        encoder        = [OneHotEncoder(dropLast=True, 
                                        inputCol ='{}_index' .format(x), 
                                        outputCol='{}_vector'.format(x))
                          for x in categorical_vars]

        independent_vars = ['{}_vector'.format(x) if x in categorical_vars else x for x in independent_vars]
    else:
        string_indexer, encoder = [], []

    assembler = VectorAssembler(inputCols=independent_vars, 
                                outputCol='indep_vars')
    pipeline  = Pipeline(stages=string_indexer+encoder+[assembler])
    model = pipeline.fit(df)
    df = model.transform(df)

    #for building the crosswalk between indicies and column names
    if summarizer:
        param_crosswalk = {}

        i = 0
        for x in independent_vars:
            if '_vector' in x[-7:]:
                xrs = x.rstrip('_vector')
                dst = df[[xrs, '{}_index'.format(xrs)]].distinct().collect()

                for row in dst:
                    param_crosswalk[int(row['{}_index'.format(xrs)]+i)] = row[xrs]
                maxind = max(param_crosswalk.keys())
                del param_crosswalk[maxind] #for droplast
                i += len(dst)
            elif '_index' in x[:-6]:
                pass
            else:
                param_crosswalk[i] = x
                i += 1
        """
        {0: 'carat',
         1: u'SI1',
         2: u'VS2',
         3: u'SI2',
         4: u'VS1',
         5: u'VVS2',
         6: u'VVS1',
         7: u'IF'}
        """
        make_summary = Summarizer(param_crosswalk)


    if not keep_intermediate:
        fcols = [c for c in df.columns if '_index' not in c[-6:] and '_vector' not in c[-7:]]
        df = df[fcols]

    if summarizer:
        return df, make_summary
    else:
        return df

```


```python
class Summarizer(object):
    def __init__(self, param_crosswalk):
        self.param_crosswalk = param_crosswalk
        self.precision = 4
        self.screen_width = 57
        self.hsep = '-'
        self.vsep = '|'

    def summarize(self, model, show=True, return_str=False):
        coefs = list(model.coefficients)
        inter = model.intercept
        tstat = model.summary.tValues
        stder = model.summary.coefficientStandardErrors
        pvals = model.summary.pValues

        #if model includes an intercept:
        if len(coefs) == len(tstat)-1:
            coefs.insert(0, inter)
            x = {0:'intercept'}
            for k, v in self.param_crosswalk.items():
                x[k+1] = v
        else:
            x = self.param_crosswalk

        assert(len(coefs) == len(tstat) == len(stder) == len(pvals))

        p = self.precision
        h = self.hsep
        v = self.vsep
        w = self.screen_width

        coefs = [str(round(e, p)).center(10) for e in coefs]
        tstat = [str(round(e, p)).center(10) for e in tstat]
        stder = [str(round(e, p)).center(10) for e in stder]
        pvals = [str(round(e, p)).center(10) for e in pvals]

        lines = ''
        for i in range(len(coefs)):
            lines += str(x[i]).rjust(15) + v + coefs[i] + stder[i] + tstat[i] + pvals[i] + '\n'

        labels = ''.rjust(15) + v + 'Coef'.center(10) + 'Std Err'.center(10) + 'T Stat'.center(10) + 'P Val'.center(10)
        pad    = ''.rjust(15) + v

        output = """{hline}\n{labels}\n{hline}\n{lines}{hline}""".format(
                    hline=h*w, 
                    labels=labels,
                    lines=lines)
        if show:
            print(output)
        if return_str:
            return output
```


```python
df, summarizer = build_indep_vars(df, 
                                  ['carat', 'clarity'], 
                                  categorical_vars=['clarity'], 
                                  keep_intermediate=False, 
                                  summarizer=True)
```


```python
df.show(5)
```

    +-----+-------+-----+------------------+--------------------+
    |carat|clarity|price|            lprice|          indep_vars|
    +-----+-------+-----+------------------+--------------------+
    | 0.23|    SI2|  326| 5.786897381366708|(8,[0,3],[0.23,1.0])|
    | 0.21|    SI1|  326| 5.786897381366708|(8,[0,1],[0.21,1.0])|
    | 0.23|    VS1|  327|5.7899601708972535|(8,[0,4],[0.23,1.0])|
    | 0.29|    VS2|  334| 5.811140992976701|(8,[0,2],[0.29,1.0])|
    | 0.31|    SI2|  335| 5.814130531825066|(8,[0,3],[0.31,1.0])|
    +-----+-------+-----+------------------+--------------------+
    only showing top 5 rows
    



```python
from pyspark.ml.regression import GeneralizedLinearRegression

glm = GeneralizedLinearRegression(family='gaussian', 
                                  link='identity', 
                                  labelCol='lprice', 
                                  featuresCol='indep_vars', 
                                  fitIntercept=True)

model = glm.fit(df)
```


```python
model.coefficients
```




    [2.08084271022,0.722096073558,0.817236425461,0.568221989598,0.85550589515,0.934075323936,0.919472035668,0.997943848244]




```python
model.intercept
```




    5.356165273724909




```python
model.summary.coefficientStandardErrors
```




    [0.0036263332330212492, 0.014091572711412436, 0.014154515091987462, 0.014180739640881894, 0.014369486052126049, 0.01479661778476952, 0.015205659565030371, 0.01644307646451729, 0.014396269149497595]




```python
summarizer.summarize(model)
```

    ---------------------------------------------------------
                   |   Coef    Std Err    T Stat    P Val   
    ---------------------------------------------------------
          intercept|  5.3562    0.0036   573.8145    0.0    
              carat|  2.0808    0.0141   51.2431     0.0    
                SI1|  0.7221    0.0142   57.7368     0.0    
                VS2|  0.8172    0.0142    40.07      0.0    
                SI2|  0.5682    0.0144   59.5363     0.0    
                VS1|  0.8555    0.0148   63.1276     0.0    
               VVS2|  0.9341    0.0152   60.4691     0.0    
               VVS1|  0.9195    0.0164   60.6908     0.0    
                 IF|  0.9979    0.0144   372.0523    0.0    
    ---------------------------------------------------------


### Pivoting


```python
from pyspark.sql import Row

row = Row('state', 'industry', 'hq', 'jobs')

df = sc.parallelize([
    row('MI', 'auto', 'domestic', 716),
    row('MI', 'auto', 'foreign', 123),
    row('MI', 'auto', 'domestic', 1340),
    row('MI', 'retail', 'foreign', 12),
    row('MI', 'retail', 'foreign', 33),
    row('OH', 'auto', 'domestic', 349),
    row('OH', 'auto', 'foreign', 101),
    row('OH', 'auto', 'foreign', 77),
    row('OH', 'retail', 'domestic', 45),
    row('OH', 'retail', 'foreign', 12)
    ]).toDF()
df.show()
```

    +-----+--------+--------+----+
    |state|industry|      hq|jobs|
    +-----+--------+--------+----+
    |   MI|    auto|domestic| 716|
    |   MI|    auto| foreign| 123|
    |   MI|    auto|domestic|1340|
    |   MI|  retail| foreign|  12|
    |   MI|  retail| foreign|  33|
    |   OH|    auto|domestic| 349|
    |   OH|    auto| foreign| 101|
    |   OH|    auto| foreign|  77|
    |   OH|  retail|domestic|  45|
    |   OH|  retail| foreign|  12|
    +-----+--------+--------+----+
    



```python
df_pivot1 = df.groupby('state').pivot('hq', values=['domestic', 'foreign']).sum('jobs')
df_pivot1.show()
```

    +-----+--------+-------+
    |state|domestic|foreign|
    +-----+--------+-------+
    |   MI|    2056|    168|
    |   OH|     394|    190|
    +-----+--------+-------+
    



```python
df_pivot2 = df.groupBy('state', 'industry').pivot('hq', values=['domestic', 'foreign']).sum('jobs')
df_pivot2.show()
```

    +-----+--------+--------+-------+
    |state|industry|domestic|foreign|
    +-----+--------+--------+-------+
    |   OH|  retail|      45|     12|
    |   MI|    auto|    2056|    123|
    |   OH|    auto|     349|    178|
    |   MI|  retail|    null|     45|
    +-----+--------+--------+-------+
    



```python
row = Row('state', 'industry', 'hq', 'jobs', 'firm')

df = sc.parallelize([
    row('MI', 'auto', 'domestic', 716, 'A'),
    row('MI', 'auto', 'foreign', 123, 'B'),
    row('MI', 'auto', 'domestic', 1340, 'C'),
    row('MI', 'retail', 'foreign', 12, 'D'),
    row('MI', 'retail', 'foreign', 33, 'E'),
    row('OH', 'retail', 'mixed', 978, 'F'),
    row('OH', 'auto', 'domestic', 349, 'G'),
    row('OH', 'auto', 'foreign', 101, 'H'),
    row('OH', 'auto', 'foreign', 77, 'I'),
    row('OH', 'retail', 'domestic', 45, 'J'),
    row('OH', 'retail', 'foreign', 12, 'K'),
    row('OH', 'retail', 'mixed', 1, 'L'),
    row('OH', 'auto', 'other', 120, 'M'),
    row('OH', 'auto', 'domestic', 96, 'A'),
    row('MI', 'auto', 'foreign', 1117, 'A'),
    row('MI', 'retail', 'mixed', 9, 'F'),
    row('MI', 'auto', 'foreign', 11, 'B')
    ]).toDF()
df.show()
```

    +-----+--------+--------+----+----+
    |state|industry|      hq|jobs|firm|
    +-----+--------+--------+----+----+
    |   MI|    auto|domestic| 716|   A|
    |   MI|    auto| foreign| 123|   B|
    |   MI|    auto|domestic|1340|   C|
    |   MI|  retail| foreign|  12|   D|
    |   MI|  retail| foreign|  33|   E|
    |   OH|  retail|   mixed| 978|   F|
    |   OH|    auto|domestic| 349|   G|
    |   OH|    auto| foreign| 101|   H|
    |   OH|    auto| foreign|  77|   I|
    |   OH|  retail|domestic|  45|   J|
    |   OH|  retail| foreign|  12|   K|
    |   OH|  retail|   mixed|   1|   L|
    |   OH|    auto|   other| 120|   M|
    |   OH|    auto|domestic|  96|   A|
    |   MI|    auto| foreign|1117|   A|
    |   MI|  retail|   mixed|   9|   F|
    |   MI|    auto| foreign|  11|   B|
    +-----+--------+--------+----+----+
    



```python
df_pivot3 = df.groupBy('firm', 'state', 'industry').pivot('hq', values=['domestic', 'foreign', 'mixed', 'other']).sum('jobs')
df_pivot3.show()
```

    +----+-----+--------+--------+-------+-----+-----+
    |firm|state|industry|domestic|foreign|mixed|other|
    +----+-----+--------+--------+-------+-----+-----+
    |   D|   MI|  retail|    null|     12| null| null|
    |   I|   OH|    auto|    null|     77| null| null|
    |   G|   OH|    auto|     349|   null| null| null|
    |   J|   OH|  retail|      45|   null| null| null|
    |   C|   MI|    auto|    1340|   null| null| null|
    |   A|   MI|    auto|     716|   1117| null| null|
    |   K|   OH|  retail|    null|     12| null| null|
    |   B|   MI|    auto|    null|    134| null| null|
    |   F|   MI|  retail|    null|   null|    9| null|
    |   E|   MI|  retail|    null|     33| null| null|
    |   M|   OH|    auto|    null|   null| null|  120|
    |   H|   OH|    auto|    null|    101| null| null|
    |   F|   OH|  retail|    null|   null|  978| null|
    |   L|   OH|  retail|    null|   null|    1| null|
    |   A|   OH|    auto|      96|   null| null| null|
    +----+-----+--------+--------+-------+-----+-----+
    


### Merging


```python
from pyspark.sql import Row

row = Row("name", "pet", "count")

df1 = sc.parallelize([
    row("Sue", "cat", 16),
    row("Kim", "dog", 1),    
    row("Bob", "fish", 5)
    ]).toDF()

df2 = sc.parallelize([
    row("Fred", "cat", 2),
    row("Kate", "ant", 179),    
    row("Marc", "lizard", 5)
    ]).toDF()

df3 = sc.parallelize([
    row("Sarah", "shark", 3),
    row("Jason", "kids", 2),    
    row("Scott", "squirrel", 1)
    ]).toDF()
```


```python
df1.show()
df2.show()
df3.show()
```

    +----+----+-----+
    |name| pet|count|
    +----+----+-----+
    | Sue| cat|   16|
    | Kim| dog|    1|
    | Bob|fish|    5|
    +----+----+-----+
    
    +----+------+-----+
    |name|   pet|count|
    +----+------+-----+
    |Fred|   cat|    2|
    |Kate|   ant|  179|
    |Marc|lizard|    5|
    +----+------+-----+
    
    +-----+--------+-----+
    | name|     pet|count|
    +-----+--------+-----+
    |Sarah|   shark|    3|
    |Jason|    kids|    2|
    |Scott|squirrel|    1|
    +-----+--------+-----+
    



```python
from pyspark.sql import DataFrame
from functools import reduce

def union_many(*dfs):
    return reduce(DataFrame.unionAll, dfs)

df_union = union_many(df1, df2, df3)
df_union.show()
```

    +-----+--------+-----+
    | name|     pet|count|
    +-----+--------+-----+
    |  Sue|     cat|   16|
    |  Kim|     dog|    1|
    |  Bob|    fish|    5|
    | Fred|     cat|    2|
    | Kate|     ant|  179|
    | Marc|  lizard|    5|
    |Sarah|   shark|    3|
    |Jason|    kids|    2|
    |Scott|squirrel|    1|
    +-----+--------+-----+
    



```python
row1 = Row("name", "pet", "count")
row2 = Row("name", "pet2", "count2")

df1 = sc.parallelize([
    row1("Sue", "cat", 16),
    row1("Kim", "dog", 1),    
    row1("Bob", "fish", 5),
    row1("Libuse", "horse", 1)
    ]).toDF()

df2 = sc.parallelize([
    row2("Sue", "eagle", 2),
    row2("Kim", "ant", 179),    
    row2("Bob", "lizard", 5),
    row2("Ferdinand", "bees", 23)
    ]).toDF()
```


```python
df1.join(df2, 'name', how='inner').show()
df1.join(df2, 'name', how='outer').show()
df1.join(df2, 'name', how='left').show()
```

    +----+----+-----+------+------+
    |name| pet|count|  pet2|count2|
    +----+----+-----+------+------+
    | Sue| cat|   16| eagle|     2|
    | Bob|fish|    5|lizard|     5|
    | Kim| dog|    1|   ant|   179|
    +----+----+-----+------+------+
    
    +---------+-----+-----+------+------+
    |     name|  pet|count|  pet2|count2|
    +---------+-----+-----+------+------+
    |      Sue|  cat|   16| eagle|     2|
    |Ferdinand| null| null|  bees|    23|
    |      Bob| fish|    5|lizard|     5|
    |      Kim|  dog|    1|   ant|   179|
    |   Libuse|horse|    1|  null|  null|
    +---------+-----+-----+------+------+
    
    +------+-----+-----+------+------+
    |  name|  pet|count|  pet2|count2|
    +------+-----+-----+------+------+
    |   Sue|  cat|   16| eagle|     2|
    |   Bob| fish|    5|lizard|     5|
    |   Kim|  dog|    1|   ant|   179|
    |Libuse|horse|    1|  null|  null|
    +------+-----+-----+------+------+
    


### UDF: User Defined Function


```python
import datetime
from pyspark.sql import Row
from pyspark.sql.functions import col

row = Row("date", "name", "production")

df = sc.parallelize([
    row("08/01/2014", "Kim", 5),
    row("08/02/2014", "Kim", 14),
    row("08/01/2014", "Bob", 6),
    row("08/02/2014", "Bob", 3),
    row("08/01/2014", "Sue", 0),
    row("08/02/2014", "Sue", 22),
    row("08/01/2014", "Dan", 4),
    row("08/02/2014", "Dan", 4),
    row("08/01/2014", "Joe", 37),
    row("09/01/2014", "Kim", 6),
    row("09/02/2014", "Kim", 6),
    row("09/01/2014", "Bob", 4),
    row("09/02/2014", "Bob", 20),
    row("09/01/2014", "Sue", 11),
    row("09/02/2014", "Sue", 2),
    row("09/01/2014", "Dan", 1),
    row("09/02/2014", "Dan", 3),
    row("09/02/2014", "Joe", 29)
    ]).toDF()
df.show()
```

    +----------+----+----------+
    |      date|name|production|
    +----------+----+----------+
    |08/01/2014| Kim|         5|
    |08/02/2014| Kim|        14|
    |08/01/2014| Bob|         6|
    |08/02/2014| Bob|         3|
    |08/01/2014| Sue|         0|
    |08/02/2014| Sue|        22|
    |08/01/2014| Dan|         4|
    |08/02/2014| Dan|         4|
    |08/01/2014| Joe|        37|
    |09/01/2014| Kim|         6|
    |09/02/2014| Kim|         6|
    |09/01/2014| Bob|         4|
    |09/02/2014| Bob|        20|
    |09/01/2014| Sue|        11|
    |09/02/2014| Sue|         2|
    |09/01/2014| Dan|         1|
    |09/02/2014| Dan|         3|
    |09/02/2014| Joe|        29|
    +----------+----+----------+
    



```python
from pyspark.sql.functions import udf

def split_date(whole_date):
    try:
        mo, day, yr = whole_date.split('/')
    except ValueError:
        return 'error'
    return mo + '/' + yr

udf_split_date = udf(split_date)


df_new = df.withColumn('month_year', udf_split_date('date')).drop('date')
df_new.show()
```

    +----+----------+----------+
    |name|production|month_year|
    +----+----------+----------+
    | Kim|         5|   08/2014|
    | Kim|        14|   08/2014|
    | Bob|         6|   08/2014|
    | Bob|         3|   08/2014|
    | Sue|         0|   08/2014|
    | Sue|        22|   08/2014|
    | Dan|         4|   08/2014|
    | Dan|         4|   08/2014|
    | Joe|        37|   08/2014|
    | Kim|         6|   09/2014|
    | Kim|         6|   09/2014|
    | Bob|         4|   09/2014|
    | Bob|        20|   09/2014|
    | Sue|        11|   09/2014|
    | Sue|         2|   09/2014|
    | Dan|         1|   09/2014|
    | Dan|         3|   09/2014|
    | Joe|        29|   09/2014|
    +----+----------+----------+
    



```python
from pyspark.sql.functions import udf
from pyspark.sql.types import DateType
from datetime import datetime

dateFormat = udf(lambda x: datetime.strptime(x, '%M/%d/%Y'), DateType())
    
df_d = df.withColumn('old_date', col('date')).withColumn('new_date', dateFormat(col('date'))).drop('date')
df_d.show()
```

    +----+----------+----------+----------+
    |name|production|  old_date|  new_date|
    +----+----------+----------+----------+
    | Kim|         5|08/01/2014|2014-01-01|
    | Kim|        14|08/02/2014|2014-01-02|
    | Bob|         6|08/01/2014|2014-01-01|
    | Bob|         3|08/02/2014|2014-01-02|
    | Sue|         0|08/01/2014|2014-01-01|
    | Sue|        22|08/02/2014|2014-01-02|
    | Dan|         4|08/01/2014|2014-01-01|
    | Dan|         4|08/02/2014|2014-01-02|
    | Joe|        37|08/01/2014|2014-01-01|
    | Kim|         6|09/01/2014|2014-01-01|
    | Kim|         6|09/02/2014|2014-01-02|
    | Bob|         4|09/01/2014|2014-01-01|
    | Bob|        20|09/02/2014|2014-01-02|
    | Sue|        11|09/01/2014|2014-01-01|
    | Sue|         2|09/02/2014|2014-01-02|
    | Dan|         1|09/01/2014|2014-01-01|
    | Dan|         3|09/02/2014|2014-01-02|
    | Joe|        29|09/02/2014|2014-01-02|
    +----+----------+----------+----------+
    


[Scala和Python的Spark性能对比](https://codeday.me/bug/20170710/37415.html)
