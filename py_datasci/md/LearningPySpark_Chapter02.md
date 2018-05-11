
### Creating RDDs

There are two ways to create an RDD in PySpark. You can parallelize a list


```python
sc
```




    <SparkContext master=local[4] appName=Apache Toree>




```python
data = sc.parallelize(
    [('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12), 
     ('Amber', 9)])
```

or read from a repository (a file or a database)


```python
data_from_file = sc.textFile('/home/karen/Downloads/data/VS14MORT.txt.gz', 4)
```

http://tomdrabas.com/data/VS14MORT.txt.gz

#### Schema

RDDs are *schema-less* data structures.


```python
data_heterogenous = sc.parallelize([('Ferrari', 'fast'), {'Porsche': 100000}, ['Spain','visited', 4504]]).collect()
data_heterogenous
```




    [('Ferrari', 'fast'), {'Porsche': 100000}, ['Spain', 'visited', 4504]]




```python
data_heterogenous[1]['Porsche']
```




    100000



#### Reading from files


```python
data_from_file.take(1)
```




    ['                   1                                          2101  M1087 432311  4M4                2014U7CN                                    I64 238 070   24 0111I64                                                                                                                                                                           01 I64                                                                                                  01  11                                 100 601']



#### User defined functions


```python
def extractInformation(row):
    import re
    import numpy as np

    selected_indices = [
         2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,
         19,21,22,23,24,25,27,28,29,30,32,33,34,
         36,37,38,39,40,41,42,43,44,45,46,47,48,
         49,50,51,52,53,54,55,56,58,60,61,62,63,
         64,65,66,67,68,69,70,71,72,73,74,75,76,
         77,78,79,81,82,83,84,85,87,89
    ]

    '''
        Input record schema
        schema: n-m (o) -- xxx
            n - position from
            m - position to
            o - number of characters
            xxx - description
        1. 1-19 (19) -- reserved positions
        2. 20 (1) -- resident status
        3. 21-60 (40) -- reserved positions
        4. 61-62 (2) -- education code (1989 revision)
        5. 63 (1) -- education code (2003 revision)
        6. 64 (1) -- education reporting flag
        7. 65-66 (2) -- month of death
        8. 67-68 (2) -- reserved positions
        9. 69 (1) -- sex
        10. 70 (1) -- age: 1-years, 2-months, 4-days, 5-hours, 6-minutes, 9-not stated
        11. 71-73 (3) -- number of units (years, months etc)
        12. 74 (1) -- age substitution flag (if the age reported in positions 70-74 is calculated using dates of birth and death)
        13. 75-76 (2) -- age recoded into 52 categories
        14. 77-78 (2) -- age recoded into 27 categories
        15. 79-80 (2) -- age recoded into 12 categories
        16. 81-82 (2) -- infant age recoded into 22 categories
        17. 83 (1) -- place of death
        18. 84 (1) -- marital status
        19. 85 (1) -- day of the week of death
        20. 86-101 (16) -- reserved positions
        21. 102-105 (4) -- current year
        22. 106 (1) -- injury at work
        23. 107 (1) -- manner of death
        24. 108 (1) -- manner of disposition
        25. 109 (1) -- autopsy
        26. 110-143 (34) -- reserved positions
        27. 144 (1) -- activity code
        28. 145 (1) -- place of injury
        29. 146-149 (4) -- ICD code
        30. 150-152 (3) -- 358 cause recode
        31. 153 (1) -- reserved position
        32. 154-156 (3) -- 113 cause recode
        33. 157-159 (3) -- 130 infant cause recode
        34. 160-161 (2) -- 39 cause recode
        35. 162 (1) -- reserved position
        36. 163-164 (2) -- number of entity-axis conditions
        37-56. 165-304 (140) -- list of up to 20 conditions
        57. 305-340 (36) -- reserved positions
        58. 341-342 (2) -- number of record axis conditions
        59. 343 (1) -- reserved position
        60-79. 344-443 (100) -- record axis conditions
        80. 444 (1) -- reserve position
        81. 445-446 (2) -- race
        82. 447 (1) -- bridged race flag
        83. 448 (1) -- race imputation flag
        84. 449 (1) -- race recode (3 categories)
        85. 450 (1) -- race recode (5 categories)
        86. 461-483 (33) -- reserved positions
        87. 484-486 (3) -- Hispanic origin
        88. 487 (1) -- reserved
        89. 488 (1) -- Hispanic origin/race recode
     '''

    record_split = re\
        .compile(
            r'([\s]{19})([0-9]{1})([\s]{40})([0-9\s]{2})([0-9\s]{1})([0-9]{1})([0-9]{2})' + 
            r'([\s]{2})([FM]{1})([0-9]{1})([0-9]{3})([0-9\s]{1})([0-9]{2})([0-9]{2})' + 
            r'([0-9]{2})([0-9\s]{2})([0-9]{1})([SMWDU]{1})([0-9]{1})([\s]{16})([0-9]{4})' +
            r'([YNU]{1})([0-9\s]{1})([BCOU]{1})([YNU]{1})([\s]{34})([0-9\s]{1})([0-9\s]{1})' +
            r'([A-Z0-9\s]{4})([0-9]{3})([\s]{1})([0-9\s]{3})([0-9\s]{3})([0-9\s]{2})([\s]{1})' + 
            r'([0-9\s]{2})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' + 
            r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' + 
            r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' + 
            r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' + 
            r'([A-Z0-9\s]{7})([\s]{36})([A-Z0-9\s]{2})([\s]{1})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' + 
            r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' + 
            r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' + 
            r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' + 
            r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([\s]{1})([0-9\s]{2})([0-9\s]{1})' + 
            r'([0-9\s]{1})([0-9\s]{1})([0-9\s]{1})([\s]{33})([0-9\s]{3})([0-9\s]{1})([0-9\s]{1})')
    try:
        rs = np.array(record_split.split(row))[selected_indices]
    except:
        rs = np.array(['-99'] * len(selected_indices))
    return rs
```


```python
data_from_file_conv = data_from_file.map(extractInformation)
data_from_file_conv.map(lambda row: row).take(1)
```




    [array(['1', '  ', '2', '1', '01', 'M', '1', '087', ' ', '43', '23', '11',
           '  ', '4', 'M', '4', '2014', 'U', '7', 'C', 'N', ' ', ' ', 'I64 ',
           '238', '070', '   ', '24', '01', '11I64  ', '       ', '       ',
           '       ', '       ', '       ', '       ', '       ', '       ',
           '       ', '       ', '       ', '       ', '       ', '       ',
           '       ', '       ', '       ', '       ', '       ', '01',
           'I64  ', '     ', '     ', '     ', '     ', '     ', '     ',
           '     ', '     ', '     ', '     ', '     ', '     ', '     ',
           '     ', '     ', '     ', '     ', '     ', '     ', '01', ' ',
           ' ', '1', '1', '100', '6'],
          dtype='<U40')]



### Transformations

#### .map(...)


```python
data_2014 = data_from_file_conv.map(lambda row: int(row[16]))
data_2014.take(10)
```




    [2014, 2014, 2014, 2014, 2014, 2014, 2014, 2014, 2014, -99]




```python
data_2014_2 = data_from_file_conv.map(lambda row: (row[16], int(row[16])))
data_2014_2.take(10)
```




    [('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('2014', 2014),
     ('-99', -99)]



#### .filter(...)


```python
data_filtered = data_from_file_conv.filter(lambda row: row[5] == 'F' and row[21] == '0')
data_filtered.count()
```




    6



#### .flatMap(...)


```python
data_2014_flat = data_from_file_conv.flatMap(lambda row: (row[16], int(row[16]) + 1))
data_2014_flat.take(10)
```




    ['2014', 2015, '2014', 2015, '2014', 2015, '2014', 2015, '2014', 2015]



#### .distinct()


```python
distinct_gender = data_from_file_conv.map(lambda row: row[5]).distinct().collect()
distinct_gender
```




    ['-99', 'M', 'F']



#### .sample(...)


```python
fraction = 0.1
data_sample = data_from_file_conv.sample(False, fraction, 666)

data_sample.take(1)
```




    [array(['1', '  ', '5', '1', '01', 'F', '1', '082', ' ', '42', '22', '10',
            '  ', '4', 'W', '5', '2014', 'U', '7', 'C', 'N', ' ', ' ', 'I251',
            '215', '063', '   ', '21', '02', '11I350 ', '21I251 ', '       ',
            '       ', '       ', '       ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '02',
            'I251 ', 'I350 ', '     ', '     ', '     ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '28', ' ',
            ' ', '2', '4', '100', '8'], 
           dtype='<U40')]




```python
print('Original dataset: {0}, sample: {1}'.format(data_from_file_conv.count(), data_sample.count()))
```

    Original dataset: 2631171, sample: 263247


#### .leftOuterJoin(...)


```python
rdd1 = sc.parallelize([('a', 1), ('b', 4), ('c',10)])
rdd2 = sc.parallelize([('a', 4), ('a', 1), ('b', '6'), ('d', 15)])

rdd3 = rdd1.leftOuterJoin(rdd2)
rdd3.take(5)
```




    [('c', (10, None)), ('b', (4, '6')), ('a', (1, 4)), ('a', (1, 1))]




```python
rdd4 = rdd1.join(rdd2)
rdd4.collect()
```




    [('b', (4, '6')), ('a', (1, 4)), ('a', (1, 1))]



`.intersection(...)` that returns the records that are *equal* in both RDDs.


```python
rdd5 = rdd1.intersection(rdd2)
rdd5.collect()
```




    [('a', 1)]



#### .repartition(...)

Repartitioning the dataset changes the number of partitions the dataset is divided into.


```python
rdd1 = rdd1.repartition(4)

len(rdd1.glom().collect())
```




    4



### Actions

#### .take(...)


```python
data_first = data_from_file_conv.take(1)
data_first
```




    [array(['1', '  ', '2', '1', '01', 'M', '1', '087', ' ', '43', '23', '11',
            '  ', '4', 'M', '4', '2014', 'U', '7', 'C', 'N', ' ', ' ', 'I64 ',
            '238', '070', '   ', '24', '01', '11I64  ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '01',
            'I64  ', '     ', '     ', '     ', '     ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '01', ' ',
            ' ', '1', '1', '100', '6'], 
           dtype='<U40')]



If you want somewhat randomized records you can use `.takeSample(...)` instead.


```python
data_take_sampled = data_from_file_conv.takeSample(False, 1, 667)
data_take_sampled
```




    [array(['2', '17', ' ', '0', '08', 'M', '1', '069', ' ', '39', '19', '09',
            '  ', '1', 'M', '7', '2014', 'U', '7', 'U', 'N', ' ', ' ', 'I251',
            '215', '063', '   ', '21', '06', '11I500 ', '21I251 ', '61I499 ',
            '62I10  ', '63N189 ', '64K761 ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '       ',
            '       ', '       ', '       ', '       ', '       ', '05',
            'I251 ', 'I120 ', 'I499 ', 'I500 ', 'K761 ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '     ',
            '     ', '     ', '     ', '     ', '     ', '     ', '01', ' ',
            ' ', '1', '1', '100', '6'], 
           dtype='<U40')]



#### .reduce(...)


```python
rdd1.map(lambda row: row[1]).reduce(lambda x, y: x + y)
```




    15




```python
data_reduce = sc.parallelize([1, 2, .5, .1, 5, .2], 1)
```


```python
works = data_reduce.reduce(lambda x, y: x / y)
works
```




    10.0




```python
data_reduce = sc.parallelize([1, 2, .5, .1, 5, .2], 3)
data_reduce.reduce(lambda x, y: x / y)
```




    0.004



The `.reduceByKey(...)` method works in a similar way to the `.reduce(...)` method but performs a reduction on a key-by-key basis.


```python
data_key = sc.parallelize([('a', 4),('b', 3),('c', 2),('a', 8),('d', 2),('b', 1),('d', 3)],4)
data_key.reduceByKey(lambda x, y: x + y).collect()
```




    [('b', 4), ('c', 2), ('a', 12), ('d', 5)]



#### .count()


```python
data_reduce.count()
```




    6



If your dataset is in a form of a *key-value* you can use the `.countByKey()` method to get the counts of distinct keys.


```python
data_key.countByKey().items()
```




    dict_items([('a', 2), ('b', 2), ('d', 2), ('c', 1)])



#### .saveAsTextFile(...)


```python
data_key.saveAsTextFile('data_key.txt')
```


```python
def parseInput(row):
    import re
    
    pattern = re.compile(r'\(\'([a-z])\', ([0-9])\)')
    row_split = pattern.split(row)
    
    return (row_split[1], int(row_split[2]))
    
data_key_reread = sc \
    .textFile('data_key.txt') \
    .map(parseInput)
data_key_reread.collect()
```




    [('a', 4), ('b', 3), ('c', 2), ('a', 8), ('d', 2), ('b', 1), ('d', 3)]





```python
# Generate our own JSON data 
#   This way we don't have to access the file system yet.
stringJSONRDD = sc.parallelize((""" 
  { "id": "123",
    "name": "Katie",
    "age": 19,
    "eyeColor": "brown"
  }""",
   """{
    "id": "234",
    "name": "Michael",
    "age": 22,
    "eyeColor": "green"
  }""", 
  """{
    "id": "345",
    "name": "Simone",
    "age": 23,
    "eyeColor": "blue"
  }""")
)
```


```python
# Create DataFrame
swimmersJSON = spark.read.json(stringJSONRDD)
```


```python
# Create temporary table
swimmersJSON.createOrReplaceTempView("swimmersJSON")
```


```python
# DataFrame API
swimmersJSON.show()
```

    +---+--------+---+-------+
    |age|eyeColor| id|   name|
    +---+--------+---+-------+
    | 19|   brown|123|  Katie|
    | 22|   green|234|Michael|
    | 23|    blue|345| Simone|
    +---+--------+---+-------+
    



```python
# SQL Query
spark.sql("select * from swimmersJSON").collect()
```




    [Row(age=19, eyeColor='brown', id='123', name='Katie'), Row(age=22, eyeColor='green', id='234', name='Michael'), Row(age=23, eyeColor='blue', id='345', name='Simone')]




```python
# Print the schema
swimmersJSON.printSchema()
```

    root
     |-- age: long (nullable = true)
     |-- eyeColor: string (nullable = true)
     |-- id: string (nullable = true)
     |-- name: string (nullable = true)
    


#### Programmatically Specifying the Schema
In this case, let's specify the schema for a `CSV` text file.


```python
from pyspark.sql.types import *

stringCSVRDD = sc.parallelize([(123, 'Katie', 19, 'brown'), (234, 'Michael', 22, 'green'), (345, 'Simone', 23, 'blue')])

schemaString = "id name age eyeColor"
schema = StructType([
    StructField("id", LongType(), True),    
    StructField("name", StringType(), True),
    StructField("age", LongType(), True),
    StructField("eyeColor", StringType(), True)
])

# Apply the schema to the RDD and Create DataFrame
swimmers = spark.createDataFrame(stringCSVRDD, schema)

# Creates a temporary view using the DataFrame
swimmers.createOrReplaceTempView("swimmers")
```


```python
swimmers.printSchema()
```

    root
     |-- id: long (nullable = true)
     |-- name: string (nullable = true)
     |-- age: long (nullable = true)
     |-- eyeColor: string (nullable = true)
    


### Querying with SQL


```python
# Execute SQL Query and return the data
spark.sql("select * from swimmers").show()
```

    +---+-------+---+--------+
    | id|   name|age|eyeColor|
    +---+-------+---+--------+
    |123|  Katie| 19|   brown|
    |234|Michael| 22|   green|
    |345| Simone| 23|    blue|
    +---+-------+---+--------+
    



```python
# Get count of rows in SQL
spark.sql("select count(1) from swimmers").show()
```


```python
# Query id and age for swimmers with age = 22 via DataFrame API
swimmers.select("id", "age").filter("age = 22").show()
```


```python
# Query id and age for swimmers with age = 22 via DataFrame API in another way
swimmers.select(swimmers.id, swimmers.age).filter(swimmers.age == 22).show()
```


```python
# Query id and age for swimmers with age = 22 in SQL
spark.sql("select id, age from swimmers where age = 22").show()
```


```python
# Query name and eye color for swimmers with eye color starting with the letter 'b'
spark.sql("select name, eyeColor from swimmers where eyeColor like 'b%'").show()
```

### Querying with the DataFrame API
With DataFrames, you can start writing your queries using the DataFrame API


```python
# Show the values 
swimmers.show()
```


```python
# Using Databricks `display` command to view the data easier
display(swimmers)
```


```python
# Get count of rows
swimmers.count()
```


```python
# Get the id, age where age = 22
swimmers.select("id", "age").filter("age = 22").show()
```


```python
# Get the name, eyeColor where eyeColor like 'b%'
swimmers.select("name", "eyeColor").filter("eyeColor like 'b%'").show()
```

## On-Time Flight Performance
Query flight departure delays by State and City by joining the departure delay and join to the airport codes (to identify state and city).


```python
flightPerfFilePath = "flight-data/departuredelays.csv"
airportsFilePath = "flight-data/airport-codes-na.txt"

airports = spark.read.csv(airportsFilePath, header='true', inferSchema='true', sep='\t')
airports.createOrReplaceTempView("airports")

flightPerf = spark.read.csv(flightPerfFilePath, header='true')
flightPerf.createOrReplaceTempView("FlightPerformance")

flightPerf.cache()
airports.show()
flightPerf.show()
```

    +-----------+-----+-------+----+
    |       City|State|Country|IATA|
    +-----------+-----+-------+----+
    | Abbotsford|   BC| Canada| YXX|
    |   Aberdeen|   SD|    USA| ABR|
    |    Abilene|   TX|    USA| ABI|
    |      Akron|   OH|    USA| CAK|
    |    Alamosa|   CO|    USA| ALS|
    |     Albany|   GA|    USA| ABY|
    |     Albany|   NY|    USA| ALB|
    |Albuquerque|   NM|    USA| ABQ|
    | Alexandria|   LA|    USA| AEX|
    |  Allentown|   PA|    USA| ABE|
    |   Alliance|   NE|    USA| AIA|
    |     Alpena|   MI|    USA| APN|
    |    Altoona|   PA|    USA| AOO|
    |   Amarillo|   TX|    USA| AMA|
    |Anahim Lake|   BC| Canada| YAA|
    |  Anchorage|   AK|    USA| ANC|
    |   Appleton|   WI|    USA| ATW|
    |     Arviat|  NWT| Canada| YEK|
    |  Asheville|   NC|    USA| AVL|
    |      Aspen|   CO|    USA| ASE|
    +-----------+-----+-------+----+
    only showing top 20 rows
    
    +--------+-----+--------+------+-----------+
    |    date|delay|distance|origin|destination|
    +--------+-----+--------+------+-----------+
    |01011245|    6|     602|   ABE|        ATL|
    |01020600|   -8|     369|   ABE|        DTW|
    |01021245|   -2|     602|   ABE|        ATL|
    |01020605|   -4|     602|   ABE|        ATL|
    |01031245|   -4|     602|   ABE|        ATL|
    |01030605|    0|     602|   ABE|        ATL|
    |01041243|   10|     602|   ABE|        ATL|
    |01040605|   28|     602|   ABE|        ATL|
    |01051245|   88|     602|   ABE|        ATL|
    |01050605|    9|     602|   ABE|        ATL|
    |01061215|   -6|     602|   ABE|        ATL|
    |01061725|   69|     602|   ABE|        ATL|
    |01061230|    0|     369|   ABE|        DTW|
    |01060625|   -3|     602|   ABE|        ATL|
    |01070600|    0|     369|   ABE|        DTW|
    |01071725|    0|     602|   ABE|        ATL|
    |01071230|    0|     369|   ABE|        DTW|
    |01070625|    0|     602|   ABE|        ATL|
    |01071219|    0|     569|   ABE|        ORD|
    |01080600|    0|     369|   ABE|        DTW|
    +--------+-----+--------+------+-----------+
    only showing top 20 rows
    



```python
# Query Sum of Flight Delays by City and Origin Code (for Washington State)
spark.sql("select a.City, f.origin, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.State = 'WA' group by a.City, f.origin order by sum(f.delay) desc").show()
```

    +-------+------+--------+
    |   City|origin|  Delays|
    +-------+------+--------+
    |Seattle|   SEA|159086.0|
    |Spokane|   GEG| 12404.0|
    |  Pasco|   PSC|   949.0|
    +-------+------+--------+
    



```python
%sql
-- Query Sum of Flight Delays by City and Origin Code (for Washington State)
select a.City, f.origin, sum(f.delay) as Delays
  from FlightPerformance f
    join airports a
      on a.IATA = f.origin
 where a.State = 'WA'
 group by a.City, f.origin
 order by sum(f.delay) desc
 
```


```python
# Query Sum of Flight Delays by State (for the US)
spark.sql(
    "select a.State, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.Country = 'USA' group by a.State "
).show()
```


```python
%sql
-- Query Sum of Flight Delays by State (for the US)
select a.State, sum(f.delay) as Delays
  from FlightPerformance f
    join airports a
      on a.IATA = f.origin
 where a.Country = 'USA'
 group by a.State 
```


```python
%sql
-- Query Sum of Flight Delays by State (for the US)
select a.State, sum(f.delay) as Delays
  from FlightPerformance f
    join airports a
      on a.IATA = f.origin
 where a.Country = 'USA'
 group by a.State 
```

For more:
* [Spark SQL, DataFrames and Datasets Guide](http://spark.apache.org/docs/latest/sql-programming-guide.html#sql)
* [PySpark SQL Module: DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)
* [PySpark SQL Functions Module](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions)





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


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/pys/output_57_0.png)



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


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/pys/output_59_0.png)


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
    

