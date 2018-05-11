

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
