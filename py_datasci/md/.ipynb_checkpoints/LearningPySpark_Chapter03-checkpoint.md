
\begin{equation*}
x+y^{78}
\end{equation*}



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
%sql 
-- Query id and age for swimmers with age = 22
select id, age from swimmers where age = 22
```


```python
# Query name and eye color for swimmers with eye color starting with the letter 'b'
spark.sql("select name, eyeColor from swimmers where eyeColor like 'b%'").show()
```


```python
%sql 
-- Query name and eye color for swimmers with eye color starting with the letter 'b'
select name, eyeColor from swimmers where eyeColor like 'b%'
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

### DataFrame Queries
Let's run a flight performance using DataFrames; let's first build the DataFrames from the source datasets.


```python
# Set File Paths
flightPerfFilePath = "/databricks-datasets/flights/departuredelays.csv"
airportsFilePath = "/databricks-datasets/flights/airport-codes-na.txt"

# Obtain Airports dataset
airports = spark.read.csv(airportsFilePath, header='true', inferSchema='true', sep='\t')
airports.createOrReplaceTempView("airports")

# Obtain Departure Delays dataset
flightPerf = spark.read.csv(flightPerfFilePath, header='true')
flightPerf.createOrReplaceTempView("FlightPerformance")

# Cache the Departure Delays dataset 
flightPerf.cache()
```


```python
# Query Sum of Flight Delays by City and Origin Code (for Washington State)
spark.sql("select a.City, f.origin, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.State = 'WA' group by a.City, f.origin order by sum(f.delay) desc").show()
```


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
spark.sql("select a.State, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.Country = 'USA' group by a.State ").show()
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

For more information, please refer to:
* [Spark SQL, DataFrames and Datasets Guide](http://spark.apache.org/docs/latest/sql-programming-guide.html#sql)
* [PySpark SQL Module: DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)
* [PySpark SQL Functions Module](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions)
