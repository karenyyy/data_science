
# GraphFrames 


### Preparation
Extract the Airports and Departure Delays information from S3 / DBFS


```python
spark
```




    <pyspark.sql.session.SparkSession at 0x7f1ceff50c88>




```python
tripdelaysFilePath = "data/departuredelays.csv"
airportsnaFilePath = "data/airport-codes-na.txt"

airportsna = spark.read.csv(airportsnaFilePath, header='true', inferSchema='true', sep='\t')
airportsna.createOrReplaceTempView("airports_na")
airportsna.show()

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
    



```python
departureDelays = spark.read.csv(tripdelaysFilePath, header='true')
departureDelays.createOrReplaceTempView("departureDelays")
departureDelays.cache()
departureDelays.show(10)
```

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
    +--------+-----+--------+------+-----------+
    only showing top 10 rows
    



```python
tripIATA = spark.sql("select distinct airport from \
                        (select distinct origin as airport from departureDelays \
                            union all \
                                select distinct destination as airport from departureDelays) \
                      as airportsna")
tripIATA.createOrReplaceTempView("tripIATA")
tripIATA.show()
```

    +-------+
    |airport|
    +-------+
    |    PSE|
    |    INL|
    |    MSY|
    |    PPG|
    |    GEG|
    |    BUR|
    |    SNA|
    |    GRB|
    |    GTF|
    |    IDA|
    |    GRR|
    |    JLN|
    |    EUG|
    |    PSG|
    |    GSO|
    |    MYR|
    |    PVD|
    |    OAK|
    |    BTM|
    |    COD|
    +-------+
    only showing top 20 rows
    



```python
departureDelays.count()
```




    1391578




```python
# Build `departureDelays_geo` DataFrame
#  Obtain key attributes such as Date of flight, delays, distance, and airport information (Origin, Destination)
departureDelays_geo = spark.sql("select cast(f.date as int) as tripid,\
                                cast( concat(\
                                        concat(\
                                            concat(\
                                                concat(\
                                                    concat(\
                                                        concat('2014-',\
                                                            concat(\
                                                                concat(\
                                                                    substr(cast(f.date as string), 1, 2), \
                                                                '-')\
                                                            ),\
                                                            substr(cast(f.date as string), 3, 2)\
                                                        ),\
                                                    ' '), \
                                                substr(cast(f.date as string), 5, 2)), \
                                            ':'), \
                                        substr(cast(f.date as string), 7, 2)), \
                                    ':00') \
                                    as timestamp)\
                                as `localdate`, \
                                cast(f.delay as int), \
                                cast(f.distance as int),\
                                f.origin as src, \
                                f.destination as dst,\
                                o.city as city_src, \
                                o.state as state_src, \
                                d.city as city_dst, \
                                d.state as state_dst \
                                from departuredelays as f \
                                join airports_na as o on o.IATA = f.origin \
                                join airports_na as d on d.IATA = f.destination"
                                )

# Create Temporary View and cache
departureDelays_geo.createOrReplaceTempView("departureDelays_geo")
departureDelays_geo.show()
departureDelays_geo.cache()

# Count
departureDelays_geo.count()
```

    +-------+--------------------+-----+--------+---+---+---------+---------+--------+---------+
    | tripid|           localdate|delay|distance|src|dst| city_src|state_src|city_dst|state_dst|
    +-------+--------------------+-----+--------+---+---+---------+---------+--------+---------+
    |1011245|2014-01-01 12:45:...|    6|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1020600|2014-01-02 06:00:...|   -8|     369|ABE|DTW|Allentown|       PA| Detroit|       MI|
    |1021245|2014-01-02 12:45:...|   -2|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1020605|2014-01-02 06:05:...|   -4|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1031245|2014-01-03 12:45:...|   -4|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1030605|2014-01-03 06:05:...|    0|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1041243|2014-01-04 12:43:...|   10|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1040605|2014-01-04 06:05:...|   28|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1051245|2014-01-05 12:45:...|   88|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1050605|2014-01-05 06:05:...|    9|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1061215|2014-01-06 12:15:...|   -6|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1061725|2014-01-06 17:25:...|   69|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1061230|2014-01-06 12:30:...|    0|     369|ABE|DTW|Allentown|       PA| Detroit|       MI|
    |1060625|2014-01-06 06:25:...|   -3|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1070600|2014-01-07 06:00:...|    0|     369|ABE|DTW|Allentown|       PA| Detroit|       MI|
    |1071725|2014-01-07 17:25:...|    0|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1071230|2014-01-07 12:30:...|    0|     369|ABE|DTW|Allentown|       PA| Detroit|       MI|
    |1070625|2014-01-07 06:25:...|    0|     602|ABE|ATL|Allentown|       PA| Atlanta|       GA|
    |1071219|2014-01-07 12:19:...|    0|     569|ABE|ORD|Allentown|       PA| Chicago|       IL|
    |1080600|2014-01-08 06:00:...|    0|     369|ABE|DTW|Allentown|       PA| Detroit|       MI|
    +-------+--------------------+-----+--------+---+---+---------+---------+--------+---------+
    only showing top 20 rows
    





    1361141



## Building the Graph
- build the structure of the vertices 
- build the structure of the edges

* Rename IATA airport code to **id** in the Vertices Table
* Start and End airports to **src** and **dst** for the Edges Table (flights)



```python
from pyspark.sql.functions import *
from graphframes import *

# Create Vertices (airports) and Edges (flights)
tripVertices = airportsna.withColumnRenamed("IATA", "id").distinct()
tripEdges = departureDelays_geo.select("tripid", "delay", "src", "dst", "city_dst", "state_dst")

# Cache Vertices and Edges
tripEdges.cache()
tripVertices.cache()
```




    DataFrame[City: string, State: string, Country: string, id: string]




```python
# Vertices
#   The vertices of our graph are the airports
tripVertices.show()
```

    +--------------+-------+-------+---+
    |          City|  State|Country| id|
    +--------------+-------+-------+---+
    |    Clarksburg|     WV|    USA|CKB|
    |    Fort Dodge|     IA|    USA|FOD|
    |       Redmond|     OR|    USA|RDM|
    |        Valdez|     AK|    USA|VDZ|
    |       Lebanon|     NH|    USA|LEB|
    |         Aspen|     CO|    USA|ASE|
    | Rouyn-Noranda|     PQ| Canada|YUY|
    |       Iqaluit|Nunavut| Canada|YFB|
    |       Toronto|     ON| Canada|YYZ|
    |    Dillingham|     AK|    USA|DLG|
    |   Fort Nelson|     BC| Canada|YYE|
    |      Pellston|     MI|    USA|PLN|
    |     St. Cloud|     MN|    USA|STC|
    |St. Petersburg|     FL|    USA|PIE|
    |    Great Bend|     KS|    USA|GBD|
    |          Hilo|     HI|    USA|Big|
    |       Kelowna|     BC| Canada|YLW|
    |       Bemidji|     MN|    USA|BJI|
    |        London|     ON| Canada|YXU|
    |      Longview|     TX|    USA|GGG|
    +--------------+-------+-------+---+
    only showing top 20 rows
    



```python
# Edges
#  The edges of our graph are the flights between airports
tripEdges.show()
```

    +-------+-----+---+---+--------+---------+
    | tripid|delay|src|dst|city_dst|state_dst|
    +-------+-----+---+---+--------+---------+
    |1011245|    6|ABE|ATL| Atlanta|       GA|
    |1020600|   -8|ABE|DTW| Detroit|       MI|
    |1021245|   -2|ABE|ATL| Atlanta|       GA|
    |1020605|   -4|ABE|ATL| Atlanta|       GA|
    |1031245|   -4|ABE|ATL| Atlanta|       GA|
    |1030605|    0|ABE|ATL| Atlanta|       GA|
    |1041243|   10|ABE|ATL| Atlanta|       GA|
    |1040605|   28|ABE|ATL| Atlanta|       GA|
    |1051245|   88|ABE|ATL| Atlanta|       GA|
    |1050605|    9|ABE|ATL| Atlanta|       GA|
    |1061215|   -6|ABE|ATL| Atlanta|       GA|
    |1061725|   69|ABE|ATL| Atlanta|       GA|
    |1061230|    0|ABE|DTW| Detroit|       MI|
    |1060625|   -3|ABE|ATL| Atlanta|       GA|
    |1070600|    0|ABE|DTW| Detroit|       MI|
    |1071725|    0|ABE|ATL| Atlanta|       GA|
    |1071230|    0|ABE|DTW| Detroit|       MI|
    |1070625|    0|ABE|ATL| Atlanta|       GA|
    |1071219|    0|ABE|ORD| Chicago|       IL|
    |1080600|    0|ABE|DTW| Detroit|       MI|
    +-------+-----+---+---+--------+---------+
    only showing top 20 rows
    



```python
# Build `tripGraph` GraphFrame
#  This GraphFrame builds up on the vertices and edges based on our trips (flights)
tripGraph = GraphFrame(tripVertices, tripEdges)

# Build `tripGraphPrime` GraphFrame
#   This graphframe contains a smaller subset of data to make it easier to display motifs and subgraphs (below)
tripEdgesPrime = departureDelays_geo.select("tripid", "delay", "src", "dst")
tripGraphPrime = GraphFrame(tripVertices, tripEdgesPrime)
```

## Simple Queries
Let's start with a set of simple graph queries to understand flight performance and departure delays

#### Determine the number of airports and trips


```python
tripGraph.vertices.show()
tripGraph.edges.show()
print("Airports: %d" % tripGraph.vertices.count())
print("Trip routes: %d" % tripGraph.edges.count())
```

    +--------------+-------+-------+---+
    |          City|  State|Country| id|
    +--------------+-------+-------+---+
    |    Clarksburg|     WV|    USA|CKB|
    |    Fort Dodge|     IA|    USA|FOD|
    |       Redmond|     OR|    USA|RDM|
    |        Valdez|     AK|    USA|VDZ|
    |       Lebanon|     NH|    USA|LEB|
    |         Aspen|     CO|    USA|ASE|
    | Rouyn-Noranda|     PQ| Canada|YUY|
    |       Iqaluit|Nunavut| Canada|YFB|
    |       Toronto|     ON| Canada|YYZ|
    |    Dillingham|     AK|    USA|DLG|
    |   Fort Nelson|     BC| Canada|YYE|
    |      Pellston|     MI|    USA|PLN|
    |     St. Cloud|     MN|    USA|STC|
    |St. Petersburg|     FL|    USA|PIE|
    |    Great Bend|     KS|    USA|GBD|
    |          Hilo|     HI|    USA|Big|
    |       Kelowna|     BC| Canada|YLW|
    |       Bemidji|     MN|    USA|BJI|
    |        London|     ON| Canada|YXU|
    |      Longview|     TX|    USA|GGG|
    +--------------+-------+-------+---+
    only showing top 20 rows
    
    +-------+-----+---+---+--------+---------+
    | tripid|delay|src|dst|city_dst|state_dst|
    +-------+-----+---+---+--------+---------+
    |1011245|    6|ABE|ATL| Atlanta|       GA|
    |1020600|   -8|ABE|DTW| Detroit|       MI|
    |1021245|   -2|ABE|ATL| Atlanta|       GA|
    |1020605|   -4|ABE|ATL| Atlanta|       GA|
    |1031245|   -4|ABE|ATL| Atlanta|       GA|
    |1030605|    0|ABE|ATL| Atlanta|       GA|
    |1041243|   10|ABE|ATL| Atlanta|       GA|
    |1040605|   28|ABE|ATL| Atlanta|       GA|
    |1051245|   88|ABE|ATL| Atlanta|       GA|
    |1050605|    9|ABE|ATL| Atlanta|       GA|
    |1061215|   -6|ABE|ATL| Atlanta|       GA|
    |1061725|   69|ABE|ATL| Atlanta|       GA|
    |1061230|    0|ABE|DTW| Detroit|       MI|
    |1060625|   -3|ABE|ATL| Atlanta|       GA|
    |1070600|    0|ABE|DTW| Detroit|       MI|
    |1071725|    0|ABE|ATL| Atlanta|       GA|
    |1071230|    0|ABE|DTW| Detroit|       MI|
    |1070625|    0|ABE|ATL| Atlanta|       GA|
    |1071219|    0|ABE|ORD| Chicago|       IL|
    |1080600|    0|ABE|DTW| Detroit|       MI|
    +-------+-----+---+---+--------+---------+
    only showing top 20 rows
    
    Airports: 526
    Trip routes: 1361141


#### Determining the longest delay in this dataset


```python
tripGraph.edges.groupBy().max("delay").show()
```

    +----------+
    |max(delay)|
    +----------+
    |      1642|
    +----------+
    



```python
# Finding the longest Delay
longestDelay = tripGraph.edges.groupBy().max("delay")
display(longestDelay)
```


    DataFrame[max(delay): int]


#### Determining the number of delayed vs. on-time / early flights


```python
tripGraph.edges.filter("delay <= 0").count()
```




    780469




```python
tripGraph.edges.filter("delay > 0").count()
```




    580672



#### What flights departing SEA are most likely to have significant delays
Note, delay can be <= 0 meaning the flight left on time or early


```python
tripGraph.edges.show()
tripGraph.edges\
          .filter("src = 'SEA' and delay > 0")\
          .groupBy("src", "dst")\
          .avg("delay")\
          .sort(desc("avg(delay)"))\
          .show(5)
```

    +-------+-----+---+---+--------+---------+
    | tripid|delay|src|dst|city_dst|state_dst|
    +-------+-----+---+---+--------+---------+
    |1011245|    6|ABE|ATL| Atlanta|       GA|
    |1020600|   -8|ABE|DTW| Detroit|       MI|
    |1021245|   -2|ABE|ATL| Atlanta|       GA|
    |1020605|   -4|ABE|ATL| Atlanta|       GA|
    |1031245|   -4|ABE|ATL| Atlanta|       GA|
    |1030605|    0|ABE|ATL| Atlanta|       GA|
    |1041243|   10|ABE|ATL| Atlanta|       GA|
    |1040605|   28|ABE|ATL| Atlanta|       GA|
    |1051245|   88|ABE|ATL| Atlanta|       GA|
    |1050605|    9|ABE|ATL| Atlanta|       GA|
    |1061215|   -6|ABE|ATL| Atlanta|       GA|
    |1061725|   69|ABE|ATL| Atlanta|       GA|
    |1061230|    0|ABE|DTW| Detroit|       MI|
    |1060625|   -3|ABE|ATL| Atlanta|       GA|
    |1070600|    0|ABE|DTW| Detroit|       MI|
    |1071725|    0|ABE|ATL| Atlanta|       GA|
    |1071230|    0|ABE|DTW| Detroit|       MI|
    |1070625|    0|ABE|ATL| Atlanta|       GA|
    |1071219|    0|ABE|ORD| Chicago|       IL|
    |1080600|    0|ABE|DTW| Detroit|       MI|
    +-------+-----+---+---+--------+---------+
    only showing top 20 rows
    
    +---+---+------------------+
    |src|dst|        avg(delay)|
    +---+---+------------------+
    |SEA|PHL|55.666666666666664|
    |SEA|COS| 43.53846153846154|
    |SEA|FAT| 43.03846153846154|
    |SEA|LGB| 39.39705882352941|
    |SEA|IAD|37.733333333333334|
    +---+---+------------------+
    only showing top 5 rows
    



```python
tripGraph.edges.filter("src = 'SEA' and delay > 0")\
                .groupBy("src", "dst")\
                .avg("delay")\
                .sort(desc("avg(delay)"))\
                .show()
```

    +---+---+------------------+
    |src|dst|        avg(delay)|
    +---+---+------------------+
    |SEA|PHL|55.666666666666664|
    |SEA|COS| 43.53846153846154|
    |SEA|FAT| 43.03846153846154|
    |SEA|LGB| 39.39705882352941|
    |SEA|IAD|37.733333333333334|
    |SEA|MIA|37.325581395348834|
    |SEA|SFO| 36.50210378681627|
    |SEA|SBA| 36.48275862068966|
    |SEA|JFK|          35.03125|
    |SEA|ORD| 33.60335195530726|
    |SEA|PDX| 32.74285714285714|
    |SEA|BOS| 30.46031746031746|
    |SEA|LAS|28.933333333333334|
    |SEA|DEN|28.881294964028775|
    |SEA|IAH|27.844444444444445|
    |SEA|JAC|27.666666666666668|
    |SEA|OGG|27.473684210526315|
    |SEA|JNU|27.196969696969695|
    |SEA|HNL|26.702290076335878|
    |SEA|OAK|26.539473684210527|
    +---+---+------------------+
    only showing top 20 rows
    


## Vertex Degrees
* `inDegrees`: Incoming connections to the airport
* `outDegrees`: Outgoing connections from the airport 
* `degrees`: Total connections to and from the airport

Reviewing the various properties of the property graph to understand the incoming and outgoing connections between airports.


```python
# Degrees
#  The number of degrees - the number of incoming and outgoing connections - for various airports within this sample dataset
tripGraph.degrees.show()
tripGraph.degrees.sort(desc("degree"))\
                    .limit(20)\
                    .show()
```

    +---+------+
    | id|degree|
    +---+------+
    |INL|    89|
    |MSY| 20560|
    |GEG|  4087|
    |SNA| 18720|
    |BUR| 10157|
    |GRB|  2219|
    |GTF|   850|
    |IDA|  1309|
    |GRR|  5173|
    |JLN|   180|
    |EUG|  2542|
    |PVD|  5790|
    |GSO|  3806|
    |MYR|   765|
    |OAK| 19950|
    |MSN|  4719|
    |FSM|  1200|
    |FAR|  2439|
    |BTM|   360|
    |COD|   360|
    +---+------+
    only showing top 20 rows
    
    +---+------+
    | id|degree|
    +---+------+
    |ATL|179774|
    |DFW|133966|
    |ORD|125405|
    |LAX|106853|
    |DEN|103699|
    |IAH| 85685|
    |PHX| 79672|
    |SFO| 77635|
    |LAS| 66101|
    |CLT| 56103|
    |EWR| 54407|
    |MCO| 54300|
    |LGA| 50927|
    |SLC| 50780|
    |BOS| 49936|
    |DTW| 46705|
    |MSP| 46235|
    |SEA| 45816|
    |JFK| 43661|
    |BWI| 42526|
    +---+------+
    



```python
# inDegrees
#  The number of degrees - the number of incoming connections
tripGraph.inDegrees.sort(desc("inDegree"))\
                    .limit(20)\
                    .show()
```

    +---+--------+
    | id|inDegree|
    +---+--------+
    |ATL|   89633|
    |DFW|   65767|
    |ORD|   61654|
    |LAX|   53184|
    |DEN|   50738|
    |IAH|   42512|
    |PHX|   39619|
    |SFO|   38641|
    |LAS|   32994|
    |CLT|   28044|
    |EWR|   27201|
    |MCO|   27071|
    |LGA|   25469|
    |SLC|   25169|
    |BOS|   24973|
    |DTW|   23297|
    |SEA|   22906|
    |MSP|   22372|
    |JFK|   21832|
    |BWI|   21262|
    +---+--------+
    



```python
# outDegrees
#  The number of degrees - the number of outgoing connections 
tripGraph.outDegrees.sort(desc("outDegree"))\
                    .limit(20)\
                    .show()
```

    +---+---------+
    | id|outDegree|
    +---+---------+
    |ATL|    90141|
    |DFW|    68199|
    |ORD|    63751|
    |LAX|    53669|
    |DEN|    52961|
    |IAH|    43173|
    |PHX|    40053|
    |SFO|    38994|
    |LAS|    33107|
    |CLT|    28059|
    |MCO|    27229|
    |EWR|    27206|
    |SLC|    25611|
    |LGA|    25458|
    |BOS|    24963|
    |MSP|    23863|
    |DTW|    23408|
    |SEA|    22910|
    |JFK|    21829|
    |BWI|    21264|
    +---+---------+
    


## City / Flight Relationships through Motif(主题) Finding
To understand the complex relationship of city airports and their flights with each other, we can __use motifs to find patterns of airports (i.e. vertices) connected by flights (i.e. edges)__.

The result is a DataFrame in which the column names are given by the motif keys.

#### What delays might we blame on SFO


```python
tripGraphPrime.find("(a)-[ab]->(b); (b)-[bc]->(c)").show()
```

    +--------------------+--------------------+--------------------+--------------------+-------------------+
    |                   a|                  ab|                   b|                  bc|                  c|
    +--------------------+--------------------+--------------------+--------------------+-------------------+
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011335,-4,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011550,-2,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011845,-12,MSY,...| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1010825,-1,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011505,-4,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1010650,-7,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011010,-3,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011910,19,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1011230,-8,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1010600,-4,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021335,10,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021550,81,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021845,-4,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...| [1020825,0,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021505,18,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1020650,-1,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...| [1021010,9,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021910,-5,MSY,DFW]|[Dallas,TX,USA,DFW]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1021230,-1,MSY,MIA]| [Miami,FL,USA,MIA]|
    |[Atlanta,GA,USA,ATL]|[1012110,-3,ATL,MSY]|[New Orleans,LA,U...|[1020600,-2,MSY,MIA]| [Miami,FL,USA,MIA]|
    +--------------------+--------------------+--------------------+--------------------+-------------------+
    only showing top 20 rows
    


DSL for expressing structural patterns:
The basic unit of a pattern is an edge.

- For example, "(a)-[e]->(b)" expresses an edge e from vertex a to vertex b. Note that vertices are denoted by parentheses (a), while edges are denoted by square brackets [e].

- A pattern is expressed as a union of edges. Edge patterns can be joined with semicolons. Motif "(a)-[e]->(b); (b)-[e2]->(c)" specifies two edges from a to b to c.

- It is acceptable to omit names for vertices or edges in motifs when not needed. E.g., "(a)-[]->(b)" expresses an edge between vertices a,b but does not assign a name to the edge. There will be no column for the anonymous edge in the result DataFrame. Similarly, "(a)-[e]->()" indicates an out-edge of vertex a but does not name the destination vertex.

- 


```python
# specifies two edges from a to b to c
motifs = tripGraphPrime.find("(a)-[ab]->(b); (b)-[bc]->(c)")\
                      .filter("(b.id = 'SFO') \
                                  and \
                                  (ab.delay > 500 or bc.delay > 500) \
                                  and \
                                  bc.tripid > ab.tripid \
                                  and \
                                  bc.tripid < ab.tripid + 10000")
motifs.show()
```

    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |                   a|                  ab|                   b|                  bc|                   c|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |[Albuquerque,NM,U...| [1020600,0,ABQ,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    |[Albuquerque,NM,U...|[1210815,-12,ABQ,...|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1011635,-15,ACV,...|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1012016,-4,ACV,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1020531,-2,ACV,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1020948,-11,ACV,...|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1021506,-3,ACV,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1021318,-9,ACV,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1020837,12,ACV,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1201703,-10,ACV,...|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]| [1210545,0,ACV,SFO]|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1211503,-10,ACV,...|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1210948,28,ACV,SFO]|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]| [1211316,9,ACV,SFO]|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    | [Eureka,CA,USA,ACV]|[1210838,91,ACV,SFO]|[San Francisco,CA...|[1211508,593,SFO,...|[New York,NY,USA,...|
    |[Anchorage,AK,USA...|[1012330,-4,ANC,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    |[Atlanta,GA,USA,ATL]|[1012130,24,ATL,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    |[Atlanta,GA,USA,ATL]| [1011625,6,ATL,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    |[Atlanta,GA,USA,ATL]|[1021106,26,ATL,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    |[Atlanta,GA,USA,ATL]| [1020944,7,ATL,SFO]|[San Francisco,CA...|[1021507,536,SFO,...|[New York,NY,USA,...|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    only showing top 20 rows
    


## Determining Airport Ranking using PageRank
There are a large number of flights and connections through these various airports included in this Departure Delay Dataset. 


__Using the `pageRank` algorithm, we can iteratively traverses the graph and determines a rough estimate of how important the airport is.__


```python
# Determining Airport ranking of importance using `pageRank`
ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
ranks
```




    GraphFrame(v:[id: string, City: string ... 3 more fields], e:[src: string, dst: string ... 5 more fields])




```python
ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(10).show()
```

    +--------------+-----+-------+---+------------------+
    |          City|State|Country| id|          pagerank|
    +--------------+-----+-------+---+------------------+
    |       Atlanta|   GA|    USA|ATL|10.102340247485012|
    |        Dallas|   TX|    USA|DFW| 7.252067259651102|
    |       Chicago|   IL|    USA|ORD| 7.165214941662068|
    |        Denver|   CO|    USA|DEN| 5.041255573485869|
    |   Los Angeles|   CA|    USA|LAX| 4.178333397888139|
    |       Houston|   TX|    USA|IAH| 4.008169343175302|
    | San Francisco|   CA|    USA|SFO| 3.518595203652925|
    |Salt Lake City|   UT|    USA|SLC|3.3564822581626763|
    |       Phoenix|   AZ|    USA|PHX|3.0896771274953343|
    |     Las Vegas|   NV|    USA|LAS| 2.437744837094217|
    +--------------+-----+-------+---+------------------+
    


## Most popular flights (single city hops)
Using the `tripGraph`, we can quickly determine what are the most popular single city hop flights


```python
# Determine the most popular flights (single city hops)
import pyspark.sql.functions as func
topTrips = tripGraph \
          .edges \
          .groupBy("src", "dst") \
          .agg(func.count("delay")\
          .alias("trips"))
            
topTrips.show()
```

    +---+---+-----+
    |src|dst|trips|
    +---+---+-----+
    |ATL|GSP|  507|
    |DSM|EWR|   71|
    |FSD|ATL|   89|
    |LAS|LIT|   90|
    |LBB|DEN|  175|
    |MCI|IAH|  576|
    |MCI|MKE|  155|
    |MDW|MEM|  173|
    |ORD|PDX|  544|
    |PBI|DCA|  148|
    |PHL|MCO| 1208|
    |ROC|CLE|   24|
    |SJC|LIH|   52|
    |SMF|BUR|  571|
    |SNA|PHX| 1023|
    |HRL|BRO|    1|
    |AUS|ELP|  195|
    |BMI|MCO|   26|
    |CAE|ATL|  667|
    |CLE|MCI|  132|
    +---+---+-----+
    only showing top 20 rows
    


> groupby + agg vs. groupby + pivot ?


```python
sc
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

df.groupBy('state').pivot('hq', values=['domestic', 'foreign']).sum('jobs').show()
```

    +-----+--------+-------+
    |state|domestic|foreign|
    +-----+--------+-------+
    |   MI|    2056|    168|
    |   OH|     394|    190|
    +-----+--------+-------+
    



```python
# Show the top 20 most popular flights (single city hops)
topTrips.orderBy(topTrips.trips.desc()).limit(10).show()
```

    +---+---+-----+
    |src|dst|trips|
    +---+---+-----+
    |SFO|LAX| 3232|
    |LAX|SFO| 3198|
    |LAS|LAX| 3016|
    |LAX|LAS| 2964|
    |JFK|LAX| 2720|
    |LAX|JFK| 2719|
    |ATL|LGA| 2501|
    |LGA|ATL| 2500|
    |LAX|PHX| 2394|
    |PHX|LAX| 2387|
    +---+---+-----+
    


## Top Transfer Cities
Many airports are used as transfer points instead of the final Destination.  

An easy way to calculate this is by calculating the ratio of inDegree (the number of flights to the airport) / outDegree (the number of flights leaving the airport).  


- Values close to 1 may indicate many transfers
- whereas values < 1 indicate many outgoing flights 
- and > 1 indicate many incoming flights




```python
inDeg = tripGraph.inDegrees
outDeg = tripGraph.outDegrees

# Calculate the degreeRatio (inDeg/outDeg)
degreeRatio = inDeg.join(outDeg, inDeg.id == outDeg.id) \
  .drop(outDeg.id) \
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio") \
  .cache()
degreeRatio.show()

nonTransferAirports = degreeRatio.join(airportsna, degreeRatio.id == airportsna.IATA) \
                                  .selectExpr("id", "city", "degreeRatio") \
                                  .filter("degreeRatio < .9 or degreeRatio > 1.1")
nonTransferAirports.show()
```

    +---+------------------+
    | id|       degreeRatio|
    +---+------------------+
    |MSY|1.0005838279653596|
    |GEG|0.9995107632093934|
    |BUR|0.9998031108485922|
    |SNA| 1.000213698044663|
    |GRB| 1.000901713255185|
    |GTF|               1.0|
    |IDA|1.0015290519877675|
    |GRR|1.0011605415860736|
    |EUG|               1.0|
    |GSO|1.0010515247108307|
    |MYR|0.9973890339425587|
    |PVD|               1.0|
    |OAK|1.0002005213555243|
    |BTM|               1.0|
    |COD| 1.011173184357542|
    |FAR|1.0008203445447088|
    |FSM|               1.0|
    |MQT|               1.0|
    |MSN|1.0004239084357778|
    |DCA|1.0005288207297727|
    +---+------------------+
    only showing top 20 rows
    
    +---+-----------+-------------------+
    | id|       city|        degreeRatio|
    +---+-----------+-------------------+
    |GFK|Grand Forks| 1.3333333333333333|
    |FAI|  Fairbanks| 1.1232686980609419|
    |OME|       Nome| 0.5084745762711864|
    |BRW|     Barrow|0.28651685393258425|
    +---+-----------+-------------------+
    



```python
transferAirports = degreeRatio.join(airportsna, degreeRatio.id == airportsna.IATA) \
                              .selectExpr("id", "city", "degreeRatio") \
                              .filter("degreeRatio between 0.9 and 1.1")
  
transferAirports.orderBy("degreeRatio").limit(10).show()
```

    +---+--------------+------------------+
    | id|          city|       degreeRatio|
    +---+--------------+------------------+
    |MSP|   Minneapolis|0.9375183338222353|
    |DEN|        Denver| 0.958025717037065|
    |DFW|        Dallas| 0.964339653074092|
    |ORD|       Chicago|0.9671063983310065|
    |SLC|Salt Lake City|0.9827417906368358|
    |IAH|       Houston|0.9846895050147083|
    |PHX|       Phoenix|0.9891643572266746|
    |OGG| Kahului, Maui|0.9898718478710211|
    |HNL|Honolulu, Oahu| 0.990535889872173|
    |SFO| San Francisco|0.9909473252295224|
    +---+--------------+------------------+
    


## Breadth First Search 
Breadth-first search (BFS) is designed to traverse the graph to quickly find the desired vertices (i.e. airports) and edges (i.e flights).  Let's try to find the shortest number of connections between cities based on the dataset.  Note, these examples do not take into account of time or distance, just hops between cities.


```python
# Example 1: Direct Seattle to San Francisco 
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SEA'",
  toExpr = "id = 'SFO'",
  maxPathLength = 1)

filteredPaths.withColumnRenamed('e0','suggested_path')\
            .select('suggested_path')\
            .toPandas()\
            .head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>suggested_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(1010710, 31, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(1012125, -4, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(1011840, -5, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(1010610, -4, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(1011230, -2, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(1010955, -6, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(1011100, 2, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(1011405, 0, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(1020710, -1, SEA, SFO, San Francisco, CA)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(1022125, -4, SEA, SFO, San Francisco, CA)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example 2: Direct San Francisco and Buffalo
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SFO'",
  toExpr = "id = 'BUF'",
  maxPathLength = 1)

filteredPaths.show()
```

    +----+-----+-------+---+
    |City|State|Country| id|
    +----+-----+-------+---+
    +----+-----+-------+---+
    


__There are no direct flights between San Francisco and Buffalo.__


```python
# Example 2, maxlength=2
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SFO'",
  toExpr = "id = 'BUF'",
  maxPathLength = 2)

filteredPaths.show()

```

    +--------------------+--------------------+-------------------+--------------------+--------------------+
    |                from|                  e0|                 v1|                  e1|                  to|
    +--------------------+--------------------+-------------------+--------------------+--------------------+
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1010635,-6,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1011059,13,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1011427,19,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1020635,-4,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1021059,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1021427,194,BOS,...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1030635,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1031059,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1031427,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1040635,16,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1041552,96,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1050635,1,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1051059,48,BOS,B...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1051427,443,BOS,...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1060635,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1061059,294,BOS,...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1061427,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1070730,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1071730,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    |[San Francisco,CA...|[1010700,0,SFO,BO...|[Boston,MA,USA,BOS]|[1080710,0,BOS,BU...|[Buffalo,NY,USA,BUF]|
    +--------------------+--------------------+-------------------+--------------------+--------------------+
    only showing top 20 rows
    


But there are flights from San Francisco to Buffalo with Minneapolis as the transfer point.  But what are the most popular layovers between `SFO` and `BUF`?


```python
# Display most popular layover cities by descending count
filteredPaths.groupBy("v1.id", "v1.City")\
            .count()\
            .orderBy(desc("count"))\
            .limit(10) \
            .show()
```

    +---+---------------+-------+
    | id|           City|  count|
    +---+---------------+-------+
    |JFK|       New York|1233728|
    |ORD|        Chicago|1088283|
    |ATL|        Atlanta| 285383|
    |LAS|      Las Vegas| 275091|
    |BOS|         Boston| 238576|
    |CLT|      Charlotte| 143444|
    |PHX|        Phoenix| 104580|
    |FLL|Fort Lauderdale|  96317|
    |EWR|         Newark|  95370|
    |MCO|        Orlando|  88615|
    +---+---------------+-------+
    


## Loading the D3 Visualization
Using the airports D3 visualization to visualize airports and flight paths


```python
%scala
package d3a
// We use a package object so that we can define top level classes like Edge that need to be used in other cells

import org.apache.spark.sql._
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

case class Edge(src: String, dest: String, count: Long)

case class Node(name: String)
case class Link(source: Int, target: Int, value: Long)
case class Graph(nodes: Seq[Node], links: Seq[Link])

object graphs {
val sqlContext = SQLContext.getOrCreate(org.apache.spark.SparkContext.getOrCreate())
import sqlContext.implicits._

def force(clicks: Dataset[Edge], height: Int = 100, width: Int = 960): Unit = {
  val data = clicks.collect()
  val nodes = (data.map(_.src) ++ data.map(_.dest)).map(_.replaceAll("_", " ")).toSet.toSeq.map(Node)
  val links = data.map { t =>
    Link(nodes.indexWhere(_.name == t.src.replaceAll("_", " ")), nodes.indexWhere(_.name == t.dest.replaceAll("_", " ")), t.count / 20 + 1)
  }
  showGraph(height, width, Seq(Graph(nodes, links)).toDF().toJSON.first())
}

/**
 * Displays a force directed graph using d3
 * input: {"nodes": [{"name": "..."}], "links": [{"source": 1, "target": 2, "value": 0}]}
 */
def showGraph(height: Int, width: Int, graph: String): Unit = {

displayHTML(s"""<!DOCTYPE html>
<html>
  <head>
    <link type="text/css" rel="stylesheet" href="https://mbostock.github.io/d3/talk/20111116/style.css"/>
    <style type="text/css">
      #states path {
        fill: #ccc;
        stroke: #fff;
      }

      path.arc {
        pointer-events: none;
        fill: none;
        stroke: #000;
        display: none;
      }

      path.cell {
        fill: none;
        pointer-events: all;
      }

      circle {
        fill: steelblue;
        fill-opacity: .8;
        stroke: #fff;
      }

      #cells.voronoi path.cell {
        stroke: brown;
      }

      #cells g:hover path.arc {
        display: inherit;
      }
    </style>
  </head>
  <body>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.csv.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geo.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geom.js"></script>
    <script>
      var graph = $graph;
      var w = $width;
      var h = $height;

      var linksByOrigin = {};
      var countByAirport = {};
      var locationByAirport = {};
      var positions = [];

      var projection = d3.geo.azimuthal()
          .mode("equidistant")
          .origin([-98, 38])
          .scale(1400)
          .translate([640, 360]);

      var path = d3.geo.path()
          .projection(projection);

      var svg = d3.select("body")
          .insert("svg:svg", "h2")
          .attr("width", w)
          .attr("height", h);

      var states = svg.append("svg:g")
          .attr("id", "states");

      var circles = svg.append("svg:g")
          .attr("id", "circles");

      var cells = svg.append("svg:g")
          .attr("id", "cells");

      var arc = d3.geo.greatArc()
          .source(function(d) { return locationByAirport[d.source]; })
          .target(function(d) { return locationByAirport[d.target]; });

      d3.select("input[type=checkbox]").on("change", function() {
        cells.classed("voronoi", this.checked);
      });

      // Draw US map.
      d3.json("https://mbostock.github.io/d3/talk/20111116/us-states.json", function(collection) {
        states.selectAll("path")
          .data(collection.features)
          .enter().append("svg:path")
          .attr("d", path);
      });

      // Parse links
      graph.links.forEach(function(link) {
        var origin = graph.nodes[link.source].name;
        var destination = graph.nodes[link.target].name;

        var links = linksByOrigin[origin] || (linksByOrigin[origin] = []);
        links.push({ source: origin, target: destination });

        countByAirport[origin] = (countByAirport[origin] || 0) + 1;
        countByAirport[destination] = (countByAirport[destination] || 0) + 1;
      });

      d3.csv("https://mbostock.github.io/d3/talk/20111116/airports.csv", function(data) {

        // Build list of airports.
        var airports = graph.nodes.map(function(node) {
          return data.find(function(airport) {
            if (airport.iata === node.name) {
              var location = [+airport.longitude, +airport.latitude];
              locationByAirport[airport.iata] = location;
              positions.push(projection(location));

              return true;
            } else {
              return false;
            }
          });
        });

        // Compute the Voronoi diagram of airports' projected positions.
        var polygons = d3.geom.voronoi(positions);

        var g = cells.selectAll("g")
            .data(airports)
          .enter().append("svg:g");

        g.append("svg:path")
            .attr("class", "cell")
            .attr("d", function(d, i) { return "M" + polygons[i].join("L") + "Z"; })
            .on("mouseover", function(d, i) { d3.select("h2 span").text(d.name); });

        g.selectAll("path.arc")
            .data(function(d) { return linksByOrigin[d.iata] || []; })
          .enter().append("svg:path")
            .attr("class", "arc")
            .attr("d", function(d) { return path(arc(d)); });

        circles.selectAll("circle")
            .data(airports)
            .enter().append("svg:circle")
            .attr("cx", function(d, i) { return positions[i][0]; })
            .attr("cy", function(d, i) { return positions[i][1]; })
            .attr("r", function(d, i) { return Math.sqrt(countByAirport[d.iata]); })
            .sort(function(a, b) { return countByAirport[b.iata] - countByAirport[a.iata]; });
      });
    </script>
  </body>
</html>""")
  }

  def help() = {
displayHTML("""
<p>
Produces a force-directed graph given a collection of edges of the following form:</br>
<tt><font color="#a71d5d">case class</font> <font color="#795da3">Edge</font>(<font color="#ed6a43">src</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">dest</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">count</font>: <font color="#a71d5d">Long</font>)</tt>
</p>
<p>Usage:<br/>
<tt>%scala</tt></br>
<tt><font color="#a71d5d">import</font> <font color="#ed6a43">d3._</font></tt><br/>
<tt><font color="#795da3">graphs.force</font>(</br>
&nbsp;&nbsp;<font color="#ed6a43">height</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">width</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">clicks</font>: <font color="#795da3">Dataset</font>[<font color="#795da3">Edge</font>])</tt>
</p>""")
  }
}
```


      File "<ipython-input-34-ecc21a99b802>", line 2
        package d3a
                  ^
    SyntaxError: invalid syntax




```python
%scala d3a.graphs.help()
```

#### Visualize On-time and Early Arrivals


```python
%scala
// On-time and Early Arrivals
import d3a._
graphs.force(
  height = 800,
  width = 1200,
  clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo where delay <= 0 group by src, dst""").as[Edge])
```

#### Visualize Delayed Trips Departing from the West Coast

Notice that most of the delayed trips are with Western US cities


```python
%scala
// Delayed Trips from CA, OR, and/or WA
import d3a._
graphs.force(
  height = 800,
  width = 1200,
  clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo where state_src in ('CA', 'OR', 'WA') and delay > 0 group by src, dst""").as[Edge])
```

#### Visualize All Flights (from this dataset)


```python
%scala
// Trips (from DepartureDelays Dataset)
import d3a._
graphs.force(
  height = 800,
  width = 1200,
  clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo group by src, dst""").as[Edge])
```
