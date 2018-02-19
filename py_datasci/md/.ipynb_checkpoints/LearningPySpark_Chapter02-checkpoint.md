
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


