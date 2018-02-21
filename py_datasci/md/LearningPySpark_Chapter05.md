
## Get to know the MLib package of PySpark through a mini project (infant dataset)

### Load and transform the data


```python
import pyspark.sql.types as typ

labels = [
    ('INFANT_ALIVE_AT_REPORT', typ.StringType()),
    ('BIRTH_YEAR', typ.IntegerType()),
    ('BIRTH_MONTH', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('MOTHER_RACE_6CODE', typ.StringType()),
    ('MOTHER_EDUCATION', typ.StringType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('FATHER_EDUCATION', typ.StringType()),
    ('MONTH_PRECARE_RECODE', typ.StringType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_BMI_RECODE', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.StringType()),
    ('DIABETES_GEST', typ.StringType()),
    ('HYP_TENS_PRE', typ.StringType()),
    ('HYP_TENS_GEST', typ.StringType()),
    ('PREV_BIRTH_PRETERM', typ.StringType()),
    ('NO_RISK', typ.StringType()),
    ('NO_INFECTIONS_REPORTED', typ.StringType()),
    ('LABOR_IND', typ.StringType()),
    ('LABOR_AUGM', typ.StringType()),
    ('STEROIDS', typ.StringType()),
    ('ANTIBIOTICS', typ.StringType()),
    ('ANESTHESIA', typ.StringType()),
    ('DELIV_METHOD_RECODE_COMB', typ.StringType()),
    ('ATTENDANT_BIRTH', typ.StringType()),
    ('APGAR_5', typ.IntegerType()),
    ('APGAR_5_RECODE', typ.StringType()),
    ('APGAR_10', typ.IntegerType()),
    ('APGAR_10_RECODE', typ.StringType()),
    ('INFANT_SEX', typ.StringType()),
    ('OBSTETRIC_GESTATION_WEEKS', typ.IntegerType()),
    ('INFANT_WEIGHT_GRAMS', typ.IntegerType()),
    ('INFANT_ASSIST_VENTI', typ.StringType()),
    ('INFANT_ASSIST_VENTI_6HRS', typ.StringType()),
    ('INFANT_NICU_ADMISSION', typ.StringType()),
    ('INFANT_SURFACANT', typ.StringType()),
    ('INFANT_ANTIBIOTICS', typ.StringType()),
    ('INFANT_SEIZURES', typ.StringType()),
    ('INFANT_NO_ABNORMALITIES', typ.StringType()),
    ('INFANT_ANCEPHALY', typ.StringType()),
    ('INFANT_MENINGOMYELOCELE', typ.StringType()),
    ('INFANT_LIMB_REDUCTION', typ.StringType()),
    ('INFANT_DOWN_SYNDROME', typ.StringType()),
    ('INFANT_SUSPECTED_CHROMOSOMAL_DISORDER', typ.StringType()),
    ('INFANT_NO_CONGENITAL_ANOMALIES_CHECKED', typ.StringType()),
    ('INFANT_BREASTFED', typ.StringType())
]

schema = typ.StructType([
        typ.StructField(e[0], e[1], False) for e in labels
    ])
```


```python
births = spark.read.csv('/home/karen/Downloads/data/births_transformed.csv.gz', 
                        header=True, 
                        schema=schema)

births.take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT='0', BIRTH_YEAR=1, BIRTH_MONTH=29, BIRTH_PLACE='99', MOTHER_AGE_YEARS=0, MOTHER_RACE_6CODE='0', MOTHER_EDUCATION='0', FATHER_COMBINED_AGE=0, FATHER_EDUCATION='99', MONTH_PRECARE_RECODE='999', CIG_BEFORE=999, CIG_1_TRI=99, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=0, MOTHER_BMI_RECODE=0, MOTHER_PRE_WEIGHT=0, MOTHER_DELIVERY_WEIGHT=None, MOTHER_WEIGHT_GAIN=None, DIABETES_PRE=None, DIABETES_GEST=None, HYP_TENS_PRE=None, HYP_TENS_GEST=None, PREV_BIRTH_PRETERM=None, NO_RISK=None, NO_INFECTIONS_REPORTED=None, LABOR_IND=None, LABOR_AUGM=None, STEROIDS=None, ANTIBIOTICS=None, ANESTHESIA=None, DELIV_METHOD_RECODE_COMB=None, ATTENDANT_BIRTH=None, APGAR_5=None, APGAR_5_RECODE=None, APGAR_10=None, APGAR_10_RECODE=None, INFANT_SEX=None, OBSTETRIC_GESTATION_WEEKS=None, INFANT_WEIGHT_GRAMS=None, INFANT_ASSIST_VENTI=None, INFANT_ASSIST_VENTI_6HRS=None, INFANT_NICU_ADMISSION=None, INFANT_SURFACANT=None, INFANT_ANTIBIOTICS=None, INFANT_SEIZURES=None, INFANT_NO_ABNORMALITIES=None, INFANT_ANCEPHALY=None, INFANT_MENINGOMYELOCELE=None, INFANT_LIMB_REDUCTION=None, INFANT_DOWN_SYNDROME=None, INFANT_SUSPECTED_CHROMOSOMAL_DISORDER=None, INFANT_NO_CONGENITAL_ANOMALIES_CHECKED=None, INFANT_BREASTFED=None)]




```python
births.columns
```




    ['INFANT_ALIVE_AT_REPORT', 'BIRTH_YEAR', 'BIRTH_MONTH', 'BIRTH_PLACE', 'MOTHER_AGE_YEARS', 'MOTHER_RACE_6CODE', 'MOTHER_EDUCATION', 'FATHER_COMBINED_AGE', 'FATHER_EDUCATION', 'MONTH_PRECARE_RECODE', 'CIG_BEFORE', 'CIG_1_TRI', 'CIG_2_TRI', 'CIG_3_TRI', 'MOTHER_HEIGHT_IN', 'MOTHER_BMI_RECODE', 'MOTHER_PRE_WEIGHT', 'MOTHER_DELIVERY_WEIGHT', 'MOTHER_WEIGHT_GAIN', 'DIABETES_PRE', 'DIABETES_GEST', 'HYP_TENS_PRE', 'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM', 'NO_RISK', 'NO_INFECTIONS_REPORTED', 'LABOR_IND', 'LABOR_AUGM', 'STEROIDS', 'ANTIBIOTICS', 'ANESTHESIA', 'DELIV_METHOD_RECODE_COMB', 'ATTENDANT_BIRTH', 'APGAR_5', 'APGAR_5_RECODE', 'APGAR_10', 'APGAR_10_RECODE', 'INFANT_SEX', 'OBSTETRIC_GESTATION_WEEKS', 'INFANT_WEIGHT_GRAMS', 'INFANT_ASSIST_VENTI', 'INFANT_ASSIST_VENTI_6HRS', 'INFANT_NICU_ADMISSION', 'INFANT_SURFACANT', 'INFANT_ANTIBIOTICS', 'INFANT_SEIZURES', 'INFANT_NO_ABNORMALITIES', 'INFANT_ANCEPHALY', 'INFANT_MENINGOMYELOCELE', 'INFANT_LIMB_REDUCTION', 'INFANT_DOWN_SYNDROME', 'INFANT_SUSPECTED_CHROMOSOMAL_DISORDER', 'INFANT_NO_CONGENITAL_ANOMALIES_CHECKED', 'INFANT_BREASTFED']




```python
selected_features = [
    'INFANT_ALIVE_AT_REPORT', 
    'BIRTH_PLACE', 
    'MOTHER_AGE_YEARS', 
    'FATHER_COMBINED_AGE', 
    'CIG_BEFORE', 
    'CIG_1_TRI', 
    'CIG_2_TRI', 
    'CIG_3_TRI', 
    'MOTHER_HEIGHT_IN', 
    'MOTHER_PRE_WEIGHT', 
    'MOTHER_DELIVERY_WEIGHT', 
    'MOTHER_WEIGHT_GAIN', 
    'DIABETES_PRE', 
    'DIABETES_GEST', 
    'HYP_TENS_PRE', 
    'HYP_TENS_GEST', 
    'PREV_BIRTH_PRETERM'
]

births_trimmed = births.select(selected_features)
births_trimmed.take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT='0', BIRTH_PLACE='99', MOTHER_AGE_YEARS=0, FATHER_COMBINED_AGE=0, CIG_BEFORE=999, CIG_1_TRI=99, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=0, MOTHER_PRE_WEIGHT=0, MOTHER_DELIVERY_WEIGHT=None, MOTHER_WEIGHT_GAIN=None, DIABETES_PRE=None, DIABETES_GEST=None, HYP_TENS_PRE=None, HYP_TENS_GEST=None, PREV_BIRTH_PRETERM=None)]



Specify the recoding methods.


```python
import pyspark.sql.functions as fn

def recode(col, key):        
    return recode_dictionary[key][col] 

def correct_cig(feature):
    return fn.when(fn.col(feature) != 99, fn.col(feature)).otherwise(0)
births_trimmed.take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT='0', BIRTH_PLACE='99', MOTHER_AGE_YEARS=0, FATHER_COMBINED_AGE=0, CIG_BEFORE=999, CIG_1_TRI=99, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=0, MOTHER_PRE_WEIGHT=0, MOTHER_DELIVERY_WEIGHT=None, MOTHER_WEIGHT_GAIN=None, DIABETES_PRE=None, DIABETES_GEST=None, HYP_TENS_PRE=None, HYP_TENS_GEST=None, PREV_BIRTH_PRETERM=None)]



Correct the features related to the number of smoked cigarettes.


```python
births_transformed = births_trimmed \
    .withColumn('CIG_BEFORE', correct_cig('CIG_BEFORE'))\
    .withColumn('CIG_1_TRI', correct_cig('CIG_1_TRI'))\
    .withColumn('CIG_2_TRI', correct_cig('CIG_2_TRI'))\
    .withColumn('CIG_3_TRI', correct_cig('CIG_3_TRI'))
births_transformed.take(1)
```




    [Row(INFANT_ALIVE_AT_REPORT='0', BIRTH_PLACE='99', MOTHER_AGE_YEARS=0, FATHER_COMBINED_AGE=0, CIG_BEFORE=999, CIG_1_TRI=0, CIG_2_TRI=0, CIG_3_TRI=0, MOTHER_HEIGHT_IN=0, MOTHER_PRE_WEIGHT=0, MOTHER_DELIVERY_WEIGHT=None, MOTHER_WEIGHT_GAIN=None, DIABETES_PRE=None, DIABETES_GEST=None, HYP_TENS_PRE=None, HYP_TENS_GEST=None, PREV_BIRTH_PRETERM=None)]



### Descriptive statistics


```python
import pyspark.mllib.stat as st
import numpy as np

numeric_cols = ['MOTHER_AGE_YEARS','FATHER_COMBINED_AGE',
                'CIG_BEFORE','CIG_1_TRI','CIG_2_TRI','CIG_3_TRI',
                'MOTHER_HEIGHT_IN','MOTHER_PRE_WEIGHT',
                'MOTHER_DELIVERY_WEIGHT','MOTHER_WEIGHT_GAIN'
               ]


numeric_rdd = births_transformed.select(numeric_cols).rdd.map(lambda row:[i for i in row])
numeric_rdd.take(1)
```




    [[0, 0, 999, 0, 0, 0, 0, 0, None, None]]




```python
mllib_stats = st.Statistics.colStats(numeric_rdd)

for col, m, v in zip(numeric_cols, 
                     mllib_stats.mean(), 
                     mllib_stats.variance()):
    print('{0}: \t{1:.2f} \t {2:.2f}'.format(col, m, np.sqrt(v)))
```

    MOTHER_AGE_YEARS: 	1.43 	 5.18
    FATHER_COMBINED_AGE: 	0.58 	 3.11
    CIG_BEFORE: 	223.63 	 180.01
    CIG_1_TRI: 	22.11 	 17.02
    CIG_2_TRI: 	0.01 	 0.11
    CIG_3_TRI: 	0.04 	 0.20
    MOTHER_HEIGHT_IN: 	0.02 	 0.15
    MOTHER_PRE_WEIGHT: 	0.05 	 0.22
    MOTHER_DELIVERY_WEIGHT: 	nan 	 nan
    MOTHER_WEIGHT_GAIN: 	nan 	 nan


For the categorical variables we will calculate the frequencies of their values.


```python
categorical_cols = [e for e in births_transformed.columns 
                    if e not in numeric_cols]

categorical_rdd = births_transformed\
                       .select(categorical_cols)\
                       .rdd \
                       .map(lambda row: [e for e in row])
            
categorical_rdd.take(10)
```




    [['0', '99', None, None, None, None, None], ['0', '29', None, None, None, None, None], ['0', '40', None, None, None, None, None], ['0', '42', None, None, None, None, None], ['0', '99', None, None, None, None, None], ['0', '37', None, None, None, None, None], ['0', '25', None, None, None, None, None], ['0', '26', None, None, None, None, None], ['0', '32', None, None, None, None, None], ['0', '66', None, None, None, None, None]]




```python
for i, col in enumerate(categorical_cols):
    agg = categorical_rdd \
        .groupBy(lambda row: row[i])\
        .map(lambda row: (row[0], len(row[1])))#how many records in each group
        
    print(col, sorted(agg.collect(), 
                      key=lambda el: el[1], #sort by tuple(1), which tuple('group',count(records))
                      reverse=True))
```

    INFANT_ALIVE_AT_REPORT [('1', 23349), ('0', 22080)]
    BIRTH_PLACE [('99', 8869), ('31', 2178), ('29', 2136), ('32', 2135), ('30', 2115), ('33', 2020), ('28', 1971), ('34', 1922), ('27', 1733), ('35', 1700), ('26', 1640), ('25', 1541), ('36', 1483), ('24', 1414), ('37', 1332), ('23', 1250), ('38', 1096), ('22', 1069), ('39', 999), ('21', 859), ('40', 831), ('20', 672), ('41', 639), ('42', 551), ('19', 471), ('43', 438), ('44', 358), ('45', 303), ('18', 292), ('46', 243), ('47', 169), ('48', 149), ('49', 140), ('17', 124), ('50', 110), ('51', 83), ('52', 71), ('53', 54), ('16', 50), ('54', 42), ('55', 38), ('56', 23), ('15', 19), ('60', 18), ('57', 17), ('58', 14), ('59', 10), ('62', 9), ('65', 6), ('63', 4), ('66', 3), ('64', 3), ('14', 3), ('67', 2), ('61', 2), ('74', 1), ('73', 1), ('70', 1), ('68', 1), ('13', 1), ('69', 1)]
    DIABETES_PRE [(None, 45429)]
    DIABETES_GEST [(None, 45429)]
    HYP_TENS_PRE [(None, 45429)]
    HYP_TENS_GEST [(None, 45429)]
    PREV_BIRTH_PRETERM [(None, 45429)]


### Correlations

Correlations between our features.


```python
corrs = st.Statistics.corr(numeric_rdd)
corrs
```




    [[ 1.00000000e+00  6.23033578e-01 -4.67241960e-03  7.39077397e-03
       6.05002312e-03  2.21591500e-03  4.97124699e-03  5.27505697e-02
                  nan             nan]
     [ 6.23033578e-01  1.00000000e+00  4.83568294e-03 -1.28424070e-02
       2.92741491e-03 -2.71574945e-03 -2.79045184e-04  4.26866363e-02
                  nan             nan]
     [-4.67241960e-03  4.83568294e-03  1.00000000e+00 -2.45453996e-01
       1.51152123e-02 -9.06643091e-03  2.82410695e-02 -7.26174588e-03
                  nan             nan]
     [ 7.39077397e-03 -1.28424070e-02 -2.45453996e-01  1.00000000e+00
      -1.68567893e-02  9.83866079e-03 -2.59417178e-02 -6.46632488e-02
                  nan             nan]
     [ 6.05002312e-03  2.92741491e-03  1.51152123e-02 -1.68567893e-02
       1.00000000e+00 -2.35761167e-02  1.54738945e-01  2.98806718e-02
                  nan             nan]
     [ 2.21591500e-03 -2.71574945e-03 -9.06643091e-03  9.83866079e-03
      -2.35761167e-02  1.00000000e+00  4.52474053e-02  1.56481885e-02
                  nan             nan]
     [ 4.97124699e-03 -2.79045184e-04  2.82410695e-02 -2.59417178e-02
       1.54738945e-01  4.52474053e-02  1.00000000e+00  4.46162996e-02
                  nan             nan]
     [ 5.27505697e-02  4.26866363e-02 -7.26174588e-03 -6.46632488e-02
       2.98806718e-02  1.56481885e-02  4.46162996e-02  1.00000000e+00
                  nan             nan]
     [            nan             nan             nan             nan
                  nan             nan             nan             nan
       1.00000000e+00             nan]
     [            nan             nan             nan             nan
                  nan             nan             nan             nan
                  nan  1.00000000e+00]]




```python
for i, el in enumerate(corrs > 0.5):
    correlated = [
        (numeric_cols[j], corrs[i][j]) 
        for j, e in enumerate(el) 
        if e == 1.0 and j != i] # non-diagonal and highly correlated
    # print(correlated)
    if len(correlated) > 0:
        for e in correlated:
            print('{}-to-{}: {:.5f}' \
                  .format(numeric_cols[i], e[0], e[1]))
```

    /tmp/kernel-PySpark-4ae924a7-7d8a-43cf-bf04-7418a47a044b/pyspark_runner.py:1: RuntimeWarning: invalid value encountered in greater
      #
    MOTHER_AGE_YEARS-to-FATHER_COMBINED_AGE: 0.62303
    FATHER_COMBINED_AGE-to-MOTHER_AGE_YEARS: 0.62303


We can drop most of highly correlated features. 


```python
'''
selected_features = [
    'INFANT_ALIVE_AT_REPORT', 
    'BIRTH_PLACE', 
    'MOTHER_AGE_YEARS', 
    'FATHER_COMBINED_AGE', 
    'CIG_BEFORE', 
    'CIG_1_TRI', 
    'CIG_2_TRI', 
    'CIG_3_TRI', 
    'MOTHER_HEIGHT_IN', 
    'MOTHER_PRE_WEIGHT', 
    'MOTHER_DELIVERY_WEIGHT', 
    'MOTHER_WEIGHT_GAIN', 
    'DIABETES_PRE', 
    'DIABETES_GEST', 
    'HYP_TENS_PRE', 
    'HYP_TENS_GEST', 
    'PREV_BIRTH_PRETERM'
]
'''
features_to_keep = [
    'INFANT_ALIVE_AT_REPORT', 
    'BIRTH_PLACE', 
    'MOTHER_AGE_YEARS', 
    'FATHER_COMBINED_AGE', 
    'CIG_1_TRI', 
    'MOTHER_HEIGHT_IN', 
    'MOTHER_PRE_WEIGHT', 
    'DIABETES_PRE', 
    'DIABETES_GEST', 
    'HYP_TENS_PRE', 
    'HYP_TENS_GEST', 
    'PREV_BIRTH_PRETERM'
]
births_transformed = births_transformed.select([e for e in features_to_keep])
```

### Statistical testing

Run a Chi-square test to determine if there are significant differences for categorical variables.


```python
import pyspark.mllib.linalg as ln

for cat in categorical_cols[1:]:
    agg = births_transformed \
        .groupby('INFANT_ALIVE_AT_REPORT') \
        .pivot(cat) \
        .count()    
    print(agg.collect())
```

    [Row(INFANT_ALIVE_AT_REPORT='0', 13=1, 14=2, 15=5, 16=22, 17=61, 18=150, 19=228, 20=304, 21=401, 22=488, 23=563, 24=634, 25=670, 26=749, 27=734, 28=827, 29=929, 30=912, 31=917, 32=840, 33=807, 34=828, 35=721, 36=653, 37=564, 38=523, 39=451, 40=387, 41=317, 42=249, 43=203, 44=175, 45=137, 46=114, 47=82, 48=79, 49=70, 50=64, 51=45, 52=43, 53=28, 54=20, 55=16, 56=14, 57=8, 58=9, 59=4, 60=11, 61=2, 62=2, 63=4, 64=2, 65=3, 66=1, 67=2, 68=1, 69=1, 70=None, 73=None, 74=None, 99=6003), Row(INFANT_ALIVE_AT_REPORT='1', 13=None, 14=1, 15=14, 16=28, 17=63, 18=142, 19=243, 20=368, 21=458, 22=581, 23=687, 24=780, 25=871, 26=891, 27=999, 28=1144, 29=1207, 30=1203, 31=1261, 32=1295, 33=1213, 34=1094, 35=979, 36=830, 37=768, 38=573, 39=548, 40=444, 41=322, 42=302, 43=235, 44=183, 45=166, 46=129, 47=87, 48=70, 49=70, 50=46, 51=38, 52=28, 53=26, 54=22, 55=22, 56=9, 57=9, 58=5, 59=6, 60=7, 61=None, 62=7, 63=None, 64=1, 65=3, 66=2, 67=None, 68=None, 69=None, 70=1, 73=1, 74=1, 99=2866)]
    [Row(INFANT_ALIVE_AT_REPORT='0', null=22080), Row(INFANT_ALIVE_AT_REPORT='1', null=23349)]
    [Row(INFANT_ALIVE_AT_REPORT='0', null=22080), Row(INFANT_ALIVE_AT_REPORT='1', null=23349)]
    [Row(INFANT_ALIVE_AT_REPORT='0', null=22080), Row(INFANT_ALIVE_AT_REPORT='1', null=23349)]
    [Row(INFANT_ALIVE_AT_REPORT='0', null=22080), Row(INFANT_ALIVE_AT_REPORT='1', null=23349)]
    [Row(INFANT_ALIVE_AT_REPORT='0', null=22080), Row(INFANT_ALIVE_AT_REPORT='1', null=23349)]



```python
for cat in categorical_cols[1:]:
    agg = births_transformed \
        .groupby('INFANT_ALIVE_AT_REPORT') \
        .pivot(cat) \
        .count()    
    agg_rdd = agg \
        .rdd\
        .map(lambda row: (row[1:])) \
        .flatMap(lambda row: 
                 [0 if e == None else e for e in row]) \
        .collect()
    print(agg_rdd)
```

    [1, 2, 5, 22, 61, 150, 228, 304, 401, 488, 563, 634, 670, 749, 734, 827, 929, 912, 917, 840, 807, 828, 721, 653, 564, 523, 451, 387, 317, 249, 203, 175, 137, 114, 82, 79, 70, 64, 45, 43, 28, 20, 16, 14, 8, 9, 4, 11, 2, 2, 4, 2, 3, 1, 2, 1, 1, 0, 0, 0, 6003, 0, 1, 14, 28, 63, 142, 243, 368, 458, 581, 687, 780, 871, 891, 999, 1144, 1207, 1203, 1261, 1295, 1213, 1094, 979, 830, 768, 573, 548, 444, 322, 302, 235, 183, 166, 129, 87, 70, 70, 46, 38, 28, 26, 22, 22, 9, 9, 5, 6, 7, 0, 7, 0, 1, 3, 2, 0, 0, 0, 1, 1, 1, 2866]
    [22080, 23349]
    [22080, 23349]
    [22080, 23349]
    [22080, 23349]
    [22080, 23349]



```python
for cat in categorical_cols[1:]:
    agg = births_transformed \
        .groupby('INFANT_ALIVE_AT_REPORT') \
        .pivot(cat) \
        .count()    
    agg_rdd = agg \
        .rdd\
        .map(lambda row: (row[1:])) \
        .flatMap(lambda row: 
                 [0 if e == None else e for e in row]) \
        .collect()

    row_length = len(agg.collect()[0]) - 1
    agg = ln.Matrices.dense(row_length, 2, agg_rdd)
    print(agg)
```

    DenseMatrix([[1.000e+00, 0.000e+00],
                 [2.000e+00, 1.000e+00],
                 [5.000e+00, 1.400e+01],
                 [2.200e+01, 2.800e+01],
                 [6.100e+01, 6.300e+01],
                 [1.500e+02, 1.420e+02],
                 [2.280e+02, 2.430e+02],
                 [3.040e+02, 3.680e+02],
                 [4.010e+02, 4.580e+02],
                 [4.880e+02, 5.810e+02],
                 [5.630e+02, 6.870e+02],
                 [6.340e+02, 7.800e+02],
                 [6.700e+02, 8.710e+02],
                 [7.490e+02, 8.910e+02],
                 [7.340e+02, 9.990e+02],
                 [8.270e+02, 1.144e+03],
                 [9.290e+02, 1.207e+03],
                 [9.120e+02, 1.203e+03],
                 [9.170e+02, 1.261e+03],
                 [8.400e+02, 1.295e+03],
                 [8.070e+02, 1.213e+03],
                 [8.280e+02, 1.094e+03],
                 [7.210e+02, 9.790e+02],
                 [6.530e+02, 8.300e+02],
                 [5.640e+02, 7.680e+02],
                 [5.230e+02, 5.730e+02],
                 [4.510e+02, 5.480e+02],
                 [3.870e+02, 4.440e+02],
                 [3.170e+02, 3.220e+02],
                 [2.490e+02, 3.020e+02],
                 [2.030e+02, 2.350e+02],
                 [1.750e+02, 1.830e+02],
                 [1.370e+02, 1.660e+02],
                 [1.140e+02, 1.290e+02],
                 [8.200e+01, 8.700e+01],
                 [7.900e+01, 7.000e+01],
                 [7.000e+01, 7.000e+01],
                 [6.400e+01, 4.600e+01],
                 [4.500e+01, 3.800e+01],
                 [4.300e+01, 2.800e+01],
                 [2.800e+01, 2.600e+01],
                 [2.000e+01, 2.200e+01],
                 [1.600e+01, 2.200e+01],
                 [1.400e+01, 9.000e+00],
                 [8.000e+00, 9.000e+00],
                 [9.000e+00, 5.000e+00],
                 [4.000e+00, 6.000e+00],
                 [1.100e+01, 7.000e+00],
                 [2.000e+00, 0.000e+00],
                 [2.000e+00, 7.000e+00],
                 [4.000e+00, 0.000e+00],
                 [2.000e+00, 1.000e+00],
                 [3.000e+00, 3.000e+00],
                 [1.000e+00, 2.000e+00],
                 [2.000e+00, 0.000e+00],
                 [1.000e+00, 0.000e+00],
                 [1.000e+00, 0.000e+00],
                 [0.000e+00, 1.000e+00],
                 [0.000e+00, 1.000e+00],
                 [0.000e+00, 1.000e+00],
                 [6.003e+03, 2.866e+03]])
    DenseMatrix([[22080., 23349.]])
    
    DenseMatrix([[22080., 23349.]])
    
    DenseMatrix([[22080., 23349.]])
    
    DenseMatrix([[22080., 23349.]])
    
    DenseMatrix([[22080., 23349.]])
    



```python
for cat in categorical_cols[1:]:
    agg = births_transformed \
        .groupby('INFANT_ALIVE_AT_REPORT') \
        .pivot(cat) \
        .count()    
    agg_rdd = agg \
        .rdd\
        .map(lambda row: (row[1:])) \
        .flatMap(lambda row: 
                 [0 if e == None else e for e in row]) \
        .collect()

    row_length = len(agg.collect()[0]) - 1
    agg = ln.Matrices.dense(row_length, 2, agg_rdd)
    
    test = st.Statistics.chiSqTest(agg)
    print(cat, round(test.pValue, 4))
```

    BIRTH_PLACE 0.0
    DIABETES_PRE 1.0
    DIABETES_GEST 1.0
    HYP_TENS_PRE 1.0
    HYP_TENS_GEST 1.0
    PREV_BIRTH_PRETERM 1.0


## Create the final dataset

### Create an RDD of `LabeledPoint`

use a hashing trick to encode the `'BIRTH_PLACE'` feature.


```python
import pyspark.mllib.feature as ft
import pyspark.mllib.regression as reg

hashing = ft.HashingTF(7)

births_hashed = births_transformed \
                .rdd \
                .map(lambda row: [
                        list(hashing.transform(row[1]).toArray()) 
                            if col == 'BIRTH_PLACE' 
                            else row[i] 
                        for i, col 
                        in enumerate(features_to_keep)]) 

for i in births_hashed.take(3):
    print(i)
```

    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0, 0, 0, 0, None, None, None, None, None]
    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0, 18, 0, 0, None, None, None, None, None]
    ['0', [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], 0, 0, 12, 0, 0, None, None, None, None, None]



```python
births_hashed = births_transformed \
                .rdd \
                .map(lambda row: [
                        list(hashing.transform(row[1]).toArray()) 
                            if col == 'BIRTH_PLACE' 
                            else row[i] 
                        for i, col 
                        in enumerate(features_to_keep)]) \
                .map(lambda row: [[e] if type(e) == int else e 
                                  for e in row]) 
for i in births_hashed.take(3):
    print(i)
```

    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0], [0], [0], [0], [0], None, None, None, None, None]
    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0], [0], [18], [0], [0], None, None, None, None, None]
    ['0', [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0], [0], [12], [0], [0], None, None, None, None, None]



```python
births_hashed = births_transformed \
                .rdd \
                .map(lambda row: [
                        list(hashing.transform(row[1]).toArray()) 
                            if col == 'BIRTH_PLACE' 
                            else row[i] 
                        for i, col 
                        in enumerate(features_to_keep)]) \
                .map(lambda row: [[e] if type(e) == int else e 
                                  for e in row]) \
                .map(lambda row: [[0] if i is None else i for i in row]) 
#                 .map(lambda row: [item for sublist in row
#                                   for item in sublist ]) 
for i in births_hashed.take(3):
    print(i)
```

    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    ['0', [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0], [0], [18], [0], [0], [0], [0], [0], [0], [0]]
    ['0', [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0], [0], [12], [0], [0], [0], [0], [0], [0], [0]]



```python
births_hashed = births_transformed \
                .rdd \
                .map(lambda row: [
                        list(hashing.transform(row[1]).toArray()) 
                            if col == 'BIRTH_PLACE' 
                            else row[i] 
                        for i, col 
                        in enumerate(features_to_keep)]) \
                .map(lambda row: [[e] if type(e) == int else e 
                                  for e in row]) \
                .map(lambda row: [[0] if i is None else i for i in row]) \
                .map(lambda row: [item for sublist in row
                                         for item in sublist ]) 

for i in births_hashed.take(3):
    print(i)
```

    ['0', 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ['0', 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0]
    ['0', 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0]



```python
births_hashed = births_transformed \
                .rdd \
                .map(lambda row: [
                        list(hashing.transform(row[1]).toArray()) 
                            if col == 'BIRTH_PLACE' 
                            else row[i] 
                        for i, col 
                        in enumerate(features_to_keep)]) \
                .map(lambda row: [[e] if type(e) == int else e 
                                  for e in row]) \
                .map(lambda row: [[0] if i is None else i for i in row]) \
                .map(lambda row: [item for sublist in row
                                         for item in sublist ]) \
                .map(lambda row: reg.LabeledPoint(
                        row[0], 
                        ln.Vectors.dense(row[1:])))
for i in births_hashed.take(3):
    print(i)
```

    (0.0,[2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    (0.0,[2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    (0.0,[0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])


### Split into training and testing


```python
births_train, births_test = births_hashed.randomSplit([0.6, 0.4])
```

## Predicting infant survival

### Logistic regression in Spark

MLLib used to provide a logistic regression model estimated using a stochastic gradient descent (SGD) algorithm. 



```python
from pyspark.mllib.classification import LogisticRegressionWithSGD
LR_Model = LogisticRegressionWithSGD.train(births_train)
```

    /usr/local/lib/python3.5/dist-packages/pyspark/mllib/classification.py:313: UserWarning: Deprecated in 2.0.0. Use ml.classification.LogisticRegression or LogisticRegressionWithLBFGS.
      "Deprecated in 2.0.0. Use ml.classification.LogisticRegression or "



```python
LR_results = (
        births_test.map(lambda row: row.label) \
        .zip(LR_Model \
             .predict(births_test\
                      .map(lambda row: row.features)))
    ).map(lambda row: (row[0], row[1] * 1.0))
```


```python
import pyspark.mllib.evaluation as ev
LR_evaluation = ev.BinaryClassificationMetrics(LR_results)

print('Area under PR: {0:.2f}' \
      .format(LR_evaluation.areaUnderPR))
print('Area under ROC: {0:.2f}' \
      .format(LR_evaluation.areaUnderROC))
LR_evaluation.unpersist()
```

    Area under PR: 0.92
    Area under ROC: 0.67


### Selecting only the most predictable features

MLLib allows us to select the most predictable features using a Chi-Square selector.


```python
selector = ft.ChiSqSelector(4).fit(births_train)
selector
```




    <pyspark.mllib.feature.ChiSqSelectorModel object at 0x7f243bbdd668>




```python
topFeatures_train = (
        births_train.map(lambda row: row.label) \
        .zip(selector \
             .transform(births_train \
                        .map(lambda row: row.features)))
    ).map(lambda row: reg.LabeledPoint(row[0], row[1]))

for i in topFeatures_train.take(10):
    print(i)
```

    (0.0,[2.0,0.0,0.0,0.0])
    (0.0,[2.0,0.0,18.0,0.0])
    (0.0,[1.0,0.0,24.0,1.0])
    (0.0,[1.0,0.0,36.0,0.0])
    (0.0,[1.0,1.0,7.0,0.0])
    (0.0,[2.0,0.0,7.0,0.0])
    (0.0,[2.0,0.0,25.0,0.0])
    (0.0,[2.0,0.0,0.0,1.0])
    (0.0,[1.0,0.0,21.0,0.0])
    (0.0,[1.0,1.0,4.0,0.0])



```python
topFeatures_test = (
        births_test.map(lambda row: row.label) \
        .zip(selector \
             .transform(births_test \
                        .map(lambda row: row.features)))
    ).map(lambda row: reg.LabeledPoint(row[0], row[1]))

for i in topFeatures_test.take(10):
    print(i)
```

    (0.0,[0.0,1.0,12.0,0.0])
    (0.0,[2.0,0.0,20.0,0.0])
    (0.0,[0.0,1.0,12.0,0.0])
    (0.0,[1.0,0.0,33.0,0.0])
    (0.0,[0.0,0.0,10.0,0.0])
    (0.0,[2.0,0.0,0.0,0.0])
    (0.0,[2.0,0.0,5.0,0.0])
    (0.0,[1.0,0.0,36.0,1.0])
    (0.0,[2.0,0.0,0.0,0.0])
    (0.0,[1.0,0.0,59.0,1.0])


### Random Forest in Spark



```python
from pyspark.mllib.tree import RandomForest

RF_model = RandomForest \
    .trainClassifier(data=topFeatures_train, 
                     numClasses=2, 
                     categoricalFeaturesInfo={}, 
                     numTrees=50,  
                     featureSubsetStrategy='all',
                     seed=666)
```

Let's see how well our model did.


```python
RF_results = (
        topFeatures_test.map(lambda row: row.label) \
        .zip(RF_model \
             .predict(topFeatures_test \
                      .map(lambda row: row.features)))
    )

RF_evaluation = ev.BinaryClassificationMetrics(RF_results)

print('Area under PR: {0:.2f}' \
      .format(RF_evaluation.areaUnderPR))
print('Area under ROC: {0:.2f}' \
      .format(RF_evaluation.areaUnderROC))
RF_evaluation.unpersist()
```

    Area under PR: 0.81
    Area under ROC: 0.72


Let's see how the logistic regression would perform with reduced number of features.


```python
LR_Model_2 = LogisticRegressionWithSGD \
    .train(topFeatures_train, iterations=10) # here train with topFeatures_train

LR_results_2 = (
        topFeatures_test.map(lambda row: row.label) \
        .zip(LR_Model_2 \
             .predict(topFeatures_test \
                      .map(lambda row: row.features)))
    ).map(lambda row: (row[0], row[1] * 1.0))

LR_evaluation_2 = ev.BinaryClassificationMetrics(LR_results_2)

print('Area under PR: {0:.2f}' \
      .format(LR_evaluation_2.areaUnderPR))
print('Area under ROC: {0:.2f}' \
      .format(LR_evaluation_2.areaUnderROC))
LR_evaluation_2.unpersist()
```

    Area under PR: 0.93
    Area under ROC: 0.66


[精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么？](https://www.zhihu.com/question/30643044)
