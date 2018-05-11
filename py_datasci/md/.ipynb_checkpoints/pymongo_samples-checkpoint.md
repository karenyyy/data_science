

```python
import pymongo
import datetime
from pymongo import MongoClient
from pprint import pprint
```


```python
client = MongoClient("localhost", 27017)
db = client.mydb2
```


```python
db.collection_names(include_system_collections=False)
for i in db.mydb2.find():
    pprint(i)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'carrot', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 103.0,
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'dob': datetime.datetime(1985, 7, 4, 6, 1),
     'gender': 'f',
     'loves': ['apple', 'carrot', 'chocolate'],
     'name': 'Solnara',
     'vaccinated': True,
     'vampires': 80.0,
     'weight': 550.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'),
     'dob': datetime.datetime(1998, 3, 7, 13, 30),
     'gender': 'f',
     'loves': ['strawberry', 'lemon'],
     'name': 'Ayna',
     'vaccinated': True,
     'vampires': 40.0,
     'weight': 733.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'),
     'dob': datetime.datetime(1997, 7, 1, 14, 42),
     'gender': 'm',
     'loves': ['grape', 'lemon'],
     'name': 'Kenny',
     'vaccinated': True,
     'vampires': 39.0,
     'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'dob': datetime.datetime(2005, 5, 3, 4, 57),
     'gender': 'm',
     'loves': ['apple', 'sugar'],
     'name': 'Raleigh',
     'vaccinated': True,
     'vampires': 2.0,
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960d'),
     'dob': datetime.datetime(2001, 10, 8, 18, 53),
     'gender': 'f',
     'loves': ['apple', 'watermelon'],
     'name': 'Leia',
     'vaccinated': True,
     'vampires': 33.0,
     'weight': 601.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'),
     'dob': datetime.datetime(1997, 3, 1, 10, 3),
     'gender': 'm',
     'loves': ['apple', 'watermelon'],
     'name': 'Pilot',
     'vaccinated': True,
     'vampires': 54.0,
     'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960f'),
     'dob': datetime.datetime(1999, 12, 20, 21, 15),
     'gender': 'f',
     'loves': ['grape', 'carrot'],
     'name': 'Nimue',
     'vaccinated': True,
     'weight': 540.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}



```python
for doc in db.mydb2.find({'gender': 'm'}):
    pprint(doc)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 105.0,
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'),
     'dob': datetime.datetime(1997, 7, 1, 14, 42),
     'gender': 'm',
     'loves': ['grape', 'lemon'],
     'name': 'Kenny',
     'vaccinated': True,
     'vampires': 39.0,
     'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'dob': datetime.datetime(2005, 5, 3, 4, 57),
     'gender': 'm',
     'loves': ['apple', 'sugar'],
     'name': 'Raleigh',
     'vaccinated': True,
     'vampires': 2.0,
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'),
     'dob': datetime.datetime(1997, 3, 1, 10, 3),
     'gender': 'm',
     'loves': ['apple', 'watermelon'],
     'name': 'Pilot',
     'vaccinated': True,
     'vampires': 54.0,
     'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}



```python
for doc in db.mydb2.find({'gender': 'm', 'weight': {'$gt': 700}}):
    pprint(doc)
```

    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}



```python
db.mydb2.find_one({'name': 'Roooooodles'})
```




    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 105.0,
     'weight': 590.0}




```python
db.mydb2.update({'name': 'Roooooodles'}, {'weight': 590.0})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.





    {'n': 1, 'nModified': 1, 'ok': 1, 'updatedExisting': True}




```python
db.mydb2.update({'weight': 590.0}, {'$set': {
    'name': 'Roooooodles',
    'dob': datetime.datetime(1979, 7, 18, 18, 44),
    'loves': ['apple'],
    'gender': 'm',
'vampires': 99.0}})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:6: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      





    {'n': 1, 'nModified': 1, 'ok': 1, 'updatedExisting': True}




```python
db.mydb2.find_one({'name': 'Roooooodles'})
```




    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'gender': 'm',
     'loves': ['apple'],
     'name': 'Roooooodles',
     'vampires': 99.0,
     'weight': 590.0}




```python
db.mydb2.update({'name': 'Roooooodles'}, {'$inc': {'vampires': -2}})
db.mydb2.find_one({'name': 'Roooooodles'})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.





    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'gender': 'm',
     'loves': ['apple'],
     'name': 'Roooooodles',
     'vampires': 97.0,
     'weight': 590.0}




```python
db.mydb2.update({'name': 'Roooooodles'}, {'$push': {'loves': 'carrot'}})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.





    {'n': 1, 'nModified': 1, 'ok': 1, 'updatedExisting': True}




```python
db.mydb2.find_one({'name': 'Roooooodles'})
```




    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'gender': 'm',
     'loves': ['apple', 'carrot'],
     'name': 'Roooooodles',
     'vampires': 97.0,
     'weight': 590.0}




```python
db.mydb2.update({'name': 'Roooooodles'}, {'$push': {'loves': {'$each': ['apple', 'durian']}}, '$inc': {'vampires': 6, 'eggs': 1}})
db.mydb2.find_one({'name': 'Roooooodles'})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.





    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'carrot', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vampires': 103.0,
     'weight': 590.0}




```python
# By default, update modifies just one document4
db.mydb2.update({}, {'$set': {'vaccinated': True}})
for i in db.mydb2.find({'vaccinated': True}):
    pprint(i)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'dob': datetime.datetime(1985, 7, 4, 6, 1),
     'gender': 'f',
     'loves': ['apple', 'carrot', 'chocolate'],
     'name': 'Solnara',
     'vaccinated': True,
     'vampires': 80.0,
     'weight': 550.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'),
     'dob': datetime.datetime(1998, 3, 7, 13, 30),
     'gender': 'f',
     'loves': ['strawberry', 'lemon'],
     'name': 'Ayna',
     'vaccinated': True,
     'vampires': 40.0,
     'weight': 733.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'),
     'dob': datetime.datetime(1997, 7, 1, 14, 42),
     'gender': 'm',
     'loves': ['grape', 'lemon'],
     'name': 'Kenny',
     'vaccinated': True,
     'vampires': 39.0,
     'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'dob': datetime.datetime(2005, 5, 3, 4, 57),
     'gender': 'm',
     'loves': ['apple', 'sugar'],
     'name': 'Raleigh',
     'vaccinated': True,
     'vampires': 2.0,
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960d'),
     'dob': datetime.datetime(2001, 10, 8, 18, 53),
     'gender': 'f',
     'loves': ['apple', 'watermelon'],
     'name': 'Leia',
     'vaccinated': True,
     'vampires': 33.0,
     'weight': 601.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'),
     'dob': datetime.datetime(1997, 3, 1, 10, 3),
     'gender': 'm',
     'loves': ['apple', 'watermelon'],
     'name': 'Pilot',
     'vaccinated': True,
     'vampires': 54.0,
     'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960f'),
     'dob': datetime.datetime(1999, 12, 20, 21, 15),
     'gender': 'f',
     'loves': ['grape', 'carrot'],
     'name': 'Nimue',
     'vaccinated': True,
     'weight': 540.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}


    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:2: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      



```python
db.mydb2.update({}, {'$set': {'vaccinated': True}}, multi=True) # now can update the whole documents
for i in db.mydb2.find({'vaccinated': True}):
    pprint(i)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'carrot', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 103.0,
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'dob': datetime.datetime(1985, 7, 4, 6, 1),
     'gender': 'f',
     'loves': ['apple', 'carrot', 'chocolate'],
     'name': 'Solnara',
     'vaccinated': True,
     'vampires': 80.0,
     'weight': 550.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'),
     'dob': datetime.datetime(1998, 3, 7, 13, 30),
     'gender': 'f',
     'loves': ['strawberry', 'lemon'],
     'name': 'Ayna',
     'vaccinated': True,
     'vampires': 40.0,
     'weight': 733.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'),
     'dob': datetime.datetime(1997, 7, 1, 14, 42),
     'gender': 'm',
     'loves': ['grape', 'lemon'],
     'name': 'Kenny',
     'vaccinated': True,
     'vampires': 39.0,
     'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'dob': datetime.datetime(2005, 5, 3, 4, 57),
     'gender': 'm',
     'loves': ['apple', 'sugar'],
     'name': 'Raleigh',
     'vaccinated': True,
     'vampires': 2.0,
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960d'),
     'dob': datetime.datetime(2001, 10, 8, 18, 53),
     'gender': 'f',
     'loves': ['apple', 'watermelon'],
     'name': 'Leia',
     'vaccinated': True,
     'vampires': 33.0,
     'weight': 601.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'),
     'dob': datetime.datetime(1997, 3, 1, 10, 3),
     'gender': 'm',
     'loves': ['apple', 'watermelon'],
     'name': 'Pilot',
     'vaccinated': True,
     'vampires': 54.0,
     'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960f'),
     'dob': datetime.datetime(1999, 12, 20, 21, 15),
     'gender': 'f',
     'loves': ['grape', 'carrot'],
     'name': 'Nimue',
     'vaccinated': True,
     'weight': 540.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}


    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.



```python
db.mydb2.update({}, {'$set': {'vaccinated': True}}, multi=True);
for i in db.mydb2.find({'vaccinated': True}).limit(10):
    pprint(i)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'carrot', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 103.0,
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'dob': datetime.datetime(1985, 7, 4, 6, 1),
     'gender': 'f',
     'loves': ['apple', 'carrot', 'chocolate'],
     'name': 'Solnara',
     'vaccinated': True,
     'vampires': 80.0,
     'weight': 550.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'),
     'dob': datetime.datetime(1998, 3, 7, 13, 30),
     'gender': 'f',
     'loves': ['strawberry', 'lemon'],
     'name': 'Ayna',
     'vaccinated': True,
     'vampires': 40.0,
     'weight': 733.0}


    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:1: DeprecationWarning: update is deprecated. Use replace_one, update_one or update_many instead.
      """Entry point for launching an IPython kernel.



```python
for doc in db.mydb2.find({}, {'_id': 0, 'name': 1}): # 0:exclude, 1:include _id: default included
    pprint (doc)

# Sorting documents
for doc in db.mydb2\
            .find({}, {'name': 1, 'weight': 1})\
            .sort([('name', pymongo.ASCENDING), ('vampires', pymongo.DESCENDING)]):
    pprint(doc)
```

    {'name': 'Horny'}
    {'name': 'Aurora'}
    {'name': 'Unicrom'}
    {'name': 'Horny'}
    {'name': 'Horny'}
    {'name': 'Aurora'}
    {'name': 'Unicrom'}
    {'name': 'Roooooodles'}
    {'name': 'Solnara'}
    {'name': 'Ayna'}
    {'name': 'Kenny'}
    {'name': 'Raleigh'}
    {'name': 'Leia'}
    {'name': 'Pilot'}
    {'name': 'Nimue'}
    {'name': 'Dunx'}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'), 'name': 'Aurora', 'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'), 'name': 'Aurora', 'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'), 'name': 'Ayna', 'weight': 733.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'), 'name': 'Dunx', 'weight': 704.0}
    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'), 'name': 'Horny', 'weight': 600.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'), 'name': 'Horny', 'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'), 'name': 'Horny', 'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'), 'name': 'Kenny', 'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960d'), 'name': 'Leia', 'weight': 601.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960f'), 'name': 'Nimue', 'weight': 540.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'), 'name': 'Pilot', 'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'name': 'Raleigh',
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'name': 'Roooooodles',
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'name': 'Solnara',
     'weight': 550.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'name': 'Unicrom',
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'name': 'Unicrom',
     'weight': 984.0}



```python
# Paging results
for doc in db.mydb2.find({}).sort([('name', pymongo.ASCENDING)]).skip(3).limit(2):
    pprint(doc)
```

    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}
    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}



```python
# Creating and deleting an index on name field
# db.mydb2.ensure_index([('name', pymongo.DESCENDING)],unique=True)
# db.mydb2.drop_index([('name', pymongo.DESCENDING)])
db.unicorns.create_index([('name', pymongo.DESCENDING), ('vampires', pymongo.ASCENDING)])
for i in db.mydb2.find():
    pprint(i)
```

    {'_id': ObjectId('5a8b2beb11b3f0fe16ce5f58'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2c0511b3f0fe16ce5f59'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2c2811b3f0fe16ce5f5a'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2ddf2662b8ba6ebba3ea'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959605'),
     'dob': datetime.datetime(1992, 3, 13, 12, 47),
     'gender': 'm',
     'loves': ['carrot', 'papaya'],
     'name': 'Horny',
     'vaccinated': True,
     'vampires': 63.0,
     'weight': 600.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959606'),
     'dob': datetime.datetime(1991, 1, 24, 18, 0),
     'gender': 'f',
     'loves': ['carrot', 'grape'],
     'name': 'Aurora',
     'vaccinated': True,
     'vampires': 43.0,
     'weight': 450.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959607'),
     'dob': datetime.datetime(1973, 2, 10, 3, 10),
     'gender': 'm',
     'loves': ['energon', 'redbull'],
     'name': 'Unicrom',
     'vaccinated': True,
     'vampires': 182.0,
     'weight': 984.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959608'),
     'dob': datetime.datetime(1979, 7, 18, 18, 44),
     'eggs': 1,
     'gender': 'm',
     'loves': ['apple', 'carrot', 'apple', 'durian'],
     'name': 'Roooooodles',
     'vaccinated': True,
     'vampires': 103.0,
     'weight': 590.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959609'),
     'dob': datetime.datetime(1985, 7, 4, 6, 1),
     'gender': 'f',
     'loves': ['apple', 'carrot', 'chocolate'],
     'name': 'Solnara',
     'vaccinated': True,
     'vampires': 80.0,
     'weight': 550.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960a'),
     'dob': datetime.datetime(1998, 3, 7, 13, 30),
     'gender': 'f',
     'loves': ['strawberry', 'lemon'],
     'name': 'Ayna',
     'vaccinated': True,
     'vampires': 40.0,
     'weight': 733.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960b'),
     'dob': datetime.datetime(1997, 7, 1, 14, 42),
     'gender': 'm',
     'loves': ['grape', 'lemon'],
     'name': 'Kenny',
     'vaccinated': True,
     'vampires': 39.0,
     'weight': 690.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960c'),
     'dob': datetime.datetime(2005, 5, 3, 4, 57),
     'gender': 'm',
     'loves': ['apple', 'sugar'],
     'name': 'Raleigh',
     'vaccinated': True,
     'vampires': 2.0,
     'weight': 421.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960d'),
     'dob': datetime.datetime(2001, 10, 8, 18, 53),
     'gender': 'f',
     'loves': ['apple', 'watermelon'],
     'name': 'Leia',
     'vaccinated': True,
     'vampires': 33.0,
     'weight': 601.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960e'),
     'dob': datetime.datetime(1997, 3, 1, 10, 3),
     'gender': 'm',
     'loves': ['apple', 'watermelon'],
     'name': 'Pilot',
     'vaccinated': True,
     'vampires': 54.0,
     'weight': 650.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f95960f'),
     'dob': datetime.datetime(1999, 12, 20, 21, 15),
     'gender': 'f',
     'loves': ['grape', 'carrot'],
     'name': 'Nimue',
     'vaccinated': True,
     'weight': 540.0}
    {'_id': ObjectId('5a8b2e6c5f41dbf57f959610'),
     'dob': datetime.datetime(1976, 7, 18, 22, 18),
     'gender': 'm',
     'loves': ['grape', 'watermelon'],
     'name': 'Dunx',
     'vaccinated': True,
     'vampires': 165.0,
     'weight': 704.0}



```python
db.create_collection('employees')
```




    Collection(Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True), 'mydb2'), 'employees')




```python
from bson.objectid import ObjectId
db.employees.insert({'_id': ObjectId('4d85c7039ab0fd70a117d730'), 'name': 'Leto'})
db.employees.insert({'_id': ObjectId('4d85c7039ab0fd70a117d731'), 'name': 'Duncan',
                    'manager': ObjectId('4d85c7039ab0fd70a117d730')});
db.employees.insert({'_id': ObjectId('4d85c7039ab0fd70a117d732'), 'name': 'Moneo',
                    'manager': ObjectId('4d85c7039ab0fd70a117d730')});

```


```python
for i in db.employees.find():
    pprint(i)
```

    {'_id': ObjectId('4d85c7039ab0fd70a117d730'), 'name': 'Leto'}
    {'_id': ObjectId('4d85c7039ab0fd70a117d731'),
     'manager': ObjectId('4d85c7039ab0fd70a117d730'),
     'name': 'Duncan'}
    {'_id': ObjectId('4d85c7039ab0fd70a117d732'),
     'manager': ObjectId('4d85c7039ab0fd70a117d730'),
     'name': 'Moneo'}



```python
db.employees.insert({'_id': ObjectId('4d85c7039ab0fd70a117d734'), 
                     'name': 'Ghanima',
                    'family': {'mother': 'Chani',
                               'father': 'Paul',
                               'brother': ObjectId('4d85c7039ab0fd70a117d730')}})
```

    /home/karen/.local/lib/python3.5/site-packages/ipykernel_launcher.py:5: DeprecationWarning: insert is deprecated. Use insert_one or insert_many instead.
      """





    ObjectId('4d85c7039ab0fd70a117d734')




```python
for i in db.employees.find():
    pprint(i)
```

    {'_id': ObjectId('4d85c7039ab0fd70a117d730'), 'name': 'Leto'}
    {'_id': ObjectId('4d85c7039ab0fd70a117d731'),
     'manager': ObjectId('4d85c7039ab0fd70a117d730'),
     'name': 'Duncan'}
    {'_id': ObjectId('4d85c7039ab0fd70a117d732'),
     'manager': ObjectId('4d85c7039ab0fd70a117d730'),
     'name': 'Moneo'}
    {'_id': ObjectId('4d85c7039ab0fd70a117d734'),
     'family': {'brother': ObjectId('4d85c7039ab0fd70a117d730'),
                'father': 'Paul',
                'mother': 'Chani'},
     'name': 'Ghanima'}

