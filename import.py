import pandas as pd
raisin=pd.read_csv('https://raw.githubusercontent.com/EdoHanifauzan/data/Dataset/Raisin_Dataset.csv')
# Ordinal feature encoding
# (https://www.kaggle.com/datasets/shrutisaxena/raisin-dataset ")
df = raisin.copy()
target = 'Class'

target_mapper = {'Kecimen':0, 'Besni':1}
def target_encode(val):
    return target_mapper[val]

df['Class'] = df['Class'].apply(target_encode)

# Separating X and y
X = df.drop('Class', axis=1)
Y = df['Class']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('raisin_cls.pkl', 'wb'))