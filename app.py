import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

st.title("""Edo Hanifauzan Satria""")
st.title("""200411100058""")
st.header("""Penambangan Data 	""")

model = pickle.load(open('modelNBC_Raisin.pkl', 'rb'))
df=pd.read_csv('https://raw.githubusercontent.com/EdoHanifauzan/data/Dataset/Raisin_Dataset.csv')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
y=df["Class"]
X=df.drop(columns =["Class"])
normalized = min_max_scaler.fit_transform(X)
n=pd.DataFrame(normalized,columns=['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter'])


page1, page2, page3 = st.tabs(["Home", "Data", "Input Model"])

with page1:
    st.header("Menghitung Klasifikasi Kismis")
    st.write("Dataset Yang digunakan adalah dari [Kaggle](https://www.kaggle.com/datasets/shrutisaxena/raisin-dataset ")
    st.write("Link repository Github : [https://github.com/EdoHanifauzan/data/blob/Dataset/Raisin_Dataset.csv](https://github.com/EdoHanifauzan/data/blob/Dataset/Raisin_Dataset.csv)")
    st.subheader("Deskripsi Data")
    
    st.markdown("""
    <ul>
        <p>Fitur</p>
        <li>
            Kolom 1: Area
            <p>Dikolom Area ini Memberikan jumlah piksel dalam batas kismis</p>
        </li>
        <li>
            Kolom 2: MajorAxisLength
            <p>Dikolom MajorAxisLength ini Memberikan panjang sumbu utama </p>
        </li>
        <li>
            Kolom 3: MinorAxisLength
            <p> Dikolom MinorAxisLength ini memberikan panjang sumbu kecil, yang merupakan garis terpendek yang dapat ditarik pada kismis </p>
        </li>
        <li>
            Kolom 4: Eksentrisitas
            <p> Dikolom Eksentrisitas Ini memberikan ukuran eksentrisitas elips, yang memiliki momen yang sama dengan kismis </p>
        </li>
        <li>
            Kolom 5: ConvexArea
            <p> Dikolom ConvexArea ini memberikan jumlah piksel dari cangkang cembung terkecil dari wilayah yang dibentuk oleh kismis</p>
        </li>
        <li>
            Kolom 6: Extent
            <p> Dikolom Extent ini  memberikan rasio wilayah yang dibentuk oleh kismis terhadap total piksel dalam kotak pembatas </p>
        </li>
        <li>
            Kolom 7: Perimeter
            <p> Dikolom Perimeter inimengukur lingkungan dengan menghitung jarak antara batas kismis dan piksel di sekitarnya </p>
        </li>
        <p>Kategori</p>
        <li>
            Kolom 8: Class
            <p> Dikolom Class ini menjelaskan pembagian kelas kismis Kecimen dan Besni: </p>
        </li>
       
    </ul>
""", unsafe_allow_html=True)

with page2:
	st.header(" Raisin Dataset ")
	st.write(df)
	st.subheader("Data proccesing")
	st.write(n)
	st.write('Jumlah Baris dan Kolom :',X.shape)
	st.write('Jumlah kelas :',len(np.unique(y)))
	X_train, X_test, y_train, y_test=train_test_split(normalized, y, test_size= 0.20, random_state=4321)
	alg=KNeighborsClassifier(n_neighbors=8)
	alg.fit(X_train,y_train)
	y_pred=alg.predict(X_test)
	acc=accuracy_score(y_test, y_pred)
	st.write(f'Akurasi  =', acc)

with page3:
    st.header("Input Data Model")

    
    # membuat input
    def user_input_features():
        Area = st.slider('Area',40000,90000,63000)
        MajorAxisLength = st.slider('Major Axis Length',220.1,500.0,403.9)
        MinorAxisLength = st.slider('Minor Axis Length',150.1,300.9,189.9)
        Eksentrisitas = st.slider('Eksentrisitas',0.10,0.99,0.56)
        ConvexArea = st.slider('Convex Area',4000,9999,4350)
        Extent = st.slider('Extent',0.100,0.999,0.39)
        Perimeter = st.slider('Perimeter',800.0,1300.9,843.9)
        data={'Area':Area,
            'Major Axis Length' : MajorAxisLength,
            'Minor Axis Length' : MinorAxisLength,
            'Eksentrisitas' : Eksentrisitas,
            'Convex Area' : ConvexArea,
            'Extent' : Extent,
            'Perimeter' : Perimeter
        }
        features = pd.DataFrame(data, index=[0])
        return features
        
    data = user_input_features()
    algoritma=st.selectbox('pilih algoritma',("GaussianNB","KNN","K-Means","Decision Tree"))
    if algoritma=='KNN':
        algo=KNeighborsClassifier(n_neighbors=8)
    elif algoritma=='GaussianNB':
        algo=GaussianNB()
    elif algoritma =='K-Means':
        algo=KMeans(n_clusters=5)
    elif algoritma =='Decision Tree':
        algo= DecisionTreeClassifier(criterion="gini")

    st.subheader("Hasil :")
    st.write(data)
    algo.fit(X,y)
    prediction = model.predict(data)

    st.subheader('Prediksi')
    st.write(data[prediction])
