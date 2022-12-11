import streamlit as st
import pandas as pd 
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
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

#st.subheader('User Input Features')
df=pd.read_csv('https://raw.githubusercontent.com/EdoHanifauzan/data/Dataset/Raisin_Dataset.csv')
scaler=MinMaxScaler()
d=df.drop(columns =["Class"])
scala= scaler.fit(d)
normalized=scaler.transform(d)
y=df["Class"]
n=pd.DataFrame(normalized,columns=['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter'])

def tambah_parameter(nama_algoritma):
	params=dict()
	if nama_algoritma =='KNN':
		params['K']=8
	elif nama_algoritma =='Decision Tree':
		params=''
	elif nama_algoritma =='GaussianNB':
		params=''
	elif nama_algoritma =='K-Means':
		params['n_clusters']=6
	elif nama_algoritma =='Random Forest':
		params['max_depth']=4
		params['n_estimators']=6
	return params

def pilih_klasifikasi(nama_algoritma, params):
	algo=None
	if nama_algoritma=='KNN':
		algo=KNeighborsClassifier(n_neighbors=params['K'])
	elif nama_algoritma=='GaussianNB':
		algo=GaussianNB()
	elif nama_algoritma =='K-Means':
		algo=KMeans(n_clusters=params['n_clusters'])
	elif nama_algoritma =='Decision Tree':
		algo= DecisionTreeClassifier(criterion="gini")
	elif nama_algoritma =='Random Forest':
		algo=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'], random_state=4321)
	return algo

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
	st.write('Jumlah Baris dan Kolom :',d.shape)
	st.write('Jumlah kelas :',len(np.unique(y)))
	X_train, X_test, y_train, y_test=train_test_split(normalized, y, test_size= 0.20, random_state=4321)
	algo=KNeighborsClassifier(n_neighbors=8)
	algo.fit(X_train,y_train)
	y_pred=algo.predict(X_test)
	acc=accuracy_score(y_test, y_pred)
	st.write(f'Akurasi  =', acc)

with page3:
    st.header("Input Data Model")

    # membuat input
    Area = st.number_input('Area')
    MajorAxisLength = st.number_input('Major Axis Length')
    MinorAxisLength = st.number_input('Minor Axis Length')
    Eksentrisitas = st.number_input('Eksentrisitas')
    ConvexArea = st.number_input('Convex Area')
    Extent = st.number_input('Extent')
    Perimeter = st.number_input('Perimeter')
    algoritma=st.selectbox('pilih algoritma',("GaussianNB","KNN","K-Means","Decision Tree","Random Forest"))
    alg=pilih_klasifikasi(algoritma, params)
DataInput = np.array([[Area, MajorAxisLength, MinorAxisLength,Eksentrisitas,ConvexArea,Extent,Perimeter]])
inputan=pd.DataFrame(data,index=[0])
st.write(inputan)
normal=scaler.transform(inputan)
y=df["Class"]
alg.fit(normal,y)
prediksi=algo.predict(normal)
prediksi_proba=alg.predict_proba(normal)
st.write(y[prediksi])
st.write(y[prediksi_proba])
