# UTS_Praktikum-Machine-Learning-D
Nama  : Muhamad Khansa Khalifaturohman  
NIM   : 1247050115  
Kelas : D  

# Klasifikasi Kelayakan Beli Komputer  
Data yang digunakan dalam membuat model dengan Decision Tree disini yakni data yang diambil dari data Dataset Buys Comp, selengkapnya informasi data dapat dilihat pada file dataset_buys_comp.csv.  
Dataset ini berisi sejumlah data dengan atribut:
 - Age (Usia)
 - Income (Pendapatan)
 - Student (Mahasiswa)
 - Credit_Rating (Rating Kredit)
 - Buys_Computer (Label/Target)
Adapun cara untuk dapat melihat dataset tersebut dengan menggunakan langkah-langkah berikut ini:
1. Buatlah satu buah project baru dan lakukan load library berikut ini:
#%%  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import tree  
from sklearn.preprocessing import LabelEncoder

3. Kemudian lakukan load dan liat data:
#%%  
data = pd.read_csv('dataset_buys_comp.csv')
#%%  
#info dataset  
print(data.head(10))  #Menampilkan 10 data teratas  
print("\nInformasi dataset:")  
print(data.info())  
print("\nStatistik deskriptif:")  
print(data.describe(include='all').T)  
