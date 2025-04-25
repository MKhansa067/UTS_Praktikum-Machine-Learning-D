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

# 1. Library yang digunakan  
Sebelum dimulai, adapun Library yang akan digunakan/di install dalam pembuatan model klasiikasi ini, yang diantaranya:  
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Buatlah satu buah project baru dan lakukan load library seperti berikut ini:  

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

# 2. Load Data  
Kemudian lakukan load dan lihat data yang sesuai dengan dataset yang ada, seperti berikut ini:  

#%%  
data = pd.read_csv('dataset_buys_comp.csv')
#%%  
#info dataset  
print(data.head(10))  #Menampilkan 10 data teratas  
print("\nInformasi dataset:")  
print(data.info())  
print("\nStatistik deskriptif:")  
print(data.describe(include='all').T)  

# 3. Visualisasi Data  
Selanjutnya lakukan visualisasi data dari tiap atribut seperti berikut ini:  

a. Atribut Buys_Computer sebagai kelas target  
#%%  
#visualisasi data  
plt.figure(figsize=(12, 6))  
sns.countplot(x='Buys_Computer', data=data)  
plt.title('Distribusi Kelas Target (Buys_Computer)')  
plt.show() 

b. Atribut Age  
#Age  
plt.figure(figsize=(12, 6))  
sns.countplot(x='Age', hue='Buys_Computer', data=data)  
plt.title('Distribusi Pembelian Komputer berdasarkan Usia')  
plt.show()  

c. Atribut Income  
#Income  
plt.figure(figsize=(12, 6))  
sns.countplot(x='Income', hue='Buys_Computer', data=data)  
plt.title('Distribusi Pembelian Komputer berdasarkan Pendapatan')  
plt.show()  

d. Atribut Student  
#Student  
plt.figure(figsize=(12, 6))  
sns.countplot(x='Student', hue='Buys_Computer', data=data)  
plt.title('Distribusi Pembelian Komputer berdasarkan Status Mahasiswa')  
plt.show()  

e. Atribut Credit_Rating  
#Credit_Rating  
plt.figure(figsize=(12, 6))  
sns.countplot(x='Credit_Rating', hue='Buys_Computer', data=data)  
plt.title('Distribusi Pembelian Komputer berdasarkan Rating Kredit')  
plt.show()  

Dari ke-5 atribut tersebut, terdapat label berwarna dengan penjelasan sebagai berikut:  
- Warna biru atau 0 artinya YA.  
- Warna oranye atau 1 artinya TIDAK.  

# 5. Preprocessing: Encoding Variabel Kategorikal
Karena data berupa teks dan bukan berupa numerik, kita bisa mengubahnya menjadi variabel kategorikal dengan menggunakan LabelEncoder seperti pada kode berikut:  

#%%  
#encoding variabel kategorikal  
categorical_cols = ['Age', 'Income', 'Student', 'Credit_Rating']  
encoders = {}  
for col in categorical_cols:  
    le = LabelEncoder()  
    data[col] = le.fit_transform(data[col])  
    encoders[col] = le  # Simpan encoder per kolom  

# 6. Split Data (Training dan Testing)  
Kita dapat melakukan pembagian data dengan tujuan menggunakan data training yang dipergunakan untuk membuat model. Pembagian data yang digunakan yaitu 70% untuk data training dan 30% untuk data testing. Berikut adalah langkahnya:  

a. Memisahkan fitur dan target  
#%%  
#memisahkan fitur dan target
X = data.drop('Buys_Computer', axis=1)
y = data['Buys_Computer']

b. Membagi data menjadi training dan testing test  
#%%
#membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))
