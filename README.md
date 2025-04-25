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
Karena data berupa teks dan bukan berupa numerik, kita bisa mengubahnya menjadi variabel kategorikal yang dapat terbaca oleh model dengan menggunakan LabelEncoder. Berikut adalah kodenya:   

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

# 7. Model Klasifikasi (Decision Tree)  
Model Decision Tree digunakan sebagai model klasifikasi, dibangun menggunakan kriteria entropy. Untuk langkahnya adalah sebagai berikut:  

a. Membuat model Decision Tree  
#%%  
#membuat dan melatih model Decision Tree  
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)  
model.fit(X_train, y_train)  

b. Model untuk Prediksi  
#%%  
#memprediksi data testing  
y_pred = model.predict(X_test)  
print("\nData testing:")  
print(X_test.head())  

# 8. Classification Report  
Digunakan untuk menampilkan nilai akurasi, precision, recall, dan F1-score. Berikut adalah kodenya.  

#%%  
#evaluasi model  
print("\nClassification Report:")  
print(classification_report(y_test, y_pred))  

# 9. Confusion Matrix  
Digunakan untuk melihat hasil perbandingan prediksi benar dan salah. Berikut adalah kodenya.  

#%%  
#Confusion Matrix  
cm = confusion_matrix(y_test, y_pred)  
fig, ax = plt.subplots(figsize=(7,7))  
sns.set(font_scale=1.4)  
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16}, fmt='g')  
plt.xlabel('Predictions', fontsize=18)  
plt.ylabel('Actuals', fontsize=18)  
plt.title('Confusion Matrix', fontsize=18)  
plt.show()  

# 10. Visualisasi Decision Tree  
Berikut adalah kode untuk menampilkan model Decision Tree yang sebelumnya telah dibuat.  

#%%  
#visualisasi Decision Tree  
features = X.columns  
fig, ax = plt.subplots(figsize=(25,20))  
tree.plot_tree(model, feature_names=features, class_names=['Tidak', 'Ya'], filled=True)  
plt.show()  

# 11. Uji Data Baru  
Setelah model dibuat, bisa dengan menambahkan kode berikut sebagai data baru yang akan di tes. Agar data baru tersimpan dan terbaca oleh model, sama seperti langkah encoding sebelumnya, bahwa untuk data baru juga memerlukan LabelEncoder, bisa dibilang dengan cara mengubah data baru menjadi DataFrame dan melakukan encoding dengan encoder yang sudah disimpan. Berikut adalah langkah-langkahnya.  

a. Data baru  
#%%
new_data = {
    'Age': 'Paruh Baya',
    'Income': 'Sedang',
    'Student': 'Tidak',
    'Credit_Rating': 'Baik'
}

b. Encoding Data Baru  
#%%
#ubah data baru ke DataFrame
new_data_df = pd.DataFrame([new_data])
for col in categorical_cols:
    new_data_df[col] = encoders[col].transform(new_data_df[col])

c. Prediksi Data Baru  
#%%
#memprediksi
prediction = model.predict(new_data_df)  
print("\nPrediksi untuk data baru:", "Ya" if prediction[0] == 1 else "Tidak")  
