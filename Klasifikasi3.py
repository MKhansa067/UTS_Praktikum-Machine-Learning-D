import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#%%
data = pd.read_csv('dataset_buys_comp.csv')

#%%
# info dataset
print(data.head(10))  # Menampilkan 10 data teratas
print("\nInformasi dataset:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe(include='all').T)

#%%
# visualisasi data
plt.figure(figsize=(12, 6))
sns.countplot(x='Buys_Computer', data=data)
plt.title('Distribusi Kelas Target (Buys_Computer)')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Age', hue='Buys_Computer', data=data)
plt.title('Distribusi Pembelian Komputer berdasarkan Usia')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Income', hue='Buys_Computer', data=data)
plt.title('Distribusi Pembelian Komputer berdasarkan Pendapatan')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Student', hue='Buys_Computer', data=data)
plt.title('Distribusi Pembelian Komputer berdasarkan Status Mahasiswa')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Credit_Rating', hue='Buys_Computer', data=data)
plt.title('Distribusi Pembelian Komputer berdasarkan Rating Kredit')
plt.show()

#%%
# encoding variabel kategorikal
categorical_cols = ['Age', 'Income', 'Student', 'Credit_Rating']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le  # Simpan encoder per kolom

#%%
# memisahkan fitur dan target
X = data.drop('Buys_Computer', axis=1)
y = data['Buys_Computer']

#%%
# membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

#%%
# membuat dan melatih model Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

#%%
# memprediksi data testing
y_pred = model.predict(X_test)
print("\nData testing:")
print(X_test.head())

#%%
# evaluasi model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,7))
sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16}, fmt='g')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#%%
# visualisasi Decision Tree
features = X.columns
fig, ax = plt.subplots(figsize=(25,20))
tree.plot_tree(model, feature_names=features, class_names=['Tidak', 'Ya'], filled=True)
plt.show()

#%%
# contoh prediksi data baru
new_data = {
    'Age': 'Paruh Baya',
    'Income': 'Sedang',
    'Student': 'Tidak',
    'Credit_Rating': 'Baik'
}

#%%
# mengubah data baru menjadi DataFrame dan melakukan encoding dengan encoder yang sudah disimpan
new_data_df = pd.DataFrame([new_data])
for col in categorical_cols:
    new_data_df[col] = encoders[col].transform(new_data_df[col])

#%%
# memprediksi
prediction = model.predict(new_data_df)
print("\nPrediksi untuk data baru:", "Ya" if prediction[0] == 1 else "Tidak")
