import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title("KNN Weather Classification")

X = np.array([
    [50, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

y = np.array([0, 1, 0, 0, 1, 1])

label_map = {0: "Sunny", 1: "Rainy"}

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

st.sidebar.header("Enter New Day Data")

temp = st.sidebar.slider("Temperature", 0, 60, 26)
hum = st.sidebar.slider("Humidity", 0, 100, 78)

new_data = np.array([[temp, hum]])
prediction = knn.predict(new_data)[0]

st.subheader("Prediction Result:")
st.success(f"Predicted Weather: {label_map[prediction]}")

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(X[y==0, 0], X[y==0, 1],
           color='orange', label='Sunny', s=100, edgecolor='k')

ax.scatter(X[y==1, 0], X[y==1, 1],
           color='blue', label='Rainy', s=100, edgecolor='k')

ax.scatter(temp, hum,
           color='red' if prediction == 1 else 'orange',
           marker='*',
           s=300,
           edgecolor='black',
           label=f'New Day: {label_map[prediction]}')

ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(True)

st.pyplot(fig)
