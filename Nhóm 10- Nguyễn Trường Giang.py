import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

def install_required_packages():
    try:
        import sklearn
    except ModuleNotFoundError:
        os.system(f"{sys.executable} -m pip install scikit-learn")
install_required_packages()

# Đọc dữ liệu
df = pd.read_csv(r"c:\\Users\\Windows\\Downloads\\CO2 Emissions_Canada.csv")

# Chọn các cột quan trọng
columns_to_use = [
    "Engine Size(L)", "Cylinders", "Transmission", "Fuel Type",
    "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"
]
df = df[columns_to_use]

# Mã hóa biến phân loại
label_encoders = {}
for col in ["Transmission", "Fuel Type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Tách biến đầu vào (X) và biến mục tiêu (y)
X = df.drop(columns=["CO2 Emissions(g/km)"])
y = df["CO2 Emissions(g/km)"]

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', 
                     learning_rate_init=0.001, max_iter=500, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} g/km")

# Vẽ biểu đồ kết quả
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label='Actual CO2 Emissions', marker='o')
plt.plot(y_pred[:50], label='Predicted CO2 Emissions', marker='s')
plt.xlabel("Sample Index")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()
