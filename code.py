import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- 1. TẠO DỮ LIỆU VÀ LƯU CSV ---------
np.random.seed(42)
n_samples = 200

hours_studied = np.random.normal(6, 2.5, n_samples).clip(0, 12)
attendance = np.random.normal(75, 15, n_samples).clip(40, 100)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = 0.6 * hours_studied + 0.04 * attendance - 5
prob = sigmoid(z)
admitted = np.random.binomial(1, prob)

df = pd.DataFrame({
    'hours_studied': hours_studied,
    'attendance': attendance,
    'admitted': admitted
})

df.to_csv('university_admission.csv', index=False)
print("✅ Đã tạo file university_admission.csv")

# --------- 2. ĐỌC FILE CSV VÀ XỬ LÝ ---------
df = pd.read_csv('university_admission.csv')
X_raw = df[['hours_studied', 'attendance']].values
y = df['admitted'].values

mu = np.mean(X_raw, axis=0)
sigma = np.std(X_raw, axis=0)

X_norm = (X_raw - mu) / sigma
X = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])  # shape (200, 3)

# --------- 3. HÀM CƠ BẢN ---------
def compute_loss(y, y_pred):
    eps = 1e-9
    return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

def train_logistic(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    losses = []

    for epoch in range(epochs):
        z = X @ weights
        y_pred = sigmoid(z)
        error = y_pred - y
        gradient = X.T @ error / m
        weights -= lr * gradient
        loss = compute_loss(y, y_pred)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return weights, losses

def predict_all(X, weights):
    probs = sigmoid(X @ weights)
    return (probs >= 0.5).astype(int)

# --------- 4. HUẤN LUYỆN ---------
weights, losses = train_logistic(X, y)
y_pred = predict_all(X, weights)
accuracy = np.mean(y_pred == y)
print(f"🎯 Accuracy: {accuracy:.2%}")

# --------- 5. NHẬP ĐIỂM MỚI ---------
new_point_raw = np.array([8.5, 90])  # Nhập tại đây: [giờ học, chuyên cần %]
new_point_std = (new_point_raw - mu) / sigma
new_point = np.insert(new_point_std, 0, 1)  # thêm bias

new_pred_prob = sigmoid(np.dot(weights, new_point))
new_pred_class = int(new_pred_prob >= 0.5)
print(f"📌 Dự đoán cho ({new_point_raw[0]} giờ học, {new_point_raw[1]}% chuyên cần): "
      f"{'ĐẬU' if new_pred_class else 'RỚT'} với xác suất {new_pred_prob:.2%}")

# --------- 6. VẼ ĐỒ THỊ ---------
xx, yy = np.meshgrid(
    np.linspace(X[:, 1].min(), X[:, 1].max(), 100),
    np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
)
grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ weights).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, 25, cmap="RdBu", alpha=0.6)
plt.colorbar(label='Probability')

# Điểm dữ liệu cũ
plt.scatter(X[y == 0, 1], X[y == 0, 2], color='red', label='Trượt')
plt.scatter(X[y == 1, 1], X[y == 1, 2], color='blue', label='Đỗ')

# Vẽ điểm mới (màu vàng)
plt.scatter(new_point[1], new_point[2], color='yellow', edgecolor='black', s=100, label='Điểm mới ')
plt.text(new_point[1] + 0.1, new_point[2],
         f"({new_point_raw[0]:.1f}, {new_point_raw[1]:.1f})",
         color='black', fontsize=10)

# Accuracy bên trái
plt.text(X[:,1].min(), X[:,2].min() - 0.5,
         f"Accuracy: {accuracy:.2%}",
         fontsize=12, color='black', ha='left')

# ĐẬU bên phải
plt.text(X[:,1].max(), X[:,2].min() - 0.5,
         f"ĐẬU với xác suất {new_pred_prob:.2%}",
         fontsize=12, color='black', ha='right')

plt.xlabel("Hours Studied (normalized)")
plt.ylabel("Attendance (normalized)")
plt.title("Logistic Regression Classification Result")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
