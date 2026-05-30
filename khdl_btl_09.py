# %%
import io
import pandas as pd
from google.colab import files

print("Hãy bấm vào nút 'Choose Files' (hoặc 'Chọn tệp') phía dưới để chọn tệp từ máy tính:")
# Đoạn code này sẽ ép Colab hiện một nút chọn tệp ngay tại ô kết quả
uploaded = files.upload()

# Tự động tìm tên tệp bạn vừa tải lên để đọc dữ liệu
for file_name in uploaded.keys():
    df = pd.read_excel(io.BytesIO(uploaded[file_name]), header=1)
    print("\n--- CHÚC MỪNG! ĐÃ ĐỌC DỮ LIỆU THÀNH CÔNG ---")
    display(df.head())

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Đổi tên cột mục tiêu cuối cùng thành 'TARGET' cho dễ viết code
df.rename(columns={df.columns[-1]: 'TARGET'}, inplace=True)

# 2. Vẽ biểu đồ xem tỷ lệ nợ xấu
plt.figure(figsize=(6, 4))
sns.countplot(x='TARGET', data=df, palette='Set2')
plt.title('Tỷ lệ Khách hàng Tốt (0) vs Nợ xấu (1)')
plt.xlabel('Trạng thái (0: Trả đúng hạn, 1: Nợ xấu)')
plt.ylabel('Số lượng khách hàng')
plt.show()

# 3. Tách biến tính năng (X) và biến mục tiêu (y)
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# 4. Chia dữ liệu thành 2 tập: Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Chuẩn hóa dữ liệu để các cột tiền lớn và tuổi tác có cùng thang đo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- ĐÃ HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU ---")
print(f"Số mẫu tập Học (Train): {X_train_scaled.shape[0]} dòng")
print(f"Số mẫu tập Kiểm tra (Test): {X_test_scaled.shape[0]} dòng")

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Khởi tạo và huấn luyện mô hình Hồi quy Logistic
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 2. Dự đoán trên tập dữ liệu kiểm tra (Test)
y_pred = model.predict(X_test_scaled)

# 3. In kết quả để đưa vào bảng báo cáo
print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")

print("--- BÁO CÁO CHI TIẾT (Classification Report) ---")
print(classification_report(y_test, y_pred))

print("--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---")
print(confusion_matrix(y_test, y_pred))


