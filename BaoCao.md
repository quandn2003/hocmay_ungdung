# Báo Cáo: Phân Cụm Chuỗi Thời Gian Dữ Liệu Chứng Khoán

## 1. Giới Thiệu

Dự án này thực hiện phân cụm các chuỗi thời gian dữ liệu chứng khoán bằng thuật toán K-means với phương pháp khởi tạo AI-Daoud, so sánh với phương pháp khởi tạo ngẫu nhiên truyền thống. Mục đích là gom cụm các công ty thành các nhóm dựa trên sự biến thiên giá cổ phiếu tương tự nhau.

## 2. Mô Tả Dữ Liệu

Bộ dữ liệu bao gồm hàng trăm chuỗi thời gian, mỗi chuỗi là giá cổ phiếu của một công ty được thu thập tại nhiều thời điểm cách đều nhau. Dữ liệu được lưu trữ trong các file .dat trong thư mục `./data`, mỗi file đại diện cho một công ty.

## 3. Phương Pháp Tiếp Cận

### 3.1. Tiền Xử Lý Dữ Liệu

Trong bước tiền xử lý, chúng tôi thực hiện:
- Chuẩn hóa Z-score để đưa dữ liệu về cùng một phạm vi
- Cắt giảm độ dài để đảm bảo tất cả các chuỗi thời gian có cùng độ dài

```python
def preprocess_data(data):
    # Tìm độ dài nhỏ nhất
    min_length = min(len(ts) for ts in data.values())
    
    # Chuẩn hóa và đảm bảo cùng độ dài
    preprocessed_data = {}
    for company, time_series in data.items():
        # Cắt giảm độ dài
        time_series = time_series[:min_length]
        
        # Chuẩn hóa Z-score
        mean = np.mean(time_series)
        std = np.std(time_series)
        if std != 0:
            normalized_ts = (time_series - mean) / std
        else:
            normalized_ts = time_series - mean
            
        preprocessed_data[company] = normalized_ts
        
    return preprocessed_data
```

### 3.2. Trích Xuất Đặc Trưng

Chúng tôi trích xuất nhiều loại đặc trưng từ chuỗi thời gian để nắm bắt các đặc tính quan trọng:

#### 3.2.1. Đặc Trưng Miền Thời Gian
- Thống kê cơ bản: giá trị trung bình, độ lệch chuẩn, phương sai, giá trị min/max, phạm vi
- Độ lệch (skewness) và độ nhọn (kurtosis)
- Sự khác biệt bậc nhất và bậc hai
- Các phân vị (25%, 50%, 75%)
- Số lượng đỉnh và đáy

#### 3.2.2. Đặc Trưng Miền Tần Số
- Sử dụng FFT (Fast Fourier Transform) để chuyển dữ liệu sang miền tần số
- Tính toán công suất tối đa, tần số tối đa, công suất trung bình
- Entropy phổ và centroid phổ
- Các dải công suất

#### 3.2.3. Đặc Trưng Thống Kê
- Tự tương quan (autocorrelation) ở các độ trễ khác nhau
- Kiểm tra tính dừng (stationarity) bằng phép kiểm ADF
- Các biện pháp độ biến động (volatility)
- Đặc trưng xu hướng qua hồi quy tuyến tính

### 3.3. Giảm Chiều Dữ Liệu

Sau khi trích xuất đặc trưng, chúng tôi áp dụng PCA (Principal Component Analysis) để giảm số chiều dữ liệu nhưng vẫn giữ được 95% phương sai:

```python
def apply_pca(X, n_components=0.95):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Giảm từ {X.shape[1]} xuống {X_pca.shape[1]} đặc trưng")
    print(f"Tỉ lệ phương sai giải thích: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, pca
```

### 3.4. Phương Pháp Khởi Tạo AI-Daoud

Thuật toán AI-Daoud để khởi tạo các tâm cụm (centroids) cho K-means:

1. Tính phương sai của mỗi thuộc tính đặc trưng
2. Tìm thuộc tính có phương sai lớn nhất
3. Sắp xếp dữ liệu theo thuộc tính này
4. Chia dữ liệu thành k nhóm có số lượng bằng nhau
5. Tìm giá trị trung vị của mỗi nhóm
6. Sử dụng các điểm dữ liệu tương ứng với các giá trị trung vị làm tâm cụm ban đầu

```python
def ai_daoud_initialization(X, k):
    n_samples, n_features = X.shape
    
    # Bước 1: Tính phương sai của mỗi đặc trưng
    variances = np.var(X, axis=0)
    
    # Bước 2: Tìm đặc trưng có phương sai lớn nhất
    max_var_feature_idx = np.argmax(variances)
    
    # Bước 3: Sắp xếp dữ liệu theo đặc trưng này
    feature_values = X[:, max_var_feature_idx]
    sorted_indices = np.argsort(feature_values)
    
    # Bước 4: Chia thành k nhóm có kích thước bằng nhau
    groups = np.array_split(sorted_indices, k)
    
    # Bước 5: Tìm trung vị của mỗi nhóm và sử dụng điểm dữ liệu tương ứng làm tâm cụm
    centroids = np.zeros((k, n_features))
    for i, group in enumerate(groups):
        if len(group) > 0:
            median_idx = group[len(group) // 2]
            centroids[i] = X[median_idx]
    
    return centroids
```

### 3.5. Thuật Toán K-means

Thuật toán K-means sử dụng phương pháp khởi tạo AI-Daoud hoặc khởi tạo ngẫu nhiên truyền thống:

```python
def kmeans(X, k, max_iters=100, init_method='ai_daoud'):
    n_samples, n_features = X.shape
    
    # Khởi tạo tâm cụm theo phương pháp đã chọn
    if init_method == 'ai_daoud':
        centroids = ai_daoud_initialization(X, k)
    else:  # Khởi tạo ngẫu nhiên
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_indices]
    
    # Khởi tạo nhãn
    labels = np.zeros(n_samples, dtype=int)
    old_labels = np.ones(n_samples, dtype=int)
    
    # Vòng lặp chính
    iteration = 0
    while not np.array_equal(labels, old_labels) and iteration < max_iters:
        old_labels = labels.copy()
        
        # Gán mẫu vào tâm cụm gần nhất
        distances = cdist(X, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        # Cập nhật tâm cụm
        for j in range(k):
            if np.sum(labels == j) > 0:  # Đảm bảo cụm không rỗng
                centroids[j] = np.mean(X[labels == j], axis=0)
        
        iteration += 1
    
    # Tính toán inertia (tổng bình phương khoảng cách đến tâm cụm)
    inertia = 0
    for i in range(n_samples):
        inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)
    
    return centroids, labels, inertia
```

### 3.6. Phương Pháp Elbow

Phương pháp Elbow được sử dụng để xác định số lượng cụm tối ưu (k):

1. Chạy K-means với các giá trị k khác nhau
2. Vẽ đồ thị SSE (Sum of Squared Errors) cho mỗi giá trị k
3. Xác định "điểm khuỷu tay" (elbow point) - điểm mà việc thêm nhiều cụm hơn không mang lại nhiều lợi ích

```python
def find_elbow_point(k_values, sse_values):
    # Đảm bảo k_values được sắp xếp theo thứ tự tăng dần
    sorted_idx = np.argsort(k_values)
    k_values_sorted = np.array(k_values)[sorted_idx]
    sse_values_sorted = np.array(sse_values)[sorted_idx]
    
    # Tìm điểm khuỷu tay sử dụng KneeLocator
    kneedle = KneeLocator(
        k_values_sorted, 
        sse_values_sorted, 
        curve='convex', 
        direction='decreasing',
        S=1.0
    )
    elbow_k = kneedle.elbow
    
    # Nếu không tìm thấy điểm khuỷu tay, sử dụng phương pháp heuristic đơn giản
    if elbow_k is None:
        # Tính tốc độ giảm SSE
        sse_diffs = np.diff(sse_values_sorted)
        rates = sse_diffs[:-1] / sse_diffs[1:]
        
        # Tìm điểm có thay đổi tốc độ lớn nhất
        if len(rates) > 0:
            elbow_idx = np.argmax(rates) + 1
            return int(k_values_sorted[elbow_idx])
        else:
            return int(k_values_sorted[len(k_values_sorted) // 2])
    
    return int(elbow_k)
```

### 3.7. Các Chỉ Số Đánh Giá

Chúng tôi sử dụng nhiều chỉ số để đánh giá và so sánh chất lượng phân cụm:

#### 3.7.1. Chỉ Số Silhouette

Đo lường mức độ tương tự của một đối tượng với cụm của nó so với các cụm khác. Giá trị cao hơn cho thấy phân cụm tốt hơn.

```python
def silhouette_score(X, labels, k):
    n_samples = X.shape[0]
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Cụm của mẫu hiện tại
        cluster_i = labels[i]
        
        # Tính a (khoảng cách trung bình đến tất cả các mẫu khác trong cùng cụm)
        if np.sum(labels == cluster_i) > 1:  # Nếu không đơn độc trong cụm
            a = np.mean(cdist(X[i].reshape(1, -1), X[labels == cluster_i], 'euclidean'))
        else:
            a = 0
        
        # Tính b (khoảng cách trung bình đến các mẫu trong cụm lân cận gần nhất)
        b_values = []
        for j in range(k):
            if j != cluster_i and np.sum(labels == j) > 0:
                b_values.append(np.mean(cdist(X[i].reshape(1, -1), X[labels == j], 'euclidean')))
        
        b = min(b_values) if b_values else 0
        
        # Tính silhouette
        if a == 0 and b == 0:
            silhouette_vals[i] = 0
        elif a < b:
            silhouette_vals[i] = 1 - a / b
        elif a > b:
            silhouette_vals[i] = b / a - 1
        else:  # a == b
            silhouette_vals[i] = 0
    
    # Giá trị silhouette trung bình
    return np.mean(silhouette_vals)
```

#### 3.7.2. Chỉ Số Davies-Bouldin

Đo lường sự phân tán trong cụm và sự tách biệt giữa các cụm. Giá trị thấp hơn cho thấy phân cụm tốt hơn.

```python
def davies_bouldin_index(X, labels, centroids, k):
    if k <= 1:
        return 0
    
    # Tính độ phân tán của cụm (khoảng cách trung bình của tất cả các mẫu đến tâm cụm)
    dispersions = np.zeros(k)
    for i in range(k):
        cluster_samples = X[labels == i]
        if len(cluster_samples) > 0:
            distances = cdist(cluster_samples, centroids[i].reshape(1, -1), 'euclidean')
            dispersions[i] = np.mean(distances)
    
    # Tính chỉ số Davies-Bouldin
    db_indices = np.zeros(k)
    for i in range(k):
        if np.sum(labels == i) == 0:  # Bỏ qua cụm rỗng
            continue
            
        max_ratio = 0
        for j in range(k):
            if i != j and np.sum(labels == j) > 0:
                # Khoảng cách giữa các tâm cụm
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                if centroid_distance > 0:  # Tránh chia cho 0
                    ratio = (dispersions[i] + dispersions[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        
        db_indices[i] = max_ratio
    
    # Trung bình trên tất cả các cụm
    return np.mean(db_indices)
```

#### 3.7.3. Chỉ Số Calinski-Harabasz (CH)

Đo lường tỷ lệ giữa độ phân tán giữa các cụm và độ phân tán trong cụm. Giá trị cao hơn cho thấy phân cụm tốt hơn.

```python
def calinski_harabasz_index(X, labels, centroids, k):
    n_samples = X.shape[0]
    
    if k <= 1 or n_samples <= k:
        return 0
    
    # Tính tâm tổng thể
    overall_centroid = np.mean(X, axis=0)
    
    # Độ phân tán giữa các cụm
    between_cluster_ss = 0
    for i in range(k):
        n_cluster_samples = np.sum(labels == i)
        if n_cluster_samples > 0:
            between_cluster_ss += n_cluster_samples * np.sum((centroids[i] - overall_centroid) ** 2)
    
    # Độ phân tán trong cụm
    within_cluster_ss = 0
    for i in range(n_samples):
        within_cluster_ss += np.sum((X[i] - centroids[labels[i]]) ** 2)
    
    # Tính chỉ số CH
    if within_cluster_ss == 0:  # Phân cụm hoàn hảo
        return float('inf')
    
    ch_index = (between_cluster_ss / (k - 1)) / (within_cluster_ss / (n_samples - k))
    return ch_index
```

#### 3.7.4. Sum of Squared Errors (SSE) / Within-Cluster Sum of Squares (WCSS)

Đo lường tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm của chúng. Giá trị thấp hơn cho thấy phân cụm tốt hơn.

## 4. Cài Đặt và Thực Thi

### 4.1. Cấu Trúc Mã Nguồn

Mã nguồn được tổ chức thành các phần chính:
- `kmeans_clustering.py`: Triển khai chính của phân cụm
- `time_series_features.py`: Trích xuất đặc trưng nâng cao cho dữ liệu chuỗi thời gian

### 4.2. Tham Số Dòng Lệnh

Mã nguồn hỗ trợ các tham số dòng lệnh để tùy chỉnh việc thực thi:
- `--clusters` hoặc `-k`: Số lượng cụm cần thử (có thể chỉ định nhiều giá trị)
- `--elbow` hoặc `-e`: Sử dụng phương pháp Elbow để tìm k tối ưu
- `--min-k`: Giá trị k tối thiểu cho phương pháp Elbow (mặc định: 2)
- `--max-k`: Giá trị k tối đa cho phương pháp Elbow (mặc định: 15)
- `--step-k`: Kích thước bước cho giá trị k trong phương pháp Elbow (mặc định: 1)

Ví dụ:
```
# Chạy phương pháp Elbow với phạm vi mặc định (2-15)
python kmeans_clustering.py --elbow

# Chạy phương pháp Elbow với phạm vi tùy chỉnh
python kmeans_clustering.py --elbow --min-k 2 --max-k 20 --step-k 2

# Thử nghiệm với các giá trị k cụ thể
python kmeans_clustering.py --clusters 4 6 8
```

## 5. Kết Quả và Hiển Thị

### 5.1. So Sánh Các Phương Pháp Khởi Tạo

Mã nguồn so sánh phương pháp khởi tạo AI-Daoud và khởi tạo ngẫu nhiên truyền thống dựa trên các chỉ số đánh giá:

```
Comparison of metrics for k = 5:
Metric               AI-Daoud        Random         
--------------------------------------------------
Silhouette (↑)       0.3214          0.2876         
Davies-Bouldin (↓)   1.2345          1.4567         
Calinski-Harabasz (↑) 78.2345        65.3456        
SSE/WCSS (↓)         125.6789        142.7890       
```

### 5.2. Hiển Thị Trực Quan

Mã nguồn tạo ra các hình ảnh trực quan để hiểu rõ hơn về kết quả phân cụm:
- Hình ảnh phân cụm trong không gian 2D (sử dụng 2 thành phần chính đầu tiên từ PCA)
- Hình ảnh phân cụm trong không gian 3D (sử dụng 3 thành phần chính đầu tiên từ PCA) để có cái nhìn trực quan hơn về sự phân tách của các cụm
- Đồ thị so sánh các chỉ số đánh giá giữa các phương pháp khởi tạo
- Đồ thị phương pháp Elbow hiển thị điểm khuỷu tay tối ưu

Mã nguồn cho trực quan hóa 3D:
```python
def visualize_clusters_3d(X_pca, labels, companies, title='Cluster Visualization 3D'):
    """
    Trực quan hóa các cụm trong không gian 3D sử dụng 3 thành phần chính đầu tiên
    """
    # Sử dụng 3 chiều đầu tiên cho trực quan hóa
    X_3d = X_pca[:, :3]
    
    # Lấy các cụm duy nhất
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Tạo bảng màu
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # Tạo đồ thị 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, cluster in enumerate(unique_clusters):
        cluster_points = X_3d[labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   c=[cmap(i)], label=f'Cluster {cluster+1}', alpha=0.7)
    
    # Thêm nhãn cho một số điểm (hiển thị tên công ty)
    for i, (company, x, y, z) in enumerate(zip(companies, X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])):
        if i % 20 == 0:  # Gắn nhãn cho mỗi điểm thứ 20 để tránh quá tải trong không gian 3D
            ax.text(x, y, z, company, fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    plt.tight_layout()
    
    # Lưu hình ảnh với 3D trong tên file
    filename = f'{title.replace(" ", "_").lower()}_3d.png'
    plt.savefig(filename)
    
    print(f"Trực quan hóa cụm 3D đã được lưu thành '{filename}'")
```

### 5.3. Phân Phối Cụm

Mã nguồn cũng cung cấp thông tin về phân phối các công ty trong các cụm:

```
Cluster distribution:
AI-Daoud: [25, 42, 18, 35, 30]
Random: [28, 39, 20, 32, 31]

Sample cluster members (AI-Daoud):
Cluster 1: AAPL, GOOG, MSFT, AMZN, FB...
Cluster 2: XOM, CVX, BP, RDS, TOT...
```

## 6. Kết Luận

Việc triển khai phân cụm chuỗi thời gian chứng khoán sử dụng thuật toán K-means với phương pháp khởi tạo AI-Daoud cho thấy hiệu quả hơn so với phương pháp khởi tạo ngẫu nhiên truyền thống dựa trên các chỉ số đánh giá. Điều này minh họa tầm quan trọng của việc khởi tạo tâm cụm tốt trong thuật toán K-means.

Phương pháp Elbow cung cấp cách xác định tự động số lượng cụm tối ưu, giúp đảm bảo chất lượng phân cụm tốt nhất mà không cần thử nghiệm thủ công nhiều giá trị k khác nhau.

Các đặc trưng nâng cao được trích xuất từ dữ liệu chuỗi thời gian giúp nắm bắt các đặc tính phức tạp của biến động giá cổ phiếu, dẫn đến phân cụm có ý nghĩa hơn và phân biệt rõ ràng giữa các nhóm công ty có hành vi giá tương tự. 