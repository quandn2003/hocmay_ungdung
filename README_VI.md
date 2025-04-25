# Phân Cụm Chuỗi Thời Gian Dữ Liệu Chứng Khoán

Dự án này thực hiện phân cụm các chuỗi thời gian dữ liệu chứng khoán bằng thuật toán K-means với phương pháp khởi tạo AI-Daoud, so sánh với phương pháp khởi tạo ngẫu nhiên truyền thống.

## Tổng Quan

Mục đích của dự án này là gom cụm các công ty dựa trên sự tương đồng trong biến động giá cổ phiếu theo thời gian. Triển khai sử dụng phương pháp khởi tạo nâng cao (AI-Daoud) cho thuật toán phân cụm K-means và so sánh với phương pháp khởi tạo ngẫu nhiên truyền thống.

## Tính Năng

- Trích xuất đặc trưng chuỗi thời gian nâng cao (miền thời gian, miền tần số, thống kê)
- Phân cụm K-means với phương pháp khởi tạo AI-Daoud
- So sánh với phương pháp khởi tạo ngẫu nhiên truyền thống
- Tự động phát hiện số lượng cụm tối ưu bằng phương pháp Elbow
- Đánh giá sử dụng nhiều chỉ số: Silhouette, Davies-Bouldin, Calinski-Harabasz, SSE/WCSS
- Trực quan hóa cụm trong không gian 2D và 3D sử dụng PCA

## Định Dạng Dữ Liệu

Bộ dữ liệu bao gồm các chuỗi thời gian giá cổ phiếu được lưu trữ trong các file .dat trong thư mục `./data`. Mỗi file đại diện cho một công ty và chứa giá cổ phiếu được thu thập tại các khoảng thời gian đều đặn.

## Cài Đặt

1. Clone repository:
   ```
   git clone https://github.com/your-username/time-series-stock-clustering.git
   cd time-series-stock-clustering
   ```

2. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

## Sử Dụng

Chạy script phân cụm chính:
```
python kmeans_clustering.py
```

Mặc định, script sẽ thử nghiệm với các giá trị k là 3, 5, 7 và 10. Bạn có thể chỉ định các giá trị cụm tùy chỉnh:

```
# Thử nghiệm với các cụm 4, 6 và 8
python kmeans_clustering.py --clusters 4 6 8

# Thử nghiệm với một giá trị cụm duy nhất
python kmeans_clustering.py -k 5
```

Để sử dụng phương pháp Elbow để tự động tìm số lượng cụm tối ưu:

```
# Chạy phương pháp Elbow với phạm vi mặc định (2-15)
python kmeans_clustering.py --elbow

# Chạy phương pháp Elbow với phạm vi tùy chỉnh
python kmeans_clustering.py --elbow --min-k 2 --max-k 20 --step-k 2
```

## Kết Quả

Script tạo ra:
- Kết quả đầu ra trên terminal với các chỉ số đánh giá và phân phối cụm
- Trực quan hóa cụm trong không gian 2D và 3D
- Biểu đồ so sánh các chỉ số đánh giá
- Biểu đồ phương pháp Elbow (khi sử dụng tùy chọn --elbow)

## Chi Tiết Triển Khai

### Thuật Toán Khởi Tạo AI-Daoud

Thuật toán khởi tạo AI-Daoud cho K-means hoạt động như sau:
1. Tính phương sai của mỗi thuộc tính đặc trưng
2. Tìm thuộc tính có phương sai lớn nhất
3. Sắp xếp các điểm dữ liệu theo thuộc tính này
4. Chia dữ liệu đã sắp xếp thành k nhóm có kích thước bằng nhau
5. Tìm giá trị trung vị của mỗi nhóm
6. Sử dụng các điểm dữ liệu tương ứng với các giá trị trung vị làm tâm cụm ban đầu

### Trích Xuất Đặc Trưng

Triển khai trích xuất nhiều loại đặc trưng từ dữ liệu chuỗi thời gian:
- Đặc trưng miền thời gian (thống kê, độ biến động,...)
- Đặc trưng miền tần số (phân tích phổ, FFT,...)
- Đặc trưng thống kê (tự tương quan, phân tích xu hướng,...)

## Tệp Tin

- `kmeans_clustering.py`: Triển khai phân cụm chính
- `time_series_features.py`: Trích xuất đặc trưng nâng cao cho dữ liệu chuỗi thời gian
- `BaoCao.md`: Báo cáo chi tiết bằng tiếng Việt
- `README_VI.md`: Tài liệu tiếng Việt
- `data/`: Thư mục chứa dữ liệu chuỗi thời gian dạng .dat

## Ngôn Ngữ

- [Tài liệu tiếng Anh](README.md)
- [Tài liệu tiếng Việt](README_VI.md)
- [Báo cáo chi tiết tiếng Việt](BaoCao.md)

## Giấy Phép

MIT 