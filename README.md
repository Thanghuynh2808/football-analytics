# Football Analytics - Team Classification & Player Detection

## Tổng quan dự án

Dự án **Football Analytics** là một hệ thống phân tích video bóng đá sử dụng AI để:
- Phát hiện và theo dõi cầu thủ trong video bóng đá
- Phân loại cầu thủ theo đội bóng dựa trên màu áo
- Xử lý và phân tích video bóng đá theo thời gian thực

## Tính năng chính

### 🔍 Phát hiện cầu thủ (YOLO Detection)
- Sử dụng mô hình YOLO (You Only Look Once) để phát hiện cầu thủ trong video
- Hỗ trợ xử lý từng frame hoặc toàn bộ video
- Confidence threshold có thể điều chỉnh
- Xuất kết quả với bounding boxes

### 🎨 Phân loại đội bóng (Team Classification)
- Phân tích màu sắc áo đấu để phân loại cầu thủ theo đội
- Sử dụng các thuật toán machine learning (K-Means, Gaussian Mixture Model)
- Xử lý đa không gian màu (BGR, HSV, LAB)
- Tự động huấn luyện từ video input

### 🎥 Xử lý video
- Hỗ trợ nhiều định dạng video
- Xử lý frame theo frame hoặc batch processing
- Xuất video đã được annotate
- Lưu kết quả phân tích

## Cấu trúc dự án

```
football_analytics/
├── data/
│   ├── raw/                    # Video gốc đầu vào
│   ├── interim/               # Dữ liệu trung gian (frames, ảnh)
│   └── processed/             # Video và dữ liệu đã xử lý
├── models/
│   ├── yolo/                  # Mô hình YOLO weights
│   │   ├── best.pt           # Mô hình YOLO đã train
│   │   └── jersey.pt         # Mô hình chuyên phát hiện áo
│   └── classifier/           # Mô hình phân loại đội
│       └── team_classifier.pkl
├── src/
│   ├── detection/            # Module phát hiện
│   │   └── yolo_infer.py    # YOLO inference engine
│   ├── teams/               # Module phân loại đội
│   │   ├── team_classifier.py        # Core classifier
│   │   └── team_classifier_runner.py # Runner script
│   └── pipeline/            # Pipeline xử lý tổng thể
│       └── runner.py        # Main pipeline runner
└── README.md
```

## Cài đặt và Setup

### Yêu cầu hệ thống
- Python 3.8+
- GPU (khuyến nghị) hoặc CPU
- RAM tối thiểu 8GB

### Cài đặt dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install torch torchvision
pip install scikit-learn
pip install supervision
pip install matplotlib
pip install tqdm
```

### Tải mô hình
1. Đặt mô hình YOLO vào thư mục `models/yolo/`
2. Nếu có mô hình team classifier đã train, đặt vào `models/classifier/`

## Sử dụng

### 1. Phát hiện cầu thủ trong video

```bash
cd src/detection
python yolo_infer.py --video "../../data/raw/your_video.mp4" --frame 1000 --output "../../data/interim/detected_frame.jpg" --show
```

**Tham số:**
- `--video, -v`: Đường dẫn video input
- `--frame, -f`: Số frame cần xử lý (mặc định: 2000)
- `--model, -m`: Đường dẫn mô hình YOLO (mặc định: best.pt)
- `--conf, -c`: Ngưỡng confidence (mặc định: 0.5)
- `--output, -o`: Đường dẫn lưu kết quả
- `--show, -s`: Hiển thị kết quả

### 2. Phân loại đội bóng

```bash
cd src/teams
python team_classifier_runner.py --video "../../data/raw/your_video.mp4" --train --save_model "../../models/classifier/team_model.pkl"
```

**Workflow phân loại đội:**
1. **Training phase**: Phân tích video để học màu sắc áo đội
2. **Classification phase**: Áp dụng mô hình đã train để phân loại
3. **Visualization**: Tạo video với màu sắc phân loại

### 3. Pipeline tổng thể

```python
from src.teams.team_classifier_runner import TeamClassifierRunner

# Khởi tạo
runner = TeamClassifierRunner("models/yolo/best.pt")

# Train classifier
runner.train_classifier("data/raw/video.mp4", "models/classifier/team_model.pkl")

# Xử lý video mới
runner.process_video("data/raw/new_video.mp4", "data/processed/output.mp4")
```

## Chi tiết kỹ thuật

### YOLO Detection Engine
- **File**: `src/detection/yolo_infer.py`
- **Chức năng**: Phát hiện và tracking cầu thủ
- **Input**: Video/Image
- **Output**: Bounding boxes với confidence scores

### Team Classifier
- **File**: `src/teams/team_classifier.py`
- **Thuật toán**: K-Means Clustering, Gaussian Mixture Model
- **Features**: 
  - Dominant colors (BGR, HSV, LAB)
  - Color histograms
  - Spatial color distribution
  - Texture features

### Xử lý màu sắc nâng cao
- Phát hiện vùng áo đấu chính xác (center + upper body)
- Lọc background bằng saturation threshold
- Chuẩn hóa features với StandardScaler
- Xử lý multiple color spaces

## Kết quả và Đánh giá

### Metrics đánh giá
- **Detection**: mAP (mean Average Precision), FPS
- **Classification**: Silhouette Score, Purity Score
- **Processing Speed**: Frames per second

### Sample Output
- Detected frame với bounding boxes: `data/interim/annotated_frame.jpg`
- Team classified frame: `data/interim/team_classified_frame.jpg`
- Processed video: `data/processed/team_classified_video.mp4`

## Tùy chỉnh và Mở rộng

### Thêm thuật toán phân loại mới
```python
# Trong team_classifier.py
def fit_custom_algorithm(self, features):
    # Implement your custom clustering algorithm
    pass
```

### Tùy chỉnh feature extraction
```python
def extract_custom_features(self, crop):
    # Add your custom feature extraction logic
    return custom_features
```

## Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**
   - Giảm batch size
   - Sử dụng CPU: `device="cpu"`

2. **Mô hình không load được**
   - Kiểm tra đường dẫn model
   - Đảm bảo file .pt có sẵn

3. **Video không đọc được**
   - Kiểm tra codec video
   - Thử chuyển đổi format video

4. **Phân loại đội không chính xác**
   - Tăng số lượng samples training
   - Điều chỉnh color space weights
   - Thử thuật toán clustering khác

## Đóng góp (Contributing)

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## Tác giả

- **Huynh Thang** - *Initial work*

## Ghi nhận (Acknowledgments)

- YOLOv8 by Ultralytics
- OpenCV team
- Scikit-learn contributors
- Supervision library

## Roadmap

### Tính năng sắp tới
- [ ] Real-time processing
- [ ] Multiple camera angles support
- [ ] Player tracking across frames
- [ ] Advanced statistics generation
- [ ] Web interface
- [ ] Mobile app support

### Cải tiến hiệu suất
- [ ] Model optimization
- [ ] GPU acceleration
- [ ] Batch processing optimization
- [ ] Memory usage optimization

---

*Dự án được phát triển để phục vụ mục đích nghiên cứu và ứng dụng trong phân tích thể thao.*
