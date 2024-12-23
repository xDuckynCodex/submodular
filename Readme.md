<h1 align="center">Báo cáo nghiên cứu</h1>
<h3 align="center">Less is More: Fewer Interpretable Region via Submodular Subset Selection</h3>

## 🛠️ Cài đặt môi trường
Clone dự án về máy tính

```Terminal
git clone 
cd submodular/
```
Cài đặt thư viện CLIP
```
git clone org-14957082@github.com:openai/CLIP.git
pip install ./CLIP

```
Cài đặt môi trường cho dự án sử dụng `Anaconda`

```Terminal
conda env create -f environment.yml
conda activate submodular
```
Tạo thư mục chứa mô hình
```
mkdir -p .checkpionts/CLIP
```

## Chạy code
Khởi động Jupyter server để chạy file notebook
```
jupyter notebook
```
Áp dụng phương pháp này cho mô hình ViT (Vision Transformer) trong file [submodular-clip-vitl](visual/submodular-clip-vitl.ipynb)

*Chú ý:* Sửa `image_path`, `download_root` theo đường dẫn trên máy tính tránh lỗi
