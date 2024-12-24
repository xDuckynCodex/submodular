<h1 align="center">Báo cáo nghiên cứu</h1>
<h3 align="center">Less is More: Fewer Interpretable Region via Submodular Subset Selection</h3>

## 🛠️ Cài đặt môi trường
Clone dự án về máy tính

```Terminal
git clone https://github.com/xDuckynCodex/submodular.git
cd submodular/
```
Cài đặt môi trường cho dự án sử dụng `Anaconda`

```Terminal
conda create -n submodular python=3.10
conda activate submodular
conda install jupyter
```
Cài đặt thư viện CLIP và các thư viện khác
```
git clone org-14957082@github.com:openai/CLIP.git
pip install ./CLIP
pip install -r requirements.txt
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
Kết nối file trên với server `Jupyter` để thực thi code

Áp dụng phương pháp này cho mô hình ViT-L/14 (Vision Transformer) trong file [submodular-clip-vitl](visual/submodular-clip-vitl.ipynb)

*Chú ý:* Sửa `image_path`, `download_root` theo đường dẫn trên máy tính tránh lỗi

### Đánh giá
#### Ưu điểm
Phương pháp xây dựng hàm submodular dựa trên các đặc tính ràng buộc và kết hợp với thuật toán Tìm kiếm tham lam để đưa ra kết quả là một tập hợp hữu hạn phần tử được công bố trong bài báo **“Less is more: Fewer interpretable region via Submodular subset selection”** của nhóm tác giả: Ruoyu Chen, Hua Zhang, Siyuan Liang, Jingzhi Li1, Xiaochun Cao đã giải quyết được nhiều vấn đề liên quan đến hiệu suất và đem lại tính chính xác cao trong việc đưa ra quyết định của mô hình đối với bài toán trích chọn đặc trưng ảnh. Cụ thể:
1. **Hàm submodular giúp xác định các tập đặc trưng mà thông tin chồng chéo giữa chúng được giảm thiểu:** Bằng cách sử dụng có hiệu quả các ràng buộc: điểm tin cậy (Confidence Score) giúp xác định các vùng hình ảnh phù hợp với phân phối, đảm bảo độ chính xác cao; điểm hiệu quả (Effectiveness Score) nhằm đánh giá hiệu quả của một phần tử với một tập hợp và đo độ đa dạng trong tập hợp kết quả; điểm nhất quán (Consistency Score) để đảm bảo các vùng được chọn phù hợp với ngữ nghĩa hình ảnh, mục tiêu cụ thể; điểm cộng tác (Collaboration Score) có tác dụng đánh giá hiệu ứng tập thể của các phần tử trong tập hợp kết quả; kết hợp với các trọng số để cân bằng và điều chỉnh mức độ quan trọng của từng thành phần, đã giúp đưa ra một tập hợp các phần tử đa dạng về ngữ nghĩa, cộng tác tốt trong việc bổ sung thông tin và có độ tin cậy cao, đồng thời giảm thiểu và đưa ra những giải thích hợp lý trong trường hợp mô hình đưa ra những quyết định sai.
2. **Hàm submodular giúp làm giảm số lượng đặc trưng mà không làm giảm hiệu suất:** Tập hợp đầu ra gồm hữu hạn phần tử, nhưng mỗi phần tử đều được lựa chọn kĩ càng, đảm bảo đáp ứng đầy đủ các ràng buộc đã được xây dựng từ trước nên đảm bảo về mặt biểu hiện ngữ nghĩa của ảnh đầu vào. Điều này giúp các mô hình đưa ra quyết định một cách chính xác mà không cần xét quá nhiều vùng con trong ảnh.
3. **Hàm submodular giúp tiết kiệm tài nguyên, cân bằng giữa chất lượng và chi phí:** Việc chỉ lựa chọn ra hữu hạn các phần tử để biểu thị hình ảnh đầu vào sẽ giúp các mô hình đưa ra quyết định nhanh chóng, tiết kiệm thời gian và lượng tài nguyên dành cho quá trình tính toán. Bên cạnh đó, việc sử dụng kết hợp thuật toán Tìm kiếm tham lam giúp các phần tử được chọn vẫn đảm bảo chất lượng đầu ra so với tập hợp tối ưu, tạo nên sự cân bằng về chất lượng quyết định của mô hình và giảm thiểu chi phí, thời gian tính toán.
#### Nhược điểm
1. **Hàm submodular yêu cầu thời gian và chi phí tính toán lớn nếu số lượng vùng con tăng lên:** Hàm submodular có tính chất lợi ích biên giảm dần (diminishing returns), tức là khi thêm một phần tử vào một tập hợp nhỏ, lợi ích tăng thêm thường lớn hơn so với khi thêm phần tử đó vào một tập hợp lớn. Khi số lượng vùng con tăng lên, thời gian tính toán hàm submodular cũng tăng đáng kể vì: khi chia ảnh thành m vùng con, tổng số tập hợp con của tập V là $2^m$. Nếu thuật toán cần duyệt qua tất cả các tập hợp con để tìm tập tối ưu, thời gian tính toán sẽ tăng theo hàm mũ với m. Bên cạnh đó, trong thuật toán Tìm kiếm tham lam, mỗi khi thêm một phần tử $\alpha$ vào tập S, ta cần tính giá trị $\mathcal{F}\left(S \cup \left\{\alpha\right\}\right)$ cho tất cả các vùng còn lại trong tập V\S. Khi m tăng, số lần tính toán hàm $\mathcal{F}$ cũng tăng theo, dẫn đến thời gian và chi phí tính toán tăng lên.
2. **Hàm Submodular không thực sự hiệu quả nếu các đặc trưng độc lập và không có sự chồng chéo thông tin:** Nhược điểm này xuất phát từ bản chất của hàm submodular, vốn dựa vào tính chất "lợi ích biên giảm dần" để ưu tiên chọn các phần tử có giá trị bổ sung cho tập hợp hiện tại. Khi các đặc trưng độc lập, việc lựa chọn một đặc trưng không làm thay đổi lợi ích của các đặc trưng còn lại, dẫn đến tính bổ sung thông tin không được tận dụng. Trong trường hợp này, hàm submodular trở nên dư thừa vì không cần đến các phương pháp tối ưu hóa phức tạp; thay vào đó, có thể sử dụng các cách tiếp cận đơn giản hơn để chọn đặc trưng. Điều này làm giảm hiệu quả và tính ứng dụng của hàm submodular trong các bài toán mà thông tin giữa các đặc trưng không liên quan hoặc chồng chéo.