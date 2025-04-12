# Tạo file fix.py mới với nội dung sau:
import os
import shutil

# Tạo file này trong thư mục gốc của Space
def apply_fix():
    # Tạo file trống để ngăn utils.py tải BERT
    with open("no_bert.py", "w") as f:
        f.write("""
from transformers import BertTokenizer, BertModel
def dummy(*args, **kwargs):
    return None
BertTokenizer.from_pretrained = dummy
BertModel.from_pretrained = dummy
""")

    # Sửa lại utils.py để import no_bert
    if os.path.exists("utils.py"):
        with open("utils.py", "r") as f:
            content = f.read()
        
        # Thêm dòng import vào đầu file
        fixed_content = "import no_bert\n" + content
        
        # Lưu bản sao
        shutil.copy("utils.py", "utils.py.backup")
        
        # Ghi file mới
        with open("utils.py", "w") as f:
            f.write(fixed_content)
        
        print("Đã sửa file utils.py")
    else:
        print("Không tìm thấy file utils.py")

if __name__ == "__main__":
    apply_fix()