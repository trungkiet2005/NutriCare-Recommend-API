
import os
import requests
import pandas as pd
from tqdm import tqdm

def download_nhanes_data():
    # Định nghĩa chu kỳ và mã ký tự tương ứng
    cycles = [
        ('2003-2004', 'C', '0304'),
        ('2005-2006', 'D', '0506'),
        ('2007-2008', 'E', '0708'),
        ('2009-2010', 'F', '0910'),
        ('2011-2012', 'G', '1112'),
        ('2013-2014', 'H', '1314'),
        ('2015-2016', 'I', '1516'),
        ('2017-2018', 'J', '1718'),
    ]
    
    # Định nghĩa các bộ dữ liệu cần tải (type, code)
    datasets = [
        ('demographic', 'DEMO'),
        ('dietary', 'DR1TOT'),
        ('dietary', 'DR2TOT'),
        ('dietary', 'DR1IFF'),
        ('dietary', 'DR2IFF'),
        ('examination', 'BMX'),
        ('examination', 'BPX'),
        ('laboratory', 'L40'),
        ('laboratory', 'TRIGLY'),
        ('laboratory', 'BIOPRO'),
        ('questionnaire', 'DUQ'),
    ]
    
    # Thêm một chu kỳ đặc biệt 2017-2020 có cấu trúc khác
    special_cycle = ('2017-2020', 'P', '1720')
    
    # Thư mục gốc để lưu dữ liệu
    data_root = '../data/'
    
    # Tạo thư mục cho mỗi chu kỳ và loại dữ liệu
    for _, _, year_code in cycles + [special_cycle]:
        for data_type, _ in datasets:
            directory = os.path.join(data_root, year_code, data_type)
            os.makedirs(directory, exist_ok=True)
    
    # Tải dữ liệu cho các chu kỳ thông thường
    for cycle, char, year_code in tqdm(cycles, desc="Downloading cycles"):
        for data_type, data_code in tqdm(datasets, desc=f"Downloading datasets for {cycle}", leave=False):
            filename = f"{data_code}_{char}.XPT"
            url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{filename}"
            
            output_path = os.path.join(data_root, year_code, data_type, filename)
            
            # Kiểm tra nếu file đã tồn tại
            if os.path.exists(output_path):
                print(f"File already exists: {output_path}")
                continue
            
            try:
                print(f"Downloading {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Kiểm tra lỗi HTTP
                
                # Lưu file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {output_path}")
                
            except requests.exceptions.HTTPError as e:
                print(f"Error downloading {url}: {e}")
    
    # Tải dữ liệu cho chu kỳ đặc biệt 2017-2020
    cycle, char, year_code = special_cycle
    for data_type, data_code in tqdm(datasets, desc=f"Downloading datasets for {cycle}"):
        filename = f"P_{data_code}.XPT"
        url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{data_type}/{filename}"
        
        output_path = os.path.join(data_root, year_code, data_type, filename)
        
        # Kiểm tra nếu file đã tồn tại
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            continue
        
        try:
            print(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            
            # Lưu file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {output_path}")
            
        except requests.exceptions.HTTPError as e:
            print(f"Error downloading {url}: {e}")
    
    print("Download completed!")

def download_fndds_data():
    # URLs cho dữ liệu FNDDS
    fndds_urls = [
        ("https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx", "2015-2016 Ingredients.xlsx"),
        ("https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx", "2017-2018 Ingredients.xlsx"),
        ("https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2019-2020%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Ingredients.xlsx", "2019-2020 Ingredients.xlsx")
    ]
    
    data_root = '../data/'
    os.makedirs(data_root, exist_ok=True)
    
    for url, filename in tqdm(fndds_urls, desc="Downloading FNDDS data"):
        output_path = os.path.join(data_root, filename)
        
        # Kiểm tra nếu file đã tồn tại
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            continue
        
        try:
            print(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            
            # Lưu file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {output_path}")
            
        except requests.exceptions.HTTPError as e:
            print(f"Error downloading {url}: {e}")
    
    print("FNDDS download completed!")

# Thực thi hàm download
if __name__ == "__main__":
    # Tạo thư mục processed_data nếu chưa tồn tại
    os.makedirs('../processed_data', exist_ok=True)
    
    print("Starting NHANES data download...")
    download_nhanes_data()
    
    print("Starting FNDDS data download...")
    download_fndds_data()
    
    print("All downloads completed successfully!")