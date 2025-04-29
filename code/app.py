import torch
import pandas as pd
import numpy as np
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json 
import math
import google.generativeai as genai
import re
import requests


from RCSYS_utils import *   
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SignedConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import Linear
from RCSYS_models import GraphGenerator, GraphChannelAttLayer, SignedGCN, LightGCN
from typing import List, Optional


# Thiết lập logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Định nghĩa đường dẫn gốc và thêm vào sys.path để import module dễ dàng hơn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(BASE_DIR)
sys.path.append(parent_dir)  # Thêm thư mục cha vào path để import RCSYS_models

try:
    from RCSYS_models import SGSL
except ImportError:
    logger.error("Không thể import SGSL từ RCSYS_models. Kiểm tra lại đường dẫn!")
    # Tạo lớp giả để tránh lỗi khi import
    class SGSL(nn.Module):
        def __init__(self, graph, embedding_dim,  feature_threshold=0.3, num_heads=4, num_layer=3):
            super(SGSL, self).__init__()

            self.num_users = graph['user'].num_nodes
            self.num_foods = graph['food'].num_nodes
            self.embedding_dim = embedding_dim
            self.num_heads = num_heads
            self.num_layer = num_layer
            self.feature_threshold = feature_threshold

            self.lin_dict = torch.nn.ModuleDict()
            for node_type in graph.node_types:
                self.lin_dict[node_type] = Linear(-1, embedding_dim)
            
            # Graph generators for feature and semantic graphs
            self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
            self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
            self.fusion = GraphChannelAttLayer(3)
            self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)


        def forward(self, feature_dict, edge_index, pos_edge_index, neg_edge_index):
            # Heterogeneous Feature Mapping.
            feature_dict = {
                node_type: self.lin_dict[node_type](x).relu_()
                for node_type, x in feature_dict.items()
            }

            # Generate the feature graph. The result is a adj_matrix with the same shape as adj_ori
            mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
            mask_ori = torch.ones_like(mask_feature)

            # Generate the semantic graph. The same. 
            z = self.signed_layer(pos_edge_index, neg_edge_index)
            mask_semantic = self.signed_layer.discriminate(z, edge_index)

            # Fusion with the original adj with attention 
            edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])

            edge_index_new = edge_index[:, edge_mask]
            sparse_size = self.num_users + self.num_foods
            sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], sparse_sizes=(
                sparse_size, sparse_size))
            
            # Use the new adj, convert back to edge_index, and perform a LightGCN 
            return self.lightgcn(sparse_edge_index)

# Kiểm tra xem pytorch đã được import chưa
logger.info(f"PyTorch version: {torch.__version__}")


# Hàm mới để tạo enhanced mapping từ food_id_name_mapping.json và food_tagging.csv
def create_enhanced_mapping():
    """
    Tạo mapping nâng cấp từ food_id_name_mapping.json và food_tagging.csv
    """
    try:
        logger.info("Đang tạo enhanced mapping từ food_id_name_mapping.json và food_tagging.csv")
        
        # Tìm file food_id_name_mapping.json ở nhiều vị trí
        mapping_paths = [
            os.path.join(BASE_DIR, 'food_id_name_mapping.json'),
            os.path.join(parent_dir, 'food_id_name_mapping.json'),
            'food_id_name_mapping.json'
        ]
        
        # Tìm file food_tagging.csv ở nhiều vị trí
        tagging_paths = [
            os.path.join(BASE_DIR, 'food_tagging.csv'),
            os.path.join(parent_dir, 'food_tagging.csv'),
            'food_tagging.csv',
            os.path.join(BASE_DIR, 'food_tagging_filter.csv'),
            os.path.join(parent_dir, 'food_tagging_filter.csv')
        ]
        
        # Tìm file mapping
        mapping_file = None
        for path in mapping_paths:
            if os.path.exists(path):
                mapping_file = path
                logger.info(f"Tìm thấy file mapping tại: {mapping_file}")
                break
        
        if not mapping_file:
            logger.error("Không tìm thấy file food_id_name_mapping.json")
            return {}
            
        # Tìm file tagging
        tagging_file = None
        for path in tagging_paths:
            if os.path.exists(path):
                tagging_file = path
                logger.info(f"Tìm thấy file tagging tại: {tagging_file}")
                break
        
        if not tagging_file:
            logger.error("Không tìm thấy file food_tagging.csv hoặc food_tagging_filter.csv")
            return {}
        
        # Đọc file food_id_name_mapping.json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            food_id_name_mapping = json.load(f)
        
        # Đọc file food_tagging.csv
        food_tagging_df = pd.read_csv(tagging_file)
        
        # Tạo mapping mới với cấu trúc tương tự us_to_vn_food_simple_mapping.json
        enhanced_mapping = {}
        
        for food_id, food_name in food_id_name_mapping.items():
            # Khởi tạo với tên món ăn và nguyên liệu trống
            enhanced_mapping[food_id] = [food_name, ""]
            
            # Tìm món ăn có tên tương ứng trong food_tagging_csv
            matching_food = food_tagging_df[food_tagging_df['Tên món ăn'] == food_name]
            if not matching_food.empty:
                # Lấy nguyên liệu nếu có
                if 'Nguyên liệu' in matching_food.columns:
                    ingredients = matching_food['Nguyên liệu'].iloc[0]
                    if pd.notna(ingredients):  # Kiểm tra nếu không phải NaN
                        enhanced_mapping[food_id][1] = str(ingredients)
            
            # Nếu không tìm thấy hoặc không có cột Nguyên liệu, thử với cột Tiêu đề
            if enhanced_mapping[food_id][1] == "" and 'Tiêu đề' in food_tagging_df.columns:
                matching_food = food_tagging_df[food_tagging_df['Tiêu đề'] == food_name]
                if not matching_food.empty and 'Nguyên liệu' in matching_food.columns:
                    ingredients = matching_food['Nguyên liệu'].iloc[0]
                    if pd.notna(ingredients):  # Kiểm tra nếu không phải NaN
                        enhanced_mapping[food_id][1] = str(ingredients)
        
        logger.info(f"Đã tạo enhanced mapping với {len(enhanced_mapping)} món ăn")
        return enhanced_mapping
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo enhanced mapping: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


# Biến toàn cục để lưu trữ mapping đã nâng cấp
_enhanced_mapping = None

def get_enhanced_mapping():
    """Hàm lazy loading cho enhanced mapping"""
    global _enhanced_mapping
    if _enhanced_mapping is None:
        _enhanced_mapping = create_enhanced_mapping()
    return _enhanced_mapping


# ---------------------------------------------------------------
def convert_to_python_native(obj):
    """
    Chuyển đổi các kiểu dữ liệu NumPy, PyTorch thành kiểu dữ liệu Python tiêu chuẩn
    
    Args:
        obj: Đối tượng cần chuyển đổi, có thể là kiểu dữ liệu bất kỳ
        
    Returns:
        Đối tượng đã được chuyển đổi sang kiểu dữ liệu Python tiêu chuẩn
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {convert_to_python_native(key): convert_to_python_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_native(item) for item in obj)
    else:
        return obj


def clean_float_values(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(v) for v in obj]
    else:
        return obj


# Thêm sau phần import hoặc các hàm tiện ích khác
# Biến toàn cục để lưu trữ graph và model đã tải
_graph = None
_model = None

def get_graph():
    """Hàm lazy loading cho graph"""
    global _graph
    if _graph is None:
        graph_paths = [
            os.path.join(parent_dir, 'vn_food_graph.pt'),
            os.path.join(BASE_DIR, 'vn_food_graph.pt')
        ]
        
        graph_file = None
        for path in graph_paths:
            if os.path.exists(path):
                graph_file = path
                break
        
        if graph_file is None:
            logger.error("Không tìm thấy file graph")
            raise FileNotFoundError("Không tìm thấy file graph")
        
        logger.info(f"Đang tải graph từ {graph_file}")
        _graph = torch.load(graph_file, map_location=torch.device('cpu'))
        logger.info(f"Đã tải graph thành công: {len(_graph.node_types)} loại node, {len(_graph.edge_types)} loại cạnh")
    
    return _graph

def get_model():
    """Hàm lazy loading cho model"""
    global _model
    if _model is None:
        graph = get_graph()
        
        logger.info("Đang khởi tạo model SGSL")
        _model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=FEATURE_THRESHOLD, num_layer=LAYERS)
        logger.info("Đã khởi tạo model thành công")
        
        model_paths = [
            os.path.join(BASE_DIR, 'vn_trained_model.pth'),
            os.path.join(parent_dir, 'vn_trained_model.pth')
        ]
        
        model_file = None
        for path in model_paths:
            if os.path.exists(path):
                model_file = path
                break
        
        if model_file is None:
            logger.error("Không tìm thấy file model")
            raise FileNotFoundError("Không tìm thấy file model")
        
        logger.info(f"Đang tải model weights từ {model_file}")
        _model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        _model.eval()
        logger.info("Đã tải model weights thành công")
    
    return _model

def convert_user_input_to_numeric(user_features):
    """
    Chuyển đổi các trường input dạng chữ sang số theo bảng mã hóa đã định nghĩa.
    """
    gender_map = {"Nam": 1, "Nữ": 2}
    age_group_map = {
        "Dưới 18 tuổi": 1,
        "Từ 18 đến 24 tuổi": 2,
        "Từ 25 đến 34 tuổi": 3,
        "Từ 35 đến 44 tuổi": 4,
        "Từ 45 đến 54 tuổi": 5,
        "Từ 55 đến 64 tuổi": 6,
        "Trên 65 tuổi": 7
    }
    race_map = {
        "Kinh": 0,
        "Hoa": 1,
        "Chăm": 2,
        "Khmer": 3,
        "Tày": 4,
        "Khác": 5
    }
    household_income_map = {
        "Dưới 3 triệu/tháng": 0,
        "3 - 5 triệu/tháng": 1,
        "5 - 7 triệu/tháng": 2,
        "7 - 10 triệu/tháng": 3,
        "10 - 15 triệu/tháng": 4,
        "15 - 20 triệu/tháng": 5,
        "20 - 25 triệu/tháng": 6,
        "25 - 30 triệu/tháng": 7,
        "30 - 40 triệu/tháng": 8,
        "40 - 50 triệu/tháng": 9,
        "50 - 60 triệu/tháng": 10,
        "Trên 60 triệu/tháng": 11
    }
    education_map = {
        "Chưa đi học": 0,
        "Tiểu học": 1,
        "Trung học cơ sở": 2,
        "Trung học phổ thông": 3,
        "Trung cấp": 4,
        "Cao đẳng": 5,
        "Đại học": 6,
        "Sau đại học": 7,
        "Thạc sĩ": 8,
        "Tiến sĩ": 9
    }

    result = user_features.copy()
    if "gender" in result and isinstance(result["gender"], str):
        result["gender"] = gender_map.get(result["gender"], result["gender"])
    if "age_group" in result and isinstance(result["age_group"], str):
        result["age_group"] = age_group_map.get(result["age_group"], result["age_group"])
    if "race" in result and isinstance(result["race"], str):
        result["race"] = race_map.get(result["race"], result["race"])
    if "household_income" in result and isinstance(result["household_income"], str):
        result["household_income"] = household_income_map.get(result["household_income"], result["household_income"])
    if "education" in result and isinstance(result["education"], str):
        result["education"] = education_map.get(result["education"], result["education"])
    return result

def create_user_feature_tensor(user_features, graph=None):
    """
    Tạo tensor đặc trưng cho user mới từ thông tin đầu vào, 
    phù hợp với cấu trúc vector đặc trưng của user trong graph
    
    Args:
        user_features (dict): Đặc điểm của user mới, có thể bao gồm:
            - gender: 1 (nam) hoặc 2 (nữ)
            - age_group: Nhóm tuổi (1-7)
            - race: Chủng tộc (0-5)
            - household_income: Mức thu nhập (0-11)
            - education: Trình độ học vấn (0-9)
            - tags: Danh sách tags sức khỏe/dinh dưỡng
        graph (HeteroData, optional): Dữ liệu graph, nếu cung cấp sẽ được dùng để 
                                     tham chiếu cấu trúc vector đặc trưng
        
    Returns:
        tensor: Vector đặc trưng của user mới, định dạng tương tự với users trong graph
    """
    try:
        logger.info(f"Tạo tensor đặc trưng cho user mới với thông tin: {user_features}")
        
        # Kiểm tra cấu trúc tensor của user hiện có trong graph nếu được cung cấp
        feature_dim = 38  # Giá trị mặc định nếu không có graph
        
        if graph is not None and hasattr(graph['user'], 'x') and graph['user'].x.shape[1] > 0:
            feature_dim = int(graph['user'].x.shape[1])
            logger.info(f"Kích thước vector đặc trưng từ graph: {feature_dim}")
        
        # Khởi tạo vector đặc trưng với giá trị 0
        features = [0] * feature_dim
        
        # Nếu có graph, lấy cấu trúc cụ thể từ một user đầu tiên làm mẫu
        if graph is not None and graph['user'].num_nodes > 0:
            sample_user_features = graph['user'].x[0].cpu().numpy()
            logger.info(f"Hình dạng của sample user features: {sample_user_features.shape}")
            
            # Giả sử cấu trúc vector là:
            # - Vị trí 0-1: One-hot encoding của gender (2 vị trí)
            # - Vị trí 2-8: One-hot encoding của age_group (7 vị trí)
            # - Vị trí 9-14: One-hot encoding của race (6 vị trí)
            # - Vị trí 15-26: One-hot encoding của household_income (12 vị trí)
            # - Vị trí 27-36: One-hot encoding của education (10 vị trí)
            
            # 1. Gender encoding (vị trí 0-1)
            if 'gender' in user_features:
                gender = int(user_features['gender'])
                if gender == 1:  # nam
                    features[0] = 1
                    features[1] = 0
                elif gender == 2:  # nữ
                    features[0] = 0
                    features[1] = 1
            
            # 2. Age group encoding (vị trí 2-8)
            if 'age_group' in user_features:
                age_group = int(user_features['age_group'])
                if 1 <= age_group <= 7:
                    features[1 + age_group] = 1  # +1 vì index bắt đầu từ 0, age_group từ 1
            
            # 3. Race encoding (vị trí 9-14)
            if 'race' in user_features:
                race = int(user_features['race'])
                if 0 <= race <= 5:
                    features[9 + race] = 1
            
            # 4. Household income encoding (vị trí 15-26)
            if 'household_income' in user_features:
                income = int(user_features['household_income'])
                if 0 <= income <= 11:
                    features[15 + income] = 1
            
            # 5. Education encoding (vị trí 27-36)
            if 'education' in user_features:
                education = int(user_features['education'])
                if 0 <= education <= 9:
                    features[27 + education] = 1
        else:
            # Nếu không có graph, sử dụng phương pháp đơn giản hơn
            current_idx = 0
            
            # 1. Gender encoding
            if 'gender' in user_features:
                gender = int(user_features['gender'])
                gender_feature = [1, 0] if gender == 1 else [0, 1]  # 1: nam, 2: nữ
                features[current_idx:current_idx+2] = gender_feature
            current_idx += 2
            
            # 2. Age group encoding
            if 'age_group' in user_features:
                age_group = int(user_features['age_group'])
                age_feature = [0] * 7  # 7 nhóm tuổi
                if 1 <= age_group <= 7:
                    age_feature[age_group-1] = 1
                features[current_idx:current_idx+7] = age_feature
            current_idx += 7
            
            # 3. Race encoding
            if 'race' in user_features:
                race = int(user_features['race'])
                race_feature = [0] * 6  # 6 chủng tộc (0-5)
                if 0 <= race <= 5:
                    race_feature[race] = 1
                features[current_idx:current_idx+6] = race_feature
            current_idx += 6
            
            # 4. Household income encoding
            if 'household_income' in user_features:
                income = int(user_features['household_income'])
                income_feature = [0] * 12  # 12 mức thu nhập (0-11)
                if 0 <= income <= 11:
                    income_feature[income] = 1
                features[current_idx:current_idx+12] = income_feature
            current_idx += 12
            
            # 5. Education encoding
            if 'education' in user_features:
                education = int(user_features['education'])
                education_feature = [0] * 10  # 10 trình độ học vấn (0-9)
                if 0 <= education <= 9:
                    education_feature[education] = 1
                features[current_idx:current_idx+10] = education_feature
            current_idx += 10
            
            # Đảm bảo rằng vector đặc trưng có đủ kích thước
            features = features[:feature_dim]
        
        # Chuyển đổi thành tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Thêm batch dimension
        logger.info(f"Đã tạo tensor đặc trưng với kích thước: {feature_tensor.shape}")
        
        return feature_tensor
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo user feature tensor: {str(e)}")
        logger.error(f"Dữ liệu đầu vào: {user_features}")
        import traceback
        logger.error(traceback.format_exc())
        raise Exception(f"Lỗi khi tạo user feature tensor: {str(e)}")
    

def get_user_node_basic_info(graph, user_index):
    """
    Lấy thông tin cơ bản của user từ graph dựa trên index
    
    Args:
        graph (HeteroData): Dữ liệu graph
        user_index (int): Index của user trong graph
        
    Returns:
        dict: Thông tin cơ bản của user, đã chuyển đổi sang kiểu dữ liệu Python tiêu chuẩn
    """
    try:
        user_id = int(graph['user'].node_id[user_index].item())
        
        # Khởi tạo thông tin cơ bản
        user_info = {
            'user_id': user_id,
            'index': int(user_index)
        }
        
        # Phân tích vector đặc trưng để lấy thông tin
        if hasattr(graph['user'], 'x'):
            features = graph['user'].x[user_index].cpu().numpy()
            
            # Giả sử cấu trúc như đã mô tả trong create_user_feature_tensor
            # Giới tính
            if features.shape[0] >= 2:
                gender_idx = int(np.argmax(features[:2]))
                user_info['gender'] = int(gender_idx + 1)  # 1: nam, 2: nữ
            
            # Nhóm tuổi
            if features.shape[0] >= 9:
                age_idx = int(np.argmax(features[2:9]))
                user_info['age_group'] = int(age_idx + 1)
            
            # Chủng tộc
            if features.shape[0] >= 15:
                race_idx = int(np.argmax(features[9:15]))
                user_info['race'] = int(race_idx)
            
            # Thu nhập
            if features.shape[0] >= 27:
                income_idx = int(np.argmax(features[15:27]))
                user_info['household_income'] = int(income_idx)
            
            # Học vấn
            if features.shape[0] >= 37:
                education_idx = int(np.argmax(features[27:37]))
                user_info['education'] = int(education_idx)
        
        # Lấy tags nếu có
        if hasattr(graph['user'], 'tags'):
            user_info['tags'] = graph['user'].tags[user_index].cpu().numpy().tolist()
        
        # Lấy prompt nếu có
        if hasattr(graph['user'], 'prompt'):
            user_prompt = graph['user'].prompt[user_index]
            user_info['prompt'] = str(user_prompt) if user_prompt is not None else None
        
        return user_info
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin cơ bản của user: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'user_id': int(graph['user'].node_id[user_index].item()), 'error': str(e)}
    
    
def find_similar_users(new_user_features, graph, top_k=5, similarity_threshold=0.3):
    """
    Tìm kiếm các user tương tự trong graph dựa trên đặc điểm của user mới
    
    Args:
        new_user_features (dict): Thông tin đặc điểm của user mới với các khóa như:
            - 'gender': giới tính (1: nam, 2: nữ)
            - 'age_group': nhóm tuổi
            - 'race': chủng tộc
            - 'household_income': mức thu nhập
            - 'education': trình độ học vấn
            - 'tags': danh sách tags sức khỏe/dinh dưỡng
        graph (HeteroData): Dữ liệu graph đã tải
        top_k (int): Số lượng user tương tự muốn trả về
        similarity_threshold (float): Ngưỡng tương đồng tối thiểu (0-1)
    
    Returns:
        list: Danh sách các dictionary chứa thông tin top-k user tương tự nhất
              mỗi dictionary có dạng {'user_id': id, 'similarity': score}
    """
    try:
        logger.info(f"Tìm kiếm {top_k} user tương tự cho user mới với đặc điểm: {new_user_features}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Số lượng user trong graph
        num_users = int(graph['user'].num_nodes)
        logger.info(f"Tổng số user trong graph: {num_users}")
        
        # Chuyển đổi đặc điểm của new user thành tensor
        new_user_tensor = create_user_feature_tensor(new_user_features, graph).to(device)
        
        # Tính toán độ tương đồng với từng user trong graph
        similarities = []
        
        # Kiểm tra xem có tags hay không
        has_tags = 'tags' in new_user_features and new_user_features['tags'] and hasattr(graph['user'], 'tags')
        if has_tags:
            logger.info("Sử dụng thông tin tags để tính độ tương đồng")
            new_user_tags = torch.tensor(new_user_features['tags'], dtype=torch.float32).to(device)
        
        # Duyệt qua từng user trong graph
        for i in range(num_users):
            user_id = int(graph['user'].node_id[i].item())
            
            # Tính độ tương đồng dựa trên vector đặc trưng
            user_feature = graph['user'].x[i].to(device)
            feature_sim = float(F.cosine_similarity(new_user_tensor, user_feature.unsqueeze(0), dim=1).item())
            
            # Tính độ tương đồng dựa trên tags nếu có
            tag_sim = 0.0
            if has_tags:
                user_tag = graph['user'].tags[i].to(device)
                # Tính Jaccard similarity giữa tags
                intersection = torch.sum(torch.min(new_user_tags, user_tag))
                union = torch.sum(torch.max(new_user_tags, user_tag))
                tag_sim = float((intersection / (union + 1e-8)).item())
            
            # Phân tích chi tiết hơn về người dùng
            user_details = get_user_node_basic_info(graph, i)
            
            # Tính độ tương đồng kết hợp
            # Có thể điều chỉnh trọng số giữa feature_sim và tag_sim
            combined_sim = 0.6 * feature_sim + 0.4 * tag_sim if has_tags else feature_sim
            
            # Chỉ thêm vào danh sách nếu đạt ngưỡng tương đồng tối thiểu
            if combined_sim >= similarity_threshold:
                similarities.append({
                    'user_id': user_id,
                    'similarity': float(combined_sim),
                    'feature_similarity': float(feature_sim),
                    'tag_similarity': float(tag_sim) if has_tags else None,
                    'details': user_details
                })
        
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Lấy top-k kết quả
        top_similar_users = similarities[:top_k]
        
        logger.info(f"Đã tìm thấy {len(top_similar_users)} user tương tự (ngưỡng: {similarity_threshold})")
        
        # Chuyển đổi tất cả các giá trị sang kiểu dữ liệu Python tiêu chuẩn
        return convert_to_python_native(top_similar_users)
        
    except Exception as e:
        logger.error(f"Lỗi khi tìm user tương tự: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise Exception(f"Lỗi khi tìm user tương tự: {str(e)}")
        

#----------------------------------------------------------------
# Thêm vào phần định nghĩa schema
class NewUserInput(BaseModel):
    gender: str = "Nam" # "Nam" hoặc "Nữ"
    age_group: Optional[str] = "Dưới 18 tuổi"  # "Dưới 18 tuổi", "Từ 18 đến 24 tuổi", ...
    race: Optional[str] = "Kinh"  # "Kinh", "Hoa", "Chăm", "Khmer", "Tày", "Khác"
    household_income: Optional[str] = "Dưới 3 triệu/tháng"  # "Dưới 3 triệu/tháng", "3 - 5 triệu/tháng", ...
    education: Optional[str] = "Chưa đi học"  # "Chưa đi học", "Tiểu học", "Trung học cơ sở", ...
    tags: Optional[List[int]] = None  # Danh sách tags sức khỏe/dinh dưỡng (giữ nguyên nếu là số)
    similarity_threshold: Optional[float] = 0.3  # Ngưỡng tương đồng tối thiểu
    top_k: Optional[int] = 5  # Số lượng user tương tự muốn trả về
    symptom: Optional[List[str]] = None  # Danh sách triệu chứng
    spefical_diet: Optional[List[str]] = None  # Danh sách chế độ ăn đặc biệt
    disease: Optional[List[str]] = None  # Danh sách bệnh lý


# Import các function đã định nghĩa
def recommend_for_user(user_id, model, graph, k=20, excluded_food_ids=None):
    try:
        if excluded_food_ids is None:
            excluded_food_ids = set()
            
        user_idx = None
        for i, node_id in enumerate(graph['user'].node_id):
            if node_id.item() == user_id:
                user_idx = i
                break
        
        if user_idx is None:
            logger.warning(f"User ID {user_id} not found")
            return []
        
        device = next(model.parameters()).device
        
        # Chuyển đổi tất cả đến device của mô hình
        feature_dict = {key: x.to(device) for key, x in graph.x_dict.items()}
        edge_index = graph[('user', 'eats', 'food')].edge_index.to(device)
        edge_label_index = graph[('user', 'eats', 'food')].edge_label_index.to(device)
        
        # Forward pass qua mô hình
        users_emb_final, _, items_emb_final, _ = model.forward(feature_dict, edge_index, edge_label_index, edge_label_index)
        
        # Tính điểm cho tất cả thực phẩm
        user_emb = users_emb_final[user_idx].unsqueeze(0)
        scores = torch.mm(user_emb, items_emb_final.t()).squeeze()
        
        # Xác định các thực phẩm đã tiêu thụ cần loại trừ
        consumed_food_indices = []
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() == user_idx:
                consumed_food_indices.append(edge_index[1, i].item())
        
        # Đặt điểm của thực phẩm đã tiêu thụ thành -inf
        scores[consumed_food_indices] = -float('inf')
        
        # Get food IDs to exclude food_ids from the exclusion list
        food_node_ids = graph['food'].node_id.cpu().numpy()
        
        # Create a mask for excluded foods
        excluded_indices = []
        for i, food_id in enumerate(food_node_ids):
            if food_id.item() in excluded_food_ids:
                excluded_indices.append(i)
        
        # Đặt điểm của thực phẩm cần loại trừ thành -inf
        scores[excluded_indices] = -float('inf')
        
        # We need to get more than k items since some might be excluded
        _, indices = torch.topk(scores, min(len(scores), k*5))  # Get 5 times as many in case of exclusions
        indices = indices.cpu().numpy()
        
        # Convert indices to food_ids
        recommended_food_ids = []
        for idx in indices:
            food_id = graph['food'].node_id[idx].item()
            if food_id not in excluded_food_ids:
                recommended_food_ids.append(food_id)
                if len(recommended_food_ids) >= k*2:  # Get twice as many to have buffer for milk filtering
                    break
        
        return recommended_food_ids[:k*2]  # We'll filter milk-containing foods later
    except Exception as e:
        logger.error(f"Lỗi trong hàm recommend_for_user: {str(e)}")
        raise Exception(f"Lỗi trong hàm recommend_for_user: {str(e)}")


def get_user_node_info(user_id):
    # Tìm index của user_id trong graph
    graph_path = os.path.join(parent_dir, 'vn_food_graph.pt')
    
    logger.info(f"Đang tìm file graph tại: {graph_path}")
    if not os.path.exists(graph_path):
        alt_path = os.path.join(BASE_DIR, 'vn_food_graph.pt')
        logger.info(f"File không tồn tại, thử đường dẫn thay thế: {alt_path}")
        if os.path.exists(alt_path):
            graph_path = alt_path
        else:
            logger.error(f"Không tìm thấy file graph ở cả hai đường dẫn")
            raise FileNotFoundError(f"Không tìm thấy file graph")
    
    try:
        logger.info(f"Đang tải graph từ {graph_path}")
        graph = torch.load(graph_path, map_location=torch.device('cpu'))
        logger.info("Đã tải graph thành công")
        
        user_indices = (graph['user'].node_id == user_id).nonzero().flatten()
        if len(user_indices) == 0:
            logger.warning(f"User ID {user_id} không tồn tại trong dữ liệu.")
            return None
        
        user_index = user_indices[0].item()
        
        # Lấy thông tin cơ bản
        user_info = {
            'user_id': user_id,
            'index': user_index
        }
        
        # Lấy vector đặc trưng nếu có
        if hasattr(graph['user'], 'x'):
            user_info['features'] = graph['user'].x[user_index].cpu().numpy().tolist()  # Convert to list for JSON serialization
        
        # Lấy tags nếu có
        if hasattr(graph['user'], 'tags'):
            user_info['tags'] = graph['user'].tags[user_index].cpu().numpy().tolist()  # Convert to list for JSON serialization
        
        # Lấy prompt nếu có
        if hasattr(graph['user'], 'prompt') and len(graph['user'].prompt) > user_index:
            user_info['prompt'] = graph['user'].prompt[user_index]
        
        # Lấy danh sách món ăn đã dùng
        edge_type = ('user', 'eats', 'food')
        if edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index
            food_indices = edge_index[1][edge_index[0] == user_index].cpu().numpy()
            food_ids = [graph['food'].node_id[idx].item() for idx in food_indices]
            user_info['eaten_foods'] = food_ids
        
        return user_info
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin user node: {e}")
        raise Exception(f"Lỗi khi lấy thông tin user node: {e}")

# Bỏ hàm food_mapping_function vì chúng ta sẽ sử dụng enhanced_mapping thay cho mapping

# ===== Khởi tạo FastAPI =====
app = FastAPI(title="Food Recommendation API", 
              description="API để đề xuất món ăn dựa trên sở thích của người dùng",
              version="1.0.0")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins trong môi trường phát triển
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Khai báo input schema =====
class UserInput(BaseModel):
    user_id: int

# ===== Cấu hình mô hình =====
HIDDEN_DIM = 128
FEATURE_THRESHOLD = 0.3
LAYERS = 3

# ===== Hàm gợi ý API =====
@app.post("/recommend_for_user")
def get_recommendation_for_user(input: UserInput):
    try:
        user_id = int(input.user_id)  # Đảm bảo là Python int
        logger.info(f"Đang xử lý yêu cầu gợi ý cho user_id: {user_id}")

        # Lazy loading graph và model
        try:
            graph = get_graph()
            model = get_model()
        except Exception as e:
            logger.error(f"Lỗi khi tải graph hoặc model: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi tải graph hoặc model: {str(e)}"}
            
        # Lấy enhanced mapping
        try:
            data = get_enhanced_mapping()
            if not data:
                logger.error("Enhanced mapping trống hoặc không thể tạo")
                return {"status": "error", "message": "Enhanced mapping trống hoặc không thể tạo"}
        except Exception as e:
            logger.error(f"Lỗi khi lấy enhanced mapping: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi lấy enhanced mapping: {str(e)}"}

        # Gợi ý top-k món ăn
        logger.info("Đang tạo gợi ý món ăn")
        food_ids = recommend_for_user(user_id, model, graph, k=20)
        
        if not food_ids:
            logger.warning(f"Không có gợi ý nào cho user {user_id}")
            return {"status": "error", "message": f"Không tìm thấy đề xuất cho user {user_id}"}

        # Lấy thông tin người dùng
        logger.info("Đang lấy thông tin người dùng")
        user_node = get_user_node_info(user_id)
        
        if not user_node:
            logger.warning(f"Không tìm thấy thông tin cho user {user_id}")
            return {"status": "error", "message": f"Không tìm thấy thông tin cho user {user_id}"}

        # Mapping thông tin món ăn
        logger.info("Đang mapping thông tin món ăn")
        vn_foods = []
        vn_ingredients = []
        
        for food_id in food_ids:
            try:
                food_id_str = str(int(food_id))  # Đảm bảo là Python string
                if food_id_str in data:
                    temp = data[food_id_str]
                    vn_foods.append(str(temp[0]))  # Đảm bảo là Python string
                    vn_ingredients.append(str(temp[1]))  # Đảm bảo là Python string
                else:
                    vn_foods.append(f'Unknown Food ({food_id_str})')
                    vn_ingredients.append('No ingredients found')
            except Exception as e:
                logger.error(f"Error mapping food {food_id}: {str(e)}")
                vn_foods.append(f'Error Food ({food_id})')
                vn_ingredients.append('Error Ingredients')

        # Trả kết quả - Xử lý các giá trị float để đảm bảo JSON compliance
        logger.info("Đang trả kết quả gợi ý")
        
        # Xử lý prompt để đảm bảo là string
        user_prompt = user_node.get('prompt', 'Unknown user')
        if not isinstance(user_prompt, str):
            user_prompt = str(user_prompt)
            
        # Tạo kết quả JSON và đảm bảo không có giá trị invalid float
        result = {
            "status": "success",
            'user_info': user_prompt,
            'recommendations': []
        }
        
        for name, ingredients in zip(vn_foods, vn_ingredients):
            # Đảm bảo name và ingredients là chuỗi
            if not isinstance(name, str):
                name = str(name)
            if not isinstance(ingredients, str):
                ingredients = str(ingredients)
                
            result['recommendations'].append({
                "name": name, 
                "ingredients": ingredients
            })
        
        logger.info(f"Đã tạo gợi ý thành công cho user_id: {user_id}")
        
        # Đảm bảo kết quả cuối cùng không chứa kiểu dữ liệu NumPy hoặc PyTorch
        result = clean_float_values(result)
        return convert_to_python_native(result)

    except Exception as e:
        logger.error(f"Lỗi trong endpoint recommendation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error", 
            "message": "Đã xảy ra lỗi khi xử lý yêu cầu", 
            "detail": str(e)
        }


def generate_nutrition_tags_with_gemini(
    symptoms=None, 
    special_diet=None, 
    diseases=None,
    api_key=None
):
    """
    Generate nutrition tags using Google's Gemini model based on user health information.
    
    Args:
        symptoms: List of user symptoms
        special_diet: List of user special diets
        diseases: List of user diseases/conditions
        api_key: Gemini API key (can be set via GEMINI_API_KEY env var)
    
    Returns:
        List[int]: List of 14 binary values (0 or 1) for nutrition tags in this order:
        [low_calorie, high_calorie, low_carb, high_carb, low_protein, high_protein,
         low_sugar, high_sugar, low_saturated_fat, high_saturated_fat,
         low_cholesterol, high_cholesterol, low_sodium, high_sodium]
    """
    # Get API key from parameter or environment variable
    api_key = api_key or "AIzaSyDHRsA1G42JsicRxJFMMZZ9chcwLxDoVZU"
    if not api_key:
        logger.error("No Gemini API key provided. Please set GEMINI_API_KEY environment variable or pass as parameter.")
        return [0] * 14  # Return all zeros if no API key

    # Format health information for the prompt
    symptoms_text = "None" if not symptoms else ", ".join(symptoms)
    special_diet_text = "None" if not special_diet else ", ".join(special_diet)
    diseases_text = "None" if not diseases else ", ".join(diseases)
    
    prompt = f"""
You are a nutrition expert system. Analyze the following user health information and determine appropriate nutrition tags.

HEALTH INFORMATION:
- Symptoms: {symptoms_text}
- Special Diets: {special_diet_text}
- Medical Conditions: {diseases_text}

Based on this information, generate nutritional tags for the user. For each tag, determine if it should be set (1) or not set (0).
You must only respond with a JSON format containing the exact 14 tags in this order:
[low_calorie, high_calorie, low_carb, high_carb, low_protein, high_protein, low_sugar, high_sugar, low_saturated_fat, high_saturated_fat, low_cholesterol, high_cholesterol, low_sodium, high_sodium]

Rules:
1. If contradictory information exists (e.g., both weight gain and weight loss needs), use medical conditions as the priority.
2. For diabetic patients, always set low_sugar and low_carb.
3. For hypertension or heart conditions, set low_sodium and low_saturated_fat.
4. For kidney disease, set low_protein and low_sodium.
5. For underweight patients, set high_calorie and high_protein.
6. For overweight patients, set low_calorie and low_saturated_fat.

RESPOND ONLY WITH A JSON ARRAY of exactly 14 binary values (0 or 1).
"""

    try:
        # Use requests library to call Gemini API directly
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Error with Gemini API call: HTTP {response.status_code} - {response.text}")
            return [0] * 14
        
        response_data = response.json()
        
        # Extract text from the response
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            if "content" in response_data["candidates"][0] and "parts" in response_data["candidates"][0]["content"]:
                parts = response_data["candidates"][0]["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    response_text = parts[0]["text"]
                else:
                    logger.error("Failed to extract text from Gemini response")
                    return [0] * 14
            else:
                logger.error("Invalid response structure from Gemini API")
                return [0] * 14
        else:
            logger.error("No candidates in Gemini response")
            return [0] * 14
        
        # Extract just the array from the response (removing any extra text)
        array_match = re.search(r'\[\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*,\s*[01]\s*\]', response_text)
        
        if array_match:
            tags_json = array_match.group(0)
            tags = json.loads(tags_json)
            
            # Ensure we have exactly 14 tags
            if len(tags) != 14:
                logger.error(f"Unexpected number of tags from Gemini: {len(tags)}")
                return [0] * 14
                
            return tags
        else:
            logger.error(f"Could not extract tags array from Gemini response: {response_text}")
            return [0] * 14
            
    except Exception as e:
        logger.error(f"Error with Gemini API call: {str(e)}")
        return [0] * 14


def get_excluded_food_ids():
    """
    Reads the first 450 lines of us_to_vn_food_mapping_all.csv and returns a set of food IDs to exclude
    """
    try:
        # Try different potential paths for the file
        mapping_file = 'us_to_vn_food_mapping_all.csv'
        logger.info(f"Loading excluded food IDs from {mapping_file}")
        
        # Read only the first 450 rows
        df_mapping = pd.read_csv(mapping_file, nrows=450)
        
        # Extract the us_food_id column and convert to a set for faster lookups
        excluded_ids = set(df_mapping['us_food_id'].astype(int).tolist())
        
        logger.info(f"Loaded {len(excluded_ids)} excluded food IDs")
        return excluded_ids
    
    except Exception as e:
        logger.error(f"Error loading excluded food IDs: {str(e)}")
        return set()  # Return empty set on error
    
    
def filter_foods_with_gemini(food_list, ingredients_list, api_key=None):
    """
    Use Gemini to filter out snacks and sweet foods from recommendations
    
    Args:
        food_list: List of food names
        ingredients_list: List of ingredients for each food
        api_key: Gemini API key
        
    Returns:
        List of booleans indicating whether each food should be kept (True) or filtered out (False)
    """
    # If no food list, return empty result
    if not food_list:
        return []
        
    # Use default API key if not provided
    api_key = api_key or "AIzaSyDHRsA1G42JsicRxJFMMZZ9chcwLxDoVZU"
    
    try:
        # Create a batch of foods to analyze (to reduce API calls)
        results = []
        batch_size = 5  # Process 5 foods at a time
        
        for i in range(0, len(food_list), batch_size):
            batch_foods = food_list[i:i+batch_size]
            batch_ingredients = ingredients_list[i:i+batch_size]
            
            # Construct prompt for the batch
            foods_text = ""
            for j, (food, ingredient) in enumerate(zip(batch_foods, batch_ingredients)):
                foods_text += f"Food {j+1}: {food}\nIngredients {j+1}: {ingredient}\n\n"
            
            prompt = f"""
As a nutrition expert, analyze these foods and identify if they are snacks or sweet foods that should be excluded from a healthy diet.

{foods_text}

For each food, determine if it should be KEPT (healthy option) or FILTERED (unhealthy snack or sweet food).
Categorize as FILTERED if:
1. It's a snack food (chips, crackers, etc.)
2. It's a dessert or sweet treat (candy, cake, cookies, ice cream, etc.)
3. It has high sugar content
4. It's a sugary beverage

Respond with ONLY a JSON array of Boolean values (true to keep, false to filter):
[true/false, true/false, ...] - one value for each food, in the same order as provided.
"""
            try:
                # Use requests library to call Gemini API directly
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                }
                
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"Error with Gemini API call: HTTP {response.status_code} - {response.text}")
                    # Keep all foods in this batch if there's an error
                    results.extend([True] * len(batch_foods))
                    continue
                
                response_data = response.json()
                
                # Extract text from the response
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    if "content" in response_data["candidates"][0] and "parts" in response_data["candidates"][0]["content"]:
                        parts = response_data["candidates"][0]["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            response_text = parts[0]["text"]
                        else:
                            logger.error("Failed to extract text from Gemini response")
                            results.extend([True] * len(batch_foods))
                            continue
                    else:
                        logger.error("Invalid response structure from Gemini API")
                        results.extend([True] * len(batch_foods))
                        continue
                else:
                    logger.error("No candidates in Gemini response")
                    results.extend([True] * len(batch_foods))
                    continue
                
                # Extract the JSON array
                import re
                json_match = re.search(r'\[\s*(true|false)(\s*,\s*(true|false))*\s*\]', response_text, re.IGNORECASE)
                
                if json_match:
                    json_array = json_match.group(0).lower()
                    # Replace JavaScript true/false with Python True/False
                    json_array = json_array.replace('true', 'True').replace('false', 'False')
                    batch_results = eval(json_array)
                    results.extend(batch_results)
                else:
                    # Fallback if parsing fails
                    logger.warning(f"Failed to parse Gemini response: {response_text}")
                    # Keep all foods in this batch as fallback
                    results.extend([True] * len(batch_foods))
            except Exception as e:
                logger.error(f"Error in Gemini filtering: {str(e)}")
                # Keep all foods in this batch if there's an error
                results.extend([True] * len(batch_foods))
                
        # Ensure we have one result for each food
        if len(results) < len(food_list):
            results.extend([True] * (len(food_list) - len(results)))
        
        return results[:len(food_list)]
        
    except Exception as e:
        logger.error(f"Failed to filter foods with Gemini: {str(e)}")
        # Default to keeping all foods if the process fails
        return [True] * len(food_list)


def filter_foods_without_gemini(food_list, ingredients_list):
    """
    Fallback function to filter foods without using Gemini API
    Uses basic keyword matching to identify snacks and sweet foods
    """
    keep_flags = []
    
    # Keywords that might indicate snacks or sweet foods
    sweet_keywords = [
        'cake', 'cookie', 'pie', 'ice cream', 'candy', 'chocolate', 'sugar', 'sweet', 
        'dessert', 'pastry', 'donut', 'doughnut', 'brownie', 'cupcake', 'syrup',
        'caramel', 'frosting', 'icing', 'jelly', 'jam', 'honey', 'pudding',
        'bánh ngọt', 'kẹo', 'sô cô la', 'đường', 'ngọt', 'tráng miệng', 'bánh rán',
        'kem', 'xi-rô', 'mứt', 'mật ong', "bánh", "pancake"
    ]
    
    snack_keywords = [
        'chip', 'crisp', 'cracker', 'popcorn', 'pretzel', 'snack', 'bar', 
        'puff', 'mix', 'nut mix', 'trail mix', 'jerky', 'candy', 'gum',
        'bánh snack', 'bim bim', 'bỏng ngô', 'bánh quy', 'đồ ăn vặt', 'hạt'
    ]
    
    for food, ingredients in zip(food_list, ingredients_list):
        food_lower = food.lower()
        ingredients_lower = ingredients.lower()
        
        # Check for sweet foods
        is_sweet = any(keyword in food_lower or keyword in ingredients_lower for keyword in sweet_keywords)
        
        # Check for snack foods
        is_snack = any(keyword in food_lower or keyword in ingredients_lower for keyword in snack_keywords)
        
        # Keep the food if it's neither a sweet food nor a snack
        keep_flags.append(not (is_sweet or is_snack))
    
    return keep_flags

# Thêm vào phần khai báo endpoint
@app.post("/recommend_for_new_user")
def get_recommendation_for_new_user(input: NewUserInput):
    try:
        logger.info(f"Đang xử lý yêu cầu gợi ý cho user mới: {input}")
        
        # Lazy loading graph
        try:
            graph = get_graph()
        except Exception as e:
            logger.error(f"Lỗi khi tải graph: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi tải graph: {str(e)}"}
        
        # Chuyển input từ chữ sang số
        user_features = input.dict()
        
        # Kiểm tra xem người dùng có ăn chay không chỉ dựa vào từ khóa
        is_vegan = False
        
        # Danh sách từ khóa đầy đủ để xác định ăn chay
        vegan_keywords = [
            # Tiếng Việt
            "chay", "ăn chay", "thuần chay", "đồ chay", "món chay", "thực đơn chay", 
            "không ăn thịt", "không ăn cá", "không ăn hải sản", "không ăn trứng", "không uống sữa",
            "không sử dụng thịt", "không sử dụng cá", "không sử dụng sữa", "không sử dụng trứng",
            "kiêng thịt", "kiêng cá", "kiêng trứng", "kiêng sữa", "ăn kiêng đạm động vật",
            "không dùng thịt", "không dùng cá", "không dùng trứng", "không dùng sữa",
            
            # Tiếng Anh 
            "vegan", "vegetarian", "plant-based", "plant based", "no meat", "no fish",
            "no seafood", "no egg", "no milk", "no dairy", "meat-free", "fish-free",
            "dairy-free", "egg-free", "lacto-vegetarian", "lacto vegetarian",
            "ovo-vegetarian", "ovo vegetarian", "vegetable-based", "vegetable based"
        ]
        
        # Chỉ kiểm tra từ khóa nếu có thông tin trong spefical_diet
        if input.spefical_diet and len(input.spefical_diet) > 0:
            logger.info("Kiểm tra từ khóa ăn chay trong spefical_diet")
            special_diet_str = " ".join(input.spefical_diet).lower()
            is_vegan = any(keyword in special_diet_str for keyword in vegan_keywords)
            logger.info(f"Phát hiện người dùng ăn chay từ từ khóa: {is_vegan}")
        else:
            logger.info("Không có thông tin spefical_diet, không phải người ăn chay")
            is_vegan = False
        
        # Generate nutrition tags using Gemini if health information is provided
        if input.symptom or input.spefical_diet or input.disease:
            logger.info("Generating nutrition tags using Gemini based on health information")
            try:
                tags = generate_nutrition_tags_with_gemini(
                    symptoms=input.symptom,
                    special_diet=input.spefical_diet,
                    diseases=input.disease
                )
                # Add tags to user features
                user_features['tags'] = tags
                logger.info(f"Generated tags with Gemini: {tags}")
            except Exception as tag_error:
                logger.error(f"Error generating tags with Gemini: {str(tag_error)}")
                logger.info("Continuing without tags")
        
        new_user_features = convert_user_input_to_numeric(user_features)
        
        # Tải danh sách món chay từ file
        vegan_dishes = set()
        if is_vegan:
            logger.info("Đang tải danh sách món chay từ file")
            try:
                vegan_file_paths = [
                    os.path.join(BASE_DIR, 'vegan_dishes.txt'),
                    os.path.join(parent_dir, 'vegan_dishes.txt'),
                    'vegan_dishes.txt'
                ]
                
                vegan_file = None
                for path in vegan_file_paths:
                    if os.path.exists(path):
                        vegan_file = path
                        break
                
                if vegan_file:
                    with open(vegan_file, 'r', encoding='utf-8') as f:
                        vegan_dishes = set(line.strip() for line in f if line.strip())
                    logger.info(f"Đã đọc {len(vegan_dishes)} món chay từ file {vegan_file}")
                else:
                    logger.error("Không tìm thấy file vegan_dishes.txt")
            except Exception as e:
                logger.error(f"Lỗi khi đọc file vegan_dishes.txt: {str(e)}")
        
        # Tìm các user tương tự
        logger.info("Đang tìm các user tương tự")
        similar_users = find_similar_users(
            new_user_features, 
            graph, 
            top_k=int(new_user_features.get("top_k", 5)),
            similarity_threshold=float(new_user_features.get("similarity_threshold", 0.3))
        )
        
        if not similar_users:
            logger.warning("Không tìm thấy user tương tự")
            return {"status": "error", "message": "Không tìm thấy user tương tự"}
        
        # Lấy khuyến nghị từ user tương tự nhất
        most_similar_user_id = int(similar_users[0]['user_id'])
        logger.info(f"Đang tạo khuyến nghị từ user tương tự nhất: {most_similar_user_id}")
        
        # Lazy loading model
        try:
            model = get_model()
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi tải model: {str(e)}"}
        
        # Load excluded food IDs
        excluded_food_ids = get_excluded_food_ids()
        
        # Request nhiều món ăn để có nhiều lựa chọn hơn
        # Yêu cầu số lượng lớn các món để có thể trả về tất cả
        num_recommendations = 1000  # Yêu cầu nhiều hơn để có đủ món sau khi lọc
        food_indices = recommend_for_user(most_similar_user_id, model, graph, k=num_recommendations, excluded_food_ids=excluded_food_ids)
        
        if not food_indices:
            logger.warning(f"Không có gợi ý nào cho user tương tự {most_similar_user_id}")
            return {"status": "error", "message": "Không tìm thấy đề xuất cho user tương tự"}
        
        # ===== THAY ĐỔI: ĐỌC TRỰC TIẾP TỪ FOOD_TAGGING.CSV =====
        logger.info("Đang đọc thông tin món ăn trực tiếp từ food_tagging.csv")
        
        # Tìm đường dẫn file food_tagging.csv
        food_tagging_paths = [
            os.path.join(BASE_DIR, 'food_tagging.csv'),
            os.path.join(parent_dir, 'food_tagging.csv'),
            'food_tagging.csv',
            os.path.join(BASE_DIR, 'food_tagging_filter.csv'),
            os.path.join(parent_dir, 'food_tagging_filter.csv')
        ]
        
        food_tagging_file = None
        for path in food_tagging_paths:
            if os.path.exists(path):
                food_tagging_file = path
                break
        
        if not food_tagging_file:
            logger.error("Không tìm thấy file food_tagging.csv")
            return {"status": "error", "message": "Không tìm thấy file food_tagging.csv"}
        
        # Đọc file food_tagging.csv
        try:
            df_food = pd.read_csv(food_tagging_file)
            logger.info(f"Đã đọc {len(df_food)} món ăn từ {food_tagging_file}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {food_tagging_file}: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi đọc file {food_tagging_file}: {str(e)}"}
        
        # Lấy tên và thông tin món ăn từ indices
        recommended_foods = []
        max_score = len(food_indices)
        
        for i, food_idx in enumerate(food_indices):
            try:
                # Chuyển từ food index sang vị trí trong DataFrame
                if food_idx < len(df_food):
                    food_row = df_food.iloc[food_idx]
                    food_name = food_row['Tên món ăn']
                    
                    # Kiểm tra món chay
                    is_food_vegan = False
                    if vegan_dishes and food_name in vegan_dishes:
                        is_food_vegan = True
                    else:
                        # Danh sách từ khóa để xác định món KHÔNG phải món chay
                        meat_keywords = [
                            # Thịt và các loại thịt
                            "thịt", "gà", "heo", "bò", "cá", "tôm", "thịt heo", "thịt bò", "thịt gà", 
                            "thịt cừu", "thịt dê", "thịt vịt", "thịt ngan", "thịt ngỗng", "thịt chim",
                            "thịt thỏ", "thịt đà điểu", "thịt ngựa", "thịt trâu", "thịt nai", "thịt hươu",
                            "thịt nhím", "thịt lợn", "thịt bê", "jambon", "giăm bông", "chả", "giò", "xúc xích",
                            "lạp xưởng", "patê", "thịt xông khói", "thịt hun khói", "bacon", "ham", "salami",
                            
                            # Hải sản
                            "cá", "tôm", "cua", "ghẹ", "sò", "ốc", "hàu", "vẹm", "ngao", "nghêu", "sứa", 
                            "mực", "bạch tuộc", "cá hồi", "cá ngừ", "cá thu", "cá chép", "cá rô", "cá lóc",
                            "cá trê", "cá trắm", "cá điêu hồng", "cá chim", "cá lăng", "cá kèo", "cá diêu hồng",
                            "tôm hùm", "tôm càng", "tôm sú", "tôm thẻ", "tôm hùm đất", "tôm hùm biển",
                            
                            # Trứng
                            "trứng", "trứng gà", "trứng vịt", "trứng cút", "trứng ngỗng", "trứng đà điểu",
                            "lòng đỏ trứng", "lòng trắng trứng", "trứng chiên", "trứng luộc", "trứng ốp la",
                            
                            # Sữa và các sản phẩm từ sữa
                            "sữa", "phô mai", "pho mát", "bơ", "cream", "sữa chua", "váng sữa", "sữa đặc",
                            "sữa tươi", "sữa bò", "sữa dê", "sữa cừu", "cheese", "bơ sữa", "kem", "yaourt",
                            "sữa chua uống", "whipping cream", "kem tươi", "kem béo", "sữa đặc có đường",
                            
                            # Các từ khóa tiếng Anh
                            "meat", "beef", "pork", "chicken", "fish", "seafood", "shrimp", "crab", "egg",
                            "milk", "dairy", "cheese", "butter", "cream", "yogurt", "turkey", "duck", "goose",
                            "lamb", "bacon", "ham", "sausage", "hamburger", "steak"
                        ]
                        
                        # Kiểm tra từ khóa trong tên món ăn
                        food_name_lower = str(food_name).lower() if not isinstance(food_name, float) else ""
                        
                        # Lấy nguyên liệu nếu có
                        ingredients = ""
                        if 'Nguyên liệu' in food_row and pd.notna(food_row['Nguyên liệu']):
                            ingredients = str(food_row['Nguyên liệu']).lower()
                        
                        is_food_vegan = not any(keyword in food_name_lower or keyword in ingredients for keyword in meat_keywords)
                    
                    # Tính điểm cho món ăn (món đầu có điểm cao nhất)
                    score = max_score - i
                    
                    # Bỏ qua món có sữa, bánh (tuỳ chọn)
                    food_name_lower = str(food_name).lower() if not isinstance(food_name, float) else ""
                    ingredients = ""
                    if 'Nguyên liệu' in food_row and pd.notna(food_row['Nguyên liệu']):
                        ingredients = str(food_row['Nguyên liệu']).lower()
                    
                    if "milk" in food_name_lower or "milk" in ingredients or "sữa" in food_name_lower or "sữa" in ingredients:
                        continue
                    
                    # Tạo đối tượng món ăn
                    food_item = {
                        'name': food_name,
                        'ingredients': ingredients if ingredients else "Không có thông tin nguyên liệu",
                        'score': score,
                        'vegan': is_food_vegan,
                        'index': int(food_idx)
                    }
                    
                    # Thêm các thông tin dinh dưỡng nếu có
                    for col in ['Carbohydrate','Calories','Protein','Sugar','Fiber dietary',
                               'Vitamin C','Vitamin D','Vitamin B12','Calcium','Iron',
                               'Cholesterol','Phosphorous','Folic acid','Saturated fat',
                               'Potassium','Sodium']:
                        if col in food_row and pd.notna(food_row[col]):
                            food_item[col] = float(food_row[col])
                    
                    # Thêm các thông tin khác nếu có
                    for col in ['Sơ chế', 'Thực hiện', 'Cách dùng', 'Mách nhỏ', 'Thực đơn', 'Lời khuyên']:
                        if col in food_row and pd.notna(food_row[col]):
                            food_item[col] = str(food_row[col])
                    
                    recommended_foods.append(food_item)
            except Exception as e:
                logger.error(f"Error processing food index {food_idx}: {str(e)}")
                continue
        
        # Nếu người dùng ăn chay, lọc ra chỉ các món chay
        if is_vegan:
            logger.info("Lọc món ăn cho người dùng ăn chay")
            
            # Kiểm tra trong vegan_dishes.txt, nếu không tìm thấy món đó thì không chọn
            if vegan_dishes:  # Chỉ lọc nếu đã tải được danh sách món chay
                logger.info(f"Lọc theo danh sách chay từ vegan_dishes.txt ({len(vegan_dishes)} món)")
                vegan_foods = [food for food in recommended_foods if food['name'] in vegan_dishes]
                if vegan_foods:  # Nếu có món chay được đề xuất từ danh sách
                    recommended_foods = vegan_foods
                    logger.info(f"Đã lọc được {len(recommended_foods)} món chay từ danh sách vegan_dishes.txt")
                else:
                    # Nếu không tìm thấy món nào trong danh sách, dùng biện pháp dự phòng với món được đánh dấu là vegan
                    backup_vegan_foods = [food for food in recommended_foods if food['vegan']]
                    if backup_vegan_foods:
                        recommended_foods = backup_vegan_foods
                        logger.info(f"Không tìm thấy món nào trong vegan_dishes.txt, sử dụng {len(backup_vegan_foods)} món được đánh dấu là chay")
                    else:
                        logger.warning("Không có món chay nào được đề xuất, giữ nguyên danh sách")
            else:
                # Nếu không có danh sách món chay, dùng biện pháp dự phòng
                backup_vegan_foods = [food for food in recommended_foods if food['vegan']]
                if backup_vegan_foods:
                    recommended_foods = backup_vegan_foods
                    logger.info(f"Không tìm thấy file vegan_dishes.txt, sử dụng {len(backup_vegan_foods)} món được đánh dấu là chay")
                else:
                    logger.warning("Không có món chay nào được đề xuất, giữ nguyên danh sách")
        
        # Giới hạn số lượng món ăn trả về tối đa là 200
        if len(recommended_foods) > 200:
            logger.info(f"Giới hạn kết quả từ {len(recommended_foods)} xuống 200 món ăn")
            recommended_foods = recommended_foods[:200]
        
        # Trả kết quả - đảm bảo tất cả đều là kiểu dữ liệu Python tiêu chuẩn
        result = {
            "status": "success",
            "similar_users": [
                {
                    "user_id": int(user['user_id']),
                    "similarity": float(round(user['similarity'], 4)),
                    "details": convert_to_python_native(user.get('details', {}))
                } for user in similar_users
            ],
            "most_similar_user": {
                "user_id": int(similar_users[0]['user_id']),
                "similarity": float(round(similar_users[0]['similarity'], 4)),
                "details": convert_to_python_native(similar_users[0].get('details', {}))
            },
            "generated_tags": user_features.get('tags'),
            "health_info": {
                "symptoms": input.symptom,
                "special_diet": input.spefical_diet,
                "diseases": input.disease
            },
            "is_vegan": is_vegan,
            "recommendations": recommended_foods
        }
        
        logger.info(f"Đã tạo gợi ý thành công cho user mới dựa trên user tương tự {most_similar_user_id}")
        # Đảm bảo kết quả cuối cùng không chứa kiểu dữ liệu NumPy hoặc PyTorch
        recommended_foods = clean_float_values(recommended_foods)
        
        return convert_to_python_native({
            "status": "success", 
            "recommendations": recommended_foods, 
            "total_recommendations": len(recommended_foods),
            "generated_tags": user_features.get('tags'),
            "health_info": {
                "symptoms": input.symptom,
                "special_diet": input.spefical_diet,
                "diseases": input.disease
            },
            "is_vegan": is_vegan
        })
        
    except Exception as e:
        logger.error(f"Lỗi trong endpoint recommendation_for_new_user: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error", 
            "message": "Đã xảy ra lỗi khi xử lý yêu cầu", 
            "detail": str(e)
        }


@app.get("/")
def read_root():
    return {"status": "success", "message": "Welcome to Food Recommendation API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    # Kiểm tra các thành phần cần thiết
    health_status = {"status": "healthy", "components": {}}
    
    # Kiểm tra PyTorch
    try:
        health_status["components"]["pytorch"] = {"status": "up", "version": torch.__version__}
    except:
        health_status["components"]["pytorch"] = {"status": "down", "error": "PyTorch not available"}
    
    # Kiểm tra file graph
    graph_file = None
    graph_paths = [
        os.path.join(parent_dir, 'vn_food_graph.pt'),
        os.path.join(BASE_DIR, 'vn_food_graph.pt')
    ]
    for path in graph_paths:
        if os.path.exists(path):
            graph_file = path
            break
    
    health_status["components"]["graph_file"] = {
        "status": "up" if graph_file else "down",
        "path": graph_file if graph_file else "Not found"
    }
    
    # Kiểm tra file model
    model_file = None
    model_paths = [
        os.path.join(BASE_DIR, 'vn_trained_model.pth'),
        os.path.join(parent_dir, 'vn_trained_model.pth')
    ]
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break
    
    health_status["components"]["model_file"] = {
        "status": "up" if model_file else "down",
        "path": model_file if model_file else "Not found"
    }
    
    # Kiểm tra food_id_name_mapping.json
    food_id_mapping_file = None
    mapping_paths = [
        os.path.join(parent_dir, 'food_id_name_mapping.json'),
        os.path.join(BASE_DIR, 'food_id_name_mapping.json'),
        'food_id_name_mapping.json'
    ]
    for path in mapping_paths:
        if os.path.exists(path):
            food_id_mapping_file = path
            break

    health_status["components"]["food_id_mapping_file"] = {
        "status": "up" if food_id_mapping_file else "down",
        "path": food_id_mapping_file if food_id_mapping_file else "Not found"
    }

    # Kiểm tra food_tagging.csv
    food_tagging_file = None
    tagging_paths = [
        os.path.join(parent_dir, 'food_tagging.csv'),
        os.path.join(BASE_DIR, 'food_tagging.csv'),
        'food_tagging.csv'
    ]
    for path in tagging_paths:
        if os.path.exists(path):
            food_tagging_file = path
            break

    health_status["components"]["food_tagging_file"] = {
        "status": "up" if food_tagging_file else "down",
        "path": food_tagging_file if food_tagging_file else "Not found"
    }

    # Kiểm tra enhanced mapping
    enhanced_mapping = get_enhanced_mapping()
    health_status["components"]["enhanced_mapping"] = {
        "status": "up" if enhanced_mapping else "down",
        "count": len(enhanced_mapping) if enhanced_mapping else 0
    }
    
    # Kiểm tra tổng thể
    if all(component["status"] == "up" for component in health_status["components"].values()):
        health_status["status"] = "healthy"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status

# For direct execution
if __name__ == "__main__":
    # Khởi tạo enhanced mapping khi ứng dụng bắt đầu
    get_enhanced_mapping()
    port = 8000
    host = "0.0.0.0"
    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)