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


# Thêm sau phần import hoặc các hàm tiện ích khác
# Biến toàn cục để lưu trữ graph và model đã tải
_graph = None
_model = None

def get_graph():
    """Hàm lazy loading cho graph"""
    global _graph
    if _graph is None:
        graph_paths = [
            os.path.join(parent_dir, 'processed_data/benchmark_macro.pt'),
            os.path.join(BASE_DIR, '../processed_data/benchmark_macro.pt')
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
            os.path.join(BASE_DIR, 'trained_model.pth'),
            os.path.join(parent_dir, 'trained_model.pth')
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
    gender: int  # 1: nam, 2: nữ
    age_group: Optional[int] = None  # 1-7
    race: Optional[int] = None  # 0-5
    household_income: Optional[int] = None  # 0-11
    education: Optional[int] = None  # 0-9
    tags: Optional[List[int]] = None  # Danh sách tags sức khỏe/dinh dưỡng
    similarity_threshold: Optional[float] = 0.3  # Ngưỡng tương đồng tối thiểu
    top_k: Optional[int] = 5  # Số lượng user tương tự muốn trả về


# Import các function đã định nghĩa
def recommend_for_user(user_id, model, graph, k=20):
    try:
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
        
        # Lấy k thực phẩm có điểm cao nhất
        _, indices = torch.topk(scores, k)
        indices = indices.cpu().numpy()
        
        # Chuyển indices thành food_ids
        recommended_food_ids = [graph['food'].node_id[idx].item() for idx in indices]
        
        return recommended_food_ids
    except Exception as e:
        logger.error(f"Lỗi trong hàm recommend_for_user: {str(e)}")
        raise Exception(f"Lỗi trong hàm recommend_for_user: {str(e)}")

def get_user_node_info(user_id):
    # Tìm index của user_id trong graph
    graph_path = os.path.join(parent_dir, 'processed_data/benchmark_macro.pt')
    
    logger.info(f"Đang tìm file graph tại: {graph_path}")
    if not os.path.exists(graph_path):
        alt_path = os.path.join(BASE_DIR, '../processed_data/benchmark_macro.pt')
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
   

def food_mapping_function(us_food_id):
    # mapping_paths = [
    #     os.path.join(parent_dir, 'us_to_vn_food_mapping_all.csv'),
    #     os.path.join(BASE_DIR, 'us_to_vn_food_mapping_all.csv'),
    #     '/kaggle/working/code/us_to_vn_food_mapping_all.csv',
    #     os.path.join(BASE_DIR, 'code/us_to_vn_food_mapping_all.csv'),
    # ]
    
    # mapping_file = None
    # for path in mapping_paths:
    #     if os.path.exists(path):
    #         mapping_file = path
    #         break
    
    # if mapping_file is None:
    #     logger.error("Không tìm thấy file mapping")
    #     raise FileNotFoundError("Không tìm thấy file mapping")
    mapping_file = 'us_to_vn_food_mapping_all.csv'
    try:
        logger.info(f"Đang đọc file mapping từ {mapping_file}")
        df_mapping = pd.read_csv('us_to_vn_food_mapping_all.csv')
        df_mapping = df_mapping[df_mapping['us_food_id'] == int(us_food_id)]
        if(len(df_mapping) > 0):
            return df_mapping.iloc[0]
        return {}
    except Exception as e:
        logger.error(f"Lỗi khi đọc file mapping: {e}")
        raise Exception(f"Lỗi khi đọc file mapping: {e}")

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
            
        # Đọc file mapping
        try:
            with open('us_to_vn_food_simple_mapping.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file mapping: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi đọc file mapping: {str(e)}"}

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
        
        # Tạo đặc trưng cho user mới
        new_user_features = {
            'gender': int(input.gender)
        }
        
        # Thêm các đặc trưng không bắt buộc nếu có
        if input.age_group is not None:
            new_user_features['age_group'] = int(input.age_group)
        
        if input.race is not None:
            new_user_features['race'] = int(input.race)
            
        if input.household_income is not None:
            new_user_features['household_income'] = int(input.household_income)
            
        if input.education is not None:
            new_user_features['education'] = int(input.education)
            
        if input.tags is not None:
            new_user_features['tags'] = [int(tag) for tag in input.tags]  # Đảm bảo tất cả là Python ints
        
        # Tìm các user tương tự
        logger.info("Đang tìm các user tương tự")
        similar_users = find_similar_users(
            new_user_features, 
            graph, 
            top_k=int(input.top_k or 5),
            similarity_threshold=float(input.similarity_threshold or 0.3)
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
        
        # Lấy khuyến nghị cho user tương tự nhất
        food_ids = recommend_for_user(most_similar_user_id, model, graph, k=20)
        
        if not food_ids:
            logger.warning(f"Không có gợi ý nào cho user tương tự {most_similar_user_id}")
            return {"status": "error", "message": "Không tìm thấy đề xuất cho user tương tự"}
        
        # Mapping thông tin món ăn
        logger.info("Đang mapping thông tin món ăn")
        try:
            with open('us_to_vn_food_simple_mapping.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file mapping: {str(e)}")
            return {"status": "error", "message": f"Lỗi khi đọc file mapping: {str(e)}"}
            
        vn_foods = []
        vn_ingredients = []
        for food_id in food_ids:
            try:
                food_id_str = str(int(food_id))  # Đảm bảo chuyển đổi sang chuỗi Python tiêu chuẩn
                if food_id_str in data:
                    temp = data[food_id_str]
                    vn_foods.append(str(temp[0]))
                    vn_ingredients.append(str(temp[1]))
                else:
                    vn_foods.append(f'Unknown Food ({food_id_str})')
                    vn_ingredients.append('No ingredients found')
            except Exception as e:
                logger.error(f"Error mapping food {food_id}: {str(e)}")
                vn_foods.append(f'Error Food ({food_id})')
                vn_ingredients.append('Error Ingredients')
        
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
            "recommendations": []
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
        
        logger.info(f"Đã tạo gợi ý thành công cho user mới dựa trên user tương tự {most_similar_user_id}")
        
        # Đảm bảo kết quả cuối cùng không chứa kiểu dữ liệu NumPy hoặc PyTorch
        return convert_to_python_native(result)
        
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
        os.path.join(parent_dir, 'processed_data/benchmark_macro.pt'),
        os.path.join(BASE_DIR, '../processed_data/benchmark_macro.pt')
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
        os.path.join(BASE_DIR, 'trained_model.pth'),
        os.path.join(parent_dir, 'trained_model.pth')
    ]
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break
    
    health_status["components"]["model_file"] = {
        "status": "up" if model_file else "down",
        "path": model_file if model_file else "Not found"
    }
    
    # Kiểm tra file mapping
    mapping_file = None
    mapping_paths = [
        os.path.join(parent_dir, 'us_to_vn_food_mapping_all.csv'),
        os.path.join(BASE_DIR, '../us_to_vn_food_mapping_all.csv')
    ]
    for path in mapping_paths:
        if os.path.exists(path):
            mapping_file = path
            break
    
    health_status["components"]["mapping_file"] = {
        "status": "up" if mapping_file else "down",
        "path": mapping_file if mapping_file else "Not found"
    }
    
    # Kiểm tra tổng thể
    if all(component["status"] == "up" for component in health_status["components"].values()):
        health_status["status"] = "healthy"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status

# For direct execution
if __name__ == "__main__":
    with open('us_to_vn_food_simple_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)   
    port = 8000
    host = "0.0.0.0"
    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
