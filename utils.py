import json
from models.attn_model import AttentionModel # <-- Chỉ cần import lớp này

def open_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'attn-hf':
        model = AttentionModel(config)
    else:
        # Xóa các trường hợp khác và báo lỗi nếu provider không được hỗ trợ
        raise ValueError(f"ERROR: Unknown or unsupported provider '{provider}'")
    return model