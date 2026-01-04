import os
from dotenv import load_dotenv

load_dotenv()

# 1. 定义默认值
text_model_path = "all-MiniLM-L6-v2" 
clip_model_path = "openai/clip-vit-base-patch32"
vision_model_path = "microsoft/Florence-2-large"

# 2. 尝试读取本地路径文件
path_file = "model_paths.txt"
if os.path.exists(path_file):
    try:
        with open(path_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TEXT_MODEL_PATH="):
                    text_model_path = line.split("=")[1]
                elif line.startswith("CLIP_MODEL_PATH="):
                    clip_model_path = line.split("=")[1]
                elif line.startswith("VISION_MODEL_PATH="):
                    vision_model_path = line.split("=")[1]
        print(f"✅ 已加载本地模型路径配置")
    except Exception as e:
        print(f"⚠️ 读取路径文件失败，将使用默认配置: {e}")

class Config:
    # 硅基流动配置
    API_KEY = os.getenv("SILICON_FLOW_API_KEY")
    BASE_URL = "https://api.siliconflow.cn/v1"
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    
    # 数据库路径
    DB_PATH = "./data/chroma_db"
    
    # 模型路径
    TEXT_MODEL_PATH = text_model_path
    CLIP_MODEL_PATH = clip_model_path
    VISION_MODEL_PATH = vision_model_path