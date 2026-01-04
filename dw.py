import os
import sys

# 检查 modelscope 是否安装
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("❌ 请先安装 modelscope: pip install modelscope")
    sys.exit(1)

# 创建保存模型的目录
os.makedirs("./models", exist_ok=True)
paths_to_save = {}

print("🚀 [1/3] 正在下载文本嵌入模型 (all-MiniLM-L6-v2)...")
try:
    text_model_dir = snapshot_download(
        'AI-ModelScope/all-MiniLM-L6-v2', 
        cache_dir='./models', 
        revision='master'
    )
    print(f"✅ 文本模型下载成功: {text_model_dir}")
    paths_to_save["TEXT_MODEL_PATH"] = text_model_dir
except Exception as e:
    print(f"❌ 文本模型下载失败: {e}")

print("\n🚀 [2/3] 正在下载 CLIP 图像模型 (clip-vit-base-patch32)...")
try:
    clip_model_dir = snapshot_download(
        'openai-mirror/clip-vit-base-patch32', 
        cache_dir='./models', 
        revision='master'
    )
    print(f"✅ CLIP 模型下载成功: {clip_model_dir}")
    paths_to_save["CLIP_MODEL_PATH"] = clip_model_dir
except Exception as e:
    print(f"❌ CLIP 模型下载失败: {e}")

print("\n🚀 [3/3] 正在下载 Florence-2 视觉大模型 (Florence-2-large)...")
try:
    florence_model_dir = snapshot_download(
        'AI-ModelScope/Florence-2-large', 
        cache_dir='./models', 
        revision='master'
    )
    print(f"✅ Florence-2 下载成功: {florence_model_dir}")
    paths_to_save["VISION_MODEL_PATH"] = florence_model_dir
except Exception as e:
    print(f"❌ Florence-2 下载失败: {e}")

# 将路径写入配置文件
with open("model_paths.txt", "w") as f:
    for key, value in paths_to_save.items():
        f.write(f"{key}={value}\n")

print("\n🎉 所有模型下载流程结束！请检查上方是否有报错。")