import chromadb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from .config import Config
import os
import torch

class DBManager:
    def __init__(self):
        os.makedirs(Config.DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=Config.DB_PATH)
        
        # 1. 论文集合
        self.paper_collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"📡 正在加载文本模型: {Config.TEXT_MODEL_PATH} ...")
        self.text_model = SentenceTransformer(Config.TEXT_MODEL_PATH, trust_remote_code=True)

        # 2. 图片集合
        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"👁️ 正在加载 CLIP 模型 (Transformers原生版): {Config.CLIP_MODEL_PATH} ...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_PATH)
            self.clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL_PATH)
        except Exception as e:
            print(f"❌ CLIP 模型加载失败: {e}")
            raise e

    def _normalize(self, embedding):
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding)
        norm = embedding.norm(p=2, dim=-1, keepdim=True)
        return (embedding / norm).tolist()

    def add_paper_chunks(self, file_path, chunks, category):
        """
        存入带有页码的 chunks
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        texts = [c['text'] for c in chunks]
        if not texts:
            return
            
        embeddings_list = self.text_model.encode(texts).tolist()

        for i, chunk in enumerate(chunks):
            # ID 格式: 文件路径_页码
            chunk_id = f"{file_path}_p{chunk['page']}"
            ids.append(chunk_id)
            embeddings.append(embeddings_list[i])
            documents.append(chunk['text'])
            metadatas.append({
                "source": file_path,
                "category": category,
                "page": chunk['page']
            })

        self.paper_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def add_image_embedding(self, file_path):
        try:
            image = Image.open(file_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            embedding = self._normalize(image_features[0])
            self.image_collection.upsert(
                ids=[file_path],
                embeddings=[embedding],
                metadatas=[{"source": file_path}]
            )
            return True
        except Exception as e:
            print(f"❌ 图片处理错误 {file_path}: {e}")
            return False

    def search_papers(self, query, n_results=3):
        query_embedding = self.text_model.encode(query).tolist()
        results = self.paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def search_images(self, text_query, n_results=3):
        inputs = self.clip_processor(text=[text_query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        embedding = self._normalize(text_features[0])
        results = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results