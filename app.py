import streamlit as st
import os
import time
from PIL import Image
from tqdm import tqdm

try:
    from src.vision_expert import VisionExpert
    from src.db_manager import DBManager
    from src.llm_client import LLMClient
    from src.file_handler import extract_text_from_pdf, move_file_to_category
except ImportError as e:
    st.error(f"❌ 导入模块失败: {e}")
    st.stop()

# ==========================================
# 1. 页面配置 & 状态初始化
# ==========================================
st.set_page_config(
    page_title="Local Multimodal Agent",
    page_icon="🤖",
    layout="wide"
)

# 使用 Session State 缓存模型和数据，避免刷新丢失
if 'agent_loaded' not in st.session_state:
    st.session_state.agent_loaded = False
if 'img_description' not in st.session_state:
    st.session_state.img_description = None

# ==========================================
# 2. 侧边栏：系统初始化
# ==========================================
st.sidebar.title("🤖 控制台")

# 模型加载函数
@st.cache_resource
def load_models():
    print("⏳ 正在初始化模型...")
    vision = VisionExpert()
    db = DBManager()
    llm = LLMClient()
    return vision, db, llm

# 加载模型
with st.sidebar:
    st.write("系统状态检测...")
    try:
        vision_expert, db_manager, llm_client = load_models()
        st.success("✅ 全系统模型已就绪")
        st.session_state.agent_loaded = True
        
        st.divider()
        st.info(f"📂 知识库路径: {os.path.abspath('./data/chroma_db')}")
        st.info(f"🖼️ 视觉模型: Florence-2-Large")
        st.info(f"🧠 推理模型: Qwen-2.5") 
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

# ==========================================
# 3. 主界面逻辑
# ==========================================
st.title("🧠 本地多模态智能体 (Local Agent)")
st.markdown("支持 **批量文献整理** | **语义检索** | **视觉理解**")

# 创建两个主要的功能选项卡
tab_knowledge, tab_vision = st.tabs(["📚 知识库专家 (Knowledge)", "👁️ 视觉专家 (Vision)"])

# --- TAB 1: 知识库功能 ---
with tab_knowledge:
    st.header("📚 智能文献管理")
    
    # 子功能选择
    k_mode = st.radio("选择操作:", ["批量入库与分类 (Batch Process)", "语义检索 (RAG Search)"], horizontal=True)
    
    if k_mode == "批量入库与分类 (Batch Process)":
        st.markdown("#### 📂 批量文档处理")
        st.info("该功能将扫描指定文件夹，自动识别论文主题，并将其**移动**到分类子文件夹中。")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # 默认指向你的 ./paper 目录
            target_dir = st.text_input("输入论文所在文件夹路径:", value="./paper")
            topics_str = st.text_input("设置分类标签 (用逗号分隔):", value="Computer Vision, NLP, Reinforcement Learning, Robotics")
        
        with col2:
            st.write("##") # 占位
            start_btn = st.button("🚀 开始批量整理", type="primary", use_container_width=True)
        
        if start_btn:
            if not os.path.exists(target_dir):
                st.error(f"❌ 路径不存在: {target_dir}")
            else:
                # === 核心批量处理逻辑 ===
                st.write(f"🔍 正在扫描 `{target_dir}` ...")
                
                # 收集 PDF
                pdf_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.pdf')]
                
                if not pdf_files:
                    st.warning("⚠️ 该目录下没有找到 PDF 文件。")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    log_area = st.container() # 用于显示日志
                    
                    processed_count = 0
                    
                    for i, file_path in enumerate(pdf_files):
                        filename = os.path.basename(file_path)
                        status_text.text(f"正在处理: {filename} ...")
                        
                        try:
                            # 1. 提取文本
                            chunks = extract_text_from_pdf(file_path)
                            if not chunks:
                                continue
                                
                            # 2. LLM 分类
                            first_page = chunks[0]['text']
                            category = llm_client.classify_paper(first_page, topics_str)
                            
                            # 3. 移动文件
                            new_path = move_file_to_category(file_path, category)
                            
                            # 4. 存入向量库
                            db_manager.add_paper_chunks(new_path, chunks, category)
                            
                            # 5. UI 反馈
                            with log_area:
                                st.success(f"✅ {filename} -> 📂 **{category}** (已入库)")
                            
                            processed_count += 1
                        except Exception as e:
                            st.error(f"处理 {filename} 失败: {e}")
                        
                        # 更新进度条
                        progress_bar.progress((i + 1) / len(pdf_files))
                    
                    status_text.text("🎉 处理完成！")
                    st.balloons()

    elif k_mode == "语义检索 (RAG Search)":
        st.markdown("#### 🧠 知识库问答")
        query = st.text_input("请输入学术问题:", placeholder="例如: Transformer 的自注意力机制是如何工作的？")
        
        if st.button("🔍 搜索并回答"):
            if query:
                with st.spinner("正在检索向量数据库并生成回答..."):
                    # 1. 检索
                    results = db_manager.search_papers(query, n_results=3)
                    
                    if not results['ids'][0]:
                        st.warning("📭 知识库中没有找到相关内容。")
                    else:
                        # 2. 构建上下文
                        context_str = ""
                        st.markdown("### 📄 参考来源")
                        for i in range(len(results['ids'][0])):
                            meta = results['metadatas'][0][i]
                            text = results['documents'][0][i]
                            score = 1 - results['distances'][0][i]
                            
                            with st.expander(f"来源 {i+1}: {os.path.basename(meta['source'])} (Page {meta['page']}) - 相关度 {score:.2f}"):
                                st.write(text)
                                st.caption(f"分类: {meta['category']}")
                            
                            context_str += f"文档: {meta['source']} (Page {meta['page']})\n内容: {text}\n\n"
                        
                        # 3. LLM 回答
                        st.markdown("### 🤖 AI 回答")
                        answer = llm_client.chat_with_context(query, context_str)
                        st.write(answer)

# --- TAB 2: 视觉功能 (核心修改区域) ---
with tab_vision:
    st.header("👁️ 视觉感知")
    v_mode = st.radio("功能:", ["图片描述 & 问答 (Caption & VQA)", "以文搜图 (Image Search)"], horizontal=True)
    
    if v_mode == "图片描述 & 问答 (Caption & VQA)":
        col_img, col_desc = st.columns([1, 1])
        
        with col_img:
            uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "webp", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="预览", use_container_width=True)
                
                # 保存临时文件供模型读取路径
                temp_path = f"temp_{uploaded_file.name}"
                image.save(temp_path)
                
                # 换图片时清空缓存
                if 'last_img' not in st.session_state or st.session_state.last_img != uploaded_file.name:
                    st.session_state.img_description = None
                    st.session_state.last_img = uploaded_file.name
        
        with col_desc:
            if uploaded_file:
                # 1. 深度描述模块
                st.markdown("#### 1. 深度描述")
                if st.button("📝 生成描述"):
                    with st.spinner("Florence-2 正在观察图片细节..."):
                        # 使用 MORE_DETAILED_CAPTION 生成最详细的文本
                        res = vision_expert.analyze_image(temp_path, prompt_type="<MORE_DETAILED_CAPTION>")
                        st.session_state.img_description = res # 存入缓存
                        st.success("分析完成")
                        st.info(res)
                
                st.divider()
                
                # 2. 视觉问答模块 (Visual RAG)
                st.markdown("#### 2. 视觉问答 (Visual RAG)")
                st.caption("🚀 升级版: 结合 Florence-2 的视觉能力与 LLM 的推理能力，支持中文！")
                
                user_q = st.text_input("问图片一个问题:", placeholder="这只羊是什么颜色的？/ What is this?")
                
                if st.button("❓ 提问"):
                    # 1. 确保有全局描述
                    if not st.session_state.img_description:
                        with st.spinner("👀 AI 正在阅读图片全局内容..."):
                            st.session_state.img_description = vision_expert.analyze_image(temp_path, prompt_type="<MORE_DETAILED_CAPTION>")
                    
                    if user_q:
                        with st.spinner("🧠 AI 正在搜集细节并思考..."):
                            dense_data = vision_expert.analyze_image(temp_path, prompt_type="<DENSE_REGION_CAPTION>")
                            
                            # 解析密集描述的数据 (它返回的是字典或者字符串)
                            dense_text = ""
                            if isinstance(dense_data, dict) and 'labels' in dense_data:
                                # 提取所有标签并去重
                                unique_labels = list(set(dense_data['labels']))
                                dense_text = ", ".join(unique_labels)
                            else:
                                dense_text = str(dense_data)

                            # === 构建超级详细的上下文 ===
                            context = f"""
                            [图片全局描述]:
                            {st.session_state.img_description}
                            
                            [图片局部细节/物体标签]:
                            {dense_text}
                            """
                            
                            answer = llm_client.chat_with_context(user_q, context)
                            
                            st.markdown("### 🤖 回答:")
                            st.success(answer)
                            
                            with st.expander("查看 AI 看到的完整视觉信息"):
                                st.text(context)
                        
    elif v_mode == "以文搜图 (Image Search)":
        st.markdown("#### 🔍 本地图片库搜索")
        
        # 索引构建工具
        with st.expander("⚙️ 索引管理 (如果搜不到图，请先点这里)"):
            img_dir = st.text_input("图片文件夹路径:", value="./images")
            if st.button("🔄 重建图片索引"):
                count = 0
                image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                progress = st.progress(0)
                files = []
                for root, _, fs in os.walk(img_dir):
                    for f in fs:
                        if os.path.splitext(f)[1].lower() in image_exts:
                            files.append(os.path.join(root, f))
                
                for i, path in enumerate(files):
                    db_manager.add_image_embedding(path)
                    progress.progress((i+1)/len(files))
                    count += 1
                st.success(f"已索引 {count} 张图片！")

        # 搜索界面
        search_q = st.text_input("描述你要找的画面:", placeholder="一只在睡觉的猫")
        if st.button("🖼️ 搜索图片"):
            if search_q:
                results = db_manager.search_images(search_q)
                if not results['ids'][0]:
                    st.warning("未找到匹配图片。")
                else:
                    cols = st.columns(3)
                    for i in range(len(results['ids'][0])):
                        img_path = results['ids'][0][i]
                        score = 1 - results['distances'][0][i]
                        if os.path.exists(img_path):
                            cols[i % 3].image(img_path, caption=f"匹配度: {score:.2f}")
                            cols[i % 3].caption(os.path.basename(img_path))