import os
import shutil
import pypdf

def extract_text_from_pdf(file_path):
    """
    返回一个列表，每个元素是字典: {'text': string, 'page': int}
    """
    chunks = []
    try:
        reader = pypdf.PdfReader(file_path)
        # 读取前 10 页 (可根据需要调整)
        num_pages = min(len(reader.pages), 10) 
        
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            if page_text and len(page_text) > 50: # 过滤掉太短的页
                chunks.append({
                    "text": page_text,
                    "page": i + 1  # 页码从 1 开始
                })
    except Exception as e:
        print(f"PDF 读取错误 {file_path}: {e}")
    return chunks

def move_file_to_category(file_path, category):
    base_dir = os.path.dirname(file_path)
    target_dir = os.path.join(base_dir, category)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    filename = os.path.basename(file_path)
    new_path = os.path.join(target_dir, filename)
    
    if os.path.exists(new_path):
        return new_path
        
    shutil.move(file_path, new_path)
    return new_path