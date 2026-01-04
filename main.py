import argparse
import os
from tqdm import tqdm
from src.db_manager import DBManager
from src.llm_client import LLMClient
from src.vision_expert import VisionExpert
from src.file_handler import extract_text_from_pdf, move_file_to_category

def main():
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent (Ultimate Version)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: add_paper
    add_parser = subparsers.add_parser("add_paper", help="Add and classify papers with page-level indexing")
    add_parser.add_argument("path", help="Path to the PDF file or directory")
    add_parser.add_argument("--topics", required=True, help="Comma separated topics")

    # Command: search_paper (Advanced RAG)
    search_parser = subparsers.add_parser("search_paper", help="Semantic search & Q/A")
    search_parser.add_argument("query", help="Question about papers")

    # Command: scan_images
    scan_img_parser = subparsers.add_parser("scan_images", help="Index all images")
    scan_img_parser.add_argument("path", help="Directory path containing images")

    # Command: search_image
    img_parser = subparsers.add_parser("search_image", help="Search images by text")
    img_parser.add_argument("query", help="Text description")

    # Command: describe_image (Florence-2)
    desc_parser = subparsers.add_parser("describe_image", help="Generate detailed caption for an image")
    desc_parser.add_argument("path", help="Path to image file")

    # Command: ask_image (Florence-2)
    ask_parser = subparsers.add_parser("ask_image", help="Ask questions about an image")
    ask_parser.add_argument("path", help="Path to image file")
    ask_parser.add_argument("question", help="Your question")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    print("⏳ 正在初始化 AI Agent (Loading All Models)...")
    db = DBManager()
    llm = LLMClient()
    # 注意：首次运行会加载 Florence-2，可能占用 2-3GB 显存
    vision_expert = VisionExpert()
    print("✅ 全系统初始化完成！\n")

    if args.command == "add_paper":
        files_to_process = []
        if os.path.isfile(args.path):
            files_to_process.append(args.path)
        elif os.path.isdir(args.path):
            for root, _, files in os.walk(args.path):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        files_to_process.append(os.path.join(root, f))
        
        print(f"🚀 开始处理 {len(files_to_process)} 个文件 (按页索引)...")
        
        for file_path in tqdm(files_to_process):
            chunks = extract_text_from_pdf(file_path)
            if not chunks:
                continue
            
            # 使用第一页内容进行分类
            first_page_text = chunks[0]['text']
            category = llm.classify_paper(first_page_text, args.topics)
            print(f"\n📄 文件: {os.path.basename(file_path)} -> 🏷️ 分类: {category}")
            
            new_path = move_file_to_category(file_path, category)
            db.add_paper_chunks(new_path, chunks, category)

    elif args.command == "search_paper":
        print(f"🔍 正在检索并思考: '{args.query}' ...")
        results = db.search_papers(args.query, n_results=3)
        
        if not results['ids'][0]:
            print("❌ 未找到相关信息。")
            return

        context_str = ""
        print("\n📚 [检索到的参考片段]:")
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            text = results['documents'][0][i]
            
            context_str += f"--- 文档: {meta['source']} (第 {meta['page']} 页) ---\n{text}\n\n"
            
            print(f"[{i+1}] {os.path.basename(meta['source'])}")
            print(f"    📍 页码: Page {meta['page']} | 匹配度: {1-dist:.4f}")
            print(f"    📝 片段: \"{text[:100].replace(chr(10), ' ')}...\"\n")

        print("🤖 [AI 智能回答]:")
        answer = llm.chat_with_context(args.query, context_str)
        print(f"{answer}\n")

    elif args.command == "scan_images":
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        count = 0
        for root, _, files in os.walk(args.path):
            for f in files:
                if os.path.splitext(f)[1].lower() in image_exts:
                    path = os.path.join(root, f)
                    try:
                        if db.add_image_embedding(path):
                            print(f"✅ 索引: {f}")
                            count += 1
                    except Exception:
                        pass
        print(f"\n🎉 已索引 {count} 张图片。")

    elif args.command == "search_image":
        print(f"🖼️ 正在寻找: '{args.query}'...")
        results = db.search_images(args.query)
        if not results['ids'][0]:
            print("未找到相关图片。")
        else:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                dist = results['distances'][0][i]
                print(f"[{i+1}] {doc_id} - 匹配度: {1-dist:.4f}")

    elif args.command == "describe_image":
        print(f"🎨 正在深度解析图片: {args.path} ...")
        result = vision_expert.analyze_image(args.path)
        print("\n📝 [图片内容描述]:")
        print(result)

    elif args.command == "ask_image":
        print(f"❓ 正在向图片提问: '{args.question}' ...")
        result = vision_expert.analyze_image(args.path, user_question=args.question)
        print("\n🤖 [回答]:")
        print(result)

if __name__ == "__main__":
    main()