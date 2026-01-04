from openai import OpenAI
from .config import Config

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.API_KEY,
            base_url=Config.BASE_URL
        )

    def classify_paper(self, text_snippet, topics):
        prompt = f"""
        你是一个专业的学术助手。请根据以下论文摘要，将其归类到以下类别之一：{topics}。
        如果文本主要讨论的是 Deep Learning, Neural Networks 等，优先归类为 'Deep Learning' 或相关具体的子领域。
        
        论文摘要：
        {text_snippet[:1000]}...

        请严格只返回类别名称，不要包含其他字符。如果无法确定，返回 "Uncategorized"。
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            category = response.choices[0].message.content.strip()
            clean_topics = [t.strip() for t in topics.split(',')]
            for topic in clean_topics:
                if topic.lower() in category.lower():
                    return topic
            return "Uncategorized"
        except Exception as e:
            print(f"LLM 分类失败: {e}")
            return "Uncategorized"

    def chat_with_context(self, query, context):
        """
        RAG 核心方法
        """
        prompt = f"""
        请基于以下参考文档片段（Context），回答用户的学术问题。
        
        [参考文档]:
        {context}
        
        [用户问题]: {query}
        
        要求：
        1. 回答要简洁、专业。
        2. 如果参考文档中没有答案，请直接说“根据现有文档无法回答”。
        3. 请使用中文回答。
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答失败: {e}"