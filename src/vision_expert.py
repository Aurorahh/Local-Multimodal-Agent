from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from .config import Config
import torch
import traceback # 引入详细报错工具

class VisionExpert:
    def __init__(self):
        print(f"👁️ 正在加载 Florence-2 视觉专家: {Config.VISION_MODEL_PATH} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.VISION_MODEL_PATH, 
                torch_dtype=self.torch_dtype, 
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                Config.VISION_MODEL_PATH, 
                trust_remote_code=True
            )
            print("✅ 视觉专家加载完毕！")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.model = None

    def analyze_image(self, image_path, prompt_type="<MORE_DETAILED_CAPTION>", user_question=None):
        if not self.model:
            return "❌ 模型未加载"
            
        try:
            # 1. 语言检查
            if user_question:
                for char in user_question:
                    if '\u4e00' <= char <= '\u9fff':
                        return "⚠️ Florence-2 仅支持英文提问 (Only English supported)."

            print(f"🔍 Debug: 正在打开图片 {image_path}")
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 2. 构建 Prompt
            # 关键：<VQA> 后面必须有空格，防止与问题粘连
            if user_question:
                task_prompt = "<VQA>"
                text_input = task_prompt + " " + user_question 
            else:
                task_prompt = prompt_type
                text_input = task_prompt

            # 3. 处理输入
            inputs = self.processor(text=text_input, images=image, return_tensors="pt")
            inputs = inputs.to(self.device, self.torch_dtype)

            # 4. 生成 (强制 Greedy Search)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False 
            )

            # 5. 【物理切片修复】只保留新生成的 token，彻底去除 Prompt 回显
            # 获取输入部分的长度
            input_token_len = inputs["input_ids"].shape[1]
            # 只取输入长度之后的部分（即纯粹的回答）
            new_tokens = generated_ids[0][input_token_len:]
            
            # 6. 解码
            answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

            # 7. 兜底检查：如果模型还是发疯输出了 <loc> 标签
            if "<loc" in answer or answer == "":
                # 尝试用官方后处理再救一次
                full_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    full_text, 
                    task=task_prompt, 
                    image_size=(image.width, image.height)
                )
                return parsed.get(task_prompt, answer)
            
            return answer
            
        except Exception as e:
            print(f"❌ 分析错误: {e}")
            traceback.print_exc()
            return f"Error: {e}"