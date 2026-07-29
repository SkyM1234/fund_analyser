"""
视觉分析服务 - 用于PDF图片分析
"""
import os
import base64
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI


class VisionService:
    """视觉分析服务"""

    def __init__(self, model: str):
        """
        初始化 Qwen VL 客户端
        
        Args:
            model: VL模型名称
        """
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("未设置 QWEN_API_KEY 环境变量")

        # Qwen VL 客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    async def analyze_image_from_file(
        self, 
        file_path: str, 
        context: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """
        分析本地图片文件并返回文字描述

        Args:
            file_path: 本地图片文件路径
            context: 上下文信息
            prompt: 自定义提示词（覆盖默认提示词）

        Returns:
            图片的文字描述
        """
        try:
            # 读取图片并转换为 base64
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # 获取文件扩展名
            ext = Path(file_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')

            # 构建 data URL
            image_url = f"data:{mime_type};base64,{image_data}"

            # 构建提示词 - 要求返回段落式描述，不要分点
            if prompt is None:
                prompt = """请详细识别并描述这张图片中的所有内容。要求：
1. 用自然流畅的段落描述，不要使用分点列举或编号
2. 如果是图表，请描述图表类型、数据趋势和关键信息
3. 如果包含文字，请识别所有可见文字
4. 整体描述要详细完整，就像在给别人口述图片内容一样"""

            if context:
                prompt = f"{context}\n\n{prompt}"

            # 调用 Qwen VL API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )

            description = response.choices[0].message.content.strip()
            return description

        except Exception as e:
            return f"[图片分析失败: {str(e)}]"


# 全局服务实例（延迟初始化）
_vision_service = None


def get_vision_service(model: str) -> VisionService:
    """
    获取全局视觉服务实例（单例模式）
    
    Args:
        model: VL模型名称
        
    Returns:
        VisionService实例
    """
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService(model=model)
    return _vision_service
