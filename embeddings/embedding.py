import requests
from PIL import Image
from typing import Dict, Any, Tuple
import torch
import open_clip

MOLDE_CACHE_DIR = "/Users/hongdongjian/Documents/workspace/python/rag_demo/models/"

class ClipEmbedder:
    """用于生成文本和图像嵌入向量的类，基于OpenCLIP模型"""

    # 支持的模型、预训练权重标识（模型会加载与该标识对应的预训练参数）
    SUPPORTED_MODELS = {
        "ViT-B-32": "laion2b_s34b_b79k",
    }

    def __init__(self, model_name: str = "ViT-B-32"):
        """
        初始化CLIP嵌入生成器

        参数:
            model_name: 模型名称，可以是预定义的名称或模型路径
        """
        print(f"正在加载模型: {model_name}...")
        self.model_name = model_name
        self.pretrained = self.SUPPORTED_MODELS.get(model_name, None)
        if self.pretrained is None:
            raise ValueError(f"不支持的模型名称: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.tokenizer = self._load_model()

    def _load_model(self):
        """加载指定的模型"""
        try:
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                model_name=self.model_name,
                device=self.device,
                pretrained=self.pretrained,
                cache_dir=MOLDE_CACHE_DIR
            )
            tokenizer = open_clip.get_tokenizer(model_name=self.model_name)
            return model, preprocess_val, tokenizer
        except Exception as e:
            raise ValueError(f"无法加载模型 {self.model_name}: {str(e)}")

    def embed_text(self, text: str, normalized: bool=True) -> Tuple[torch.Tensor, int]:
        """
        生成文本嵌入并计算消耗的tokens数量

        参数:
            text: 输入文本

        返回:
            (嵌入向量, 消耗的tokens数量)
        """
        tokenized_text = self.tokenizer([text])
        # 计算tokens数量 - tokenized_text的形状为[batch_size, sequence_length]
        num_tokens = tokenized_text.shape[1]

        with torch.no_grad(), torch.amp.autocast(device_type=self.device):
            text_features = self.model.encode_text(tokenized_text)
            if normalized:
                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features, num_tokens

    def embed_image(self, image_source: str, normalized: bool=True) -> Tuple[torch.Tensor, int]:
        """
        生成图像嵌入并计算消耗的tokens数量

        参数:
            image_source: 图像URL或文件路径

        返回:
            (嵌入向量, 消耗的tokens数量)
        """
        try:
            if image_source.startswith("http://") or image_source.startswith("https://"):
                image = Image.open(requests.get(image_source, stream=True).raw)
            else:
                image = Image.open(image_source)
        except Exception as e:
            raise ValueError(f"无法加载图像: {image_source}, 错误: {str(e)}")

        # 计算图像tokens - 基于图像分辨率估算
        # CLIP通常将图像处理为固定分辨率(如224x224)，这里我们基于原始分辨率估算复杂度
        width, height = image.size
        # 使用图像面积的平方根作为估算依据，并乘以一个系数
        # 这是一种启发式方法，可以根据实际需求调整
        image_tokens = int((width * height) ** 0.5 / 10)
        # 设置一个最小值
        image_tokens = max(100, image_tokens)

        processed_image = self.processor(image).unsqueeze(0).to(self.device)
        # torch.no_grad()
        # 关闭PyTorch的自动求导功能
        # 在推理阶段使用，不需要计算和存储梯度
        # 显著减少内存消耗（不存储中间计算结果的梯度信息）
        # 加快计算速度（不执行反向传播所需的额外计算）
        # torch.amp.autocast(device_type=self.device)
        # 启用自动混合精度计算
        # 将某些操作从默认的float32精度自动降低到float16（GPU）或bfloat16
        # 减少内存使用并提高计算效率，尤其在支持Tensor Core的NVIDIA GPU上
        # 参数device_type=self.device指定在哪种设备上（CPU或GPU）应用混合精度
        with torch.no_grad(), torch.amp.autocast(device_type=self.device):
            image_features = self.model.encode_image(processed_image)
            # 这段代码是在进行向量归一化（L2 normalization），它的作用是将文本特征向量标准化为单位向量（unit vector）。
            # 确保向量长度为1：归一化后的向量模长为1，只保留方向信息
            # 提高相似度计算准确性：在CLIP模型中，归一化后的向量可以直接通过点积计算余弦相似度
            # 统一特征空间：使所有文本和图像向量都位于同一超球面上，便于跨模态匹配
            # 提高数值稳定性：防止极大或极小值造成的计算问题
            # 在CLIP这样的多模态模型中，文本和图像特征都需要归一化处理，这样它们的相似度比较才有意义，不会受到向量大小的影响。
            if normalized:
                image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features, image_tokens

    def generate_embeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        为文本和图像生成嵌入向量

        参数:
            params: 包含input的字典
                input: 输入列表，每项可以是{"text": "文本内容"}或{"image": "图像URL或路径"}

        返回:
            包含嵌入向量和相关信息的字典
        """
        inputs = params.get("input", [])
        normalized = params.get("normalized", True)

        try:
            total_tokens = 0
            embeddings_result = []

            for idx, item in enumerate(inputs):
                embedding = None
                tokens = 0

                if "text" in item:
                    embedding, tokens = self.embed_text(item["text"], normalized=normalized)
                elif "image" in item:
                    embedding, tokens = self.embed_image(item["image"], normalized=normalized)
                else:
                    continue

                total_tokens += tokens

                if embedding is not None:
                    embeddings_result.append({
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding.tolist()
                    })

            result = {
                "model": self.model_name,
                "object": "list",
                "usage": {
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_tokens
                },
                "data": embeddings_result
            }

            return result

        except Exception as e:
            raise ValueError(f"嵌入失败: {str(e)}")

if __name__ == "__main__":
    # 使用类进行测试
    embedder = ClipEmbedder()
    result = embedder.generate_embeddings({
        "input": [
            {"text": "这是一个测试文本"},
            # {"image": "https://i.ibb.co/nQNGqL0/beach1.jpg"}
        ]
    })
    print(result)