import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval  # 或者 Blip2ForConditionalGeneration 等, 取决于模型
from PIL import Image
import requests

device = "cuda:2" if torch.cuda.is_available() else "cpu"

# 加载预训练的BLIP模型 (选择支持ITM的) 和处理器
# 例如: "Salesforce/blip-itm-base-coco" 或 "Salesforce/blip-image-captioning-base" (也能做ITM)
model_name = "data/Blip-ITM"  # 专门为检索/ITM微调过
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)


# 如果用BLIP-2, 可能需要 Blip2Processor 和 Blip2ForConditionalGeneration


def get_blip_itm_score(image_path_or_url, text_description):
    """使用BLIP ITM头计算图像和文本的匹配得分 (通常是logit或概率)."""
    try:
        if image_path_or_url.startswith('http'):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
    except Exception as e:
        print(f"Error loading image {image_path_or_url}: {e}")
        return 0.0

    try:
        # 处理图像和文本
        # 注意：BLIP的ITM任务通常需要将文本作为'caption'输入
        inputs = processor(images=image, text=text_description, return_tensors="pt").to(device)

        # 获取ITM得分 (通常是logit)
        with torch.no_grad():
            # BlipForImageTextRetrieval 输出的是 itm_score
            outputs = model(**inputs, use_itm_head=True)  # 确保使用ITM头
            itm_score = outputs.itm_score  # 通常是 [batch_size, 2] 的 logits (match vs no-match)

        # 将logits转换为匹配概率
        # itm_score[:, 1] 是匹配的logit
        itm_probability = torch.softmax(itm_score, dim=1)[:, 1].item()  # 获取匹配概率

        return itm_probability

    except Exception as e:
        print(f"Error processing with BLIP ITM: {e}")
        return 0.0


if __name__ == '__main__':
    # image_file = "data/116233.jpg"  # 或者 URL
    # aspect_term = "RT @ kaj33 : At Final Four rooting for Wisconsin"  # 来自你的MABSA数据

    # aspect_term = "RT @ brainpicker : Treat your soul to Thoreau on what it means to live life fully awake"
    # image_file = "data/1141299.jpg"

    aspect_term = "RT @ KevinMaddenDC: Al Czervik for President"
    image_file = "data/6020.jpg"

    # 计算相似度
    similarity_score = get_blip_itm_score(image_file, aspect_term)
    print(f"CLIP Similarity between image and '{aspect_term}': {similarity_score:.4f}")
