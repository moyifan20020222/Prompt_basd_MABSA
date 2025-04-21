import os
import json
from transformers import pipeline
from PIL import Image

"""
对于twitter 数据集， $T$ 用于指示Aspect的位置 0 为中立 1为 积极 -1 为 消极
每一行数据包含一个 Aspect 重复的图片ID说明 不止有一个Aspect 而是包含多个。
例子：
目标数据格式
 {"words": ["Dez", "Bryant", "believes", "in", "rookie", "CBs", "despite", "Darrelle", "Revis", "tweet", "#", "cowboys", "#", "NFL"], 
 "image_id": "17_06_13066.jpg", 
 "aspects": [{"from": 0, "to": 2, "polarity": "NEU", "term": ["Dez", "Bryant"]}, 
 {"from": 5, "to": 6, "polarity": "POS", "term": ["CBs"]}, 
 {"from": 7, "to": 9, "polarity": "NEU", "term": ["Darrelle", "Revis"]}, 
 {"from": 11, "to": 12, "polarity": "NEU", "term": ["cowboys"]}, 
 {"from": 13, "to": 14, "polarity": "NEU", "term": ["NFL"]}], 
 "opinions": [{"term": []}], "caption": "coach looks on during practice", 
 "image_path": "/images/test/17_06_13066.jpg", "aspects_num": 5}
 其中的caption为本部分代码生成内容 image_path 需要指示好 
而MADSA数据集中：
我们目标数据格式 保持统一。


"""

def generate_image_caption_clipcap(image_path):
    """
    使用 CLIP-Caption 模型为给定路径的图片生成字幕 (与之前代码相同)。
    """
    image_to_text = pipeline("image-to-text", model="data/Blip")
    image = Image.open(image_path)
    caption = image_to_text(image)[0]['generated_text']
    return caption

def process_folder_and_generate_captions(folder_path, output_json_path):
    """
    处理指定文件夹下的所有图片，生成字幕，并将结果保存到 JSON 文件。

    Args:
        folder_path (str): 包含图片的文件夹的本地路径。
        output_json_path (str): 输出 JSON 文件的本地路径。
    """

    image_caption_data = {}  #  用于存储图片文件名和字幕的字典

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # 检查是否是图片文件 (可以根据文件扩展名判断)
        if is_image_file(filename):  #  使用辅助函数判断是否是图片文件
            try:
                # 生成字幕
                caption = generate_image_caption_clipcap(image_path)

                #  使用图片文件名 (不带扩展名) 作为 JSON 字典的键
                image_name = os.path.splitext(filename)[0]  #  去掉文件扩展名
                image_caption_data[image_name] = caption  #  存储文件名和字幕

                print(f"为图片 '{filename}' 生成字幕成功: '{caption}'")  #  打印处理成功的提示信息

            except Exception as e:
                print(f"处理图片 '{filename}' 时出错: {e}")  #  打印错误信息，方便调试

    #  将 image_caption_data 字典保存到 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_caption_data, f, ensure_ascii=False, indent=4)  #  使用 indent=4 格式化 JSON，ensure_ascii=False 支持中文

    print(f"所有图片字幕已保存到 JSON 文件: '{output_json_path}'")  #  打印完成提示信息

def is_image_file(filename):
    """
    简单判断文件名是否是图片文件 (根据常见图片扩展名).
    可以根据你的实际情况扩展支持的图片格式。
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  #  常见的图片扩展名
    for ext in valid_extensions:
        if filename.lower().endswith(ext):
            return True
    return False


if __name__ == '__main__':
    #  ---  使用示例  ---
    image_folder = "data/IJCAI2019_data/twitter2015_images"  #  <---  **请替换为你的图片文件夹路径**
    output_json_file = "data/IJCAI2019_data/twitter2015_image_captions.json"  #  输出 JSON 文件名 (可以自定义)

    # 处理文件夹并生成字幕 JSON 文件
    process_folder_and_generate_captions(image_folder, output_json_file)

    print("字幕生成和 JSON 文件保存完成！")