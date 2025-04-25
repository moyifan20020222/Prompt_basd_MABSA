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
import os
import json
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch # 导入 torch 库

# 定义设备 (可以指定具体的 GPU 索引，例如 "cuda:0", "cuda:1" 等)
# 如果没有 GPU 或者想强制使用 CPU，可以改为 "cpu"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(f"将模型和数据加载到设备: {device}") # 打印使用的设备信息

# --- 在函数外部加载模型和处理器 (避免在每次生成字幕时重复加载) ---
# 第一次运行会从 Hugging Face Hub 自动下载
try:
    blip_model = BlipForConditionalGeneration.from_pretrained("data/Blip")
    blip_processor = BlipProcessor.from_pretrained("data/Blip")

    # 将模型移动到指定设备
    blip_model.to(device)
    # 设置模型为评估模式 (推理模式)
    blip_model.eval()
    print("BLIP 模型和处理器加载成功。")

except Exception as e:
    print(f"加载 BLIP 模型或处理器出错: {e}")
    print("请确保你的网络连接正常，并且 transformers 和 huggingface_hub 库已正确安装且是最新版本。")
    # 如果加载失败，可以考虑退出或进行错误处理
    exit() # 在这个示例中，如果加载失败就直接退出


def generate_image_caption_manual(image_path):
    """
    手动加载模型并移动到指定设备，为给定路径的图片生成字幕。

    Args:
        image_path (str): 图片文件的本地路径。

    Returns:
        str: 生成的图片字幕文本。
    """
    global blip_model, blip_processor, device # 声明使用外部定义的全局变量

    try:
        # 1. 打开并预处理输入图片
        image = Image.open(image_path).convert("RGB") # 确保是 RGB 格式

        # 2. 使用处理器预处理图片，并创建输入张量
        # 将输入数据也移动到指定设备 (GPU/CPU)
        inputs = blip_processor(images=image, return_tensors="pt").to(device)

        # 3. 生成字幕
        with torch.no_grad(): # 在推理时禁用梯度计算，可以节省内存和加快速度
            # 生成字幕的参数可以根据需要调整
            # 例如: max_length (最大长度), num_beams (beam search 宽度), early_stopping (是否提前停止), num_return_sequences (返回候选项数量)
            outputs = blip_model.generate(**inputs, max_length=50, num_beams=4) # 示例参数：beam search 宽度为 4

        # 4. 解码生成的 token 为文本
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

        # 5. 返回生成的字幕
        return caption

    except Exception as e:
        # 在生成字幕过程中捕获错误，例如图片处理失败、模型推理错误等
        raise e # 将异常重新抛出，由调用函数处理


def process_folder_and_generate_captions(folder_path, output_json_path):
    """
    处理指定文件夹下的所有图片，生成字幕，并将结果保存到 JSON 文件 (使用手动加载模型)。

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
                # 生成字幕 (调用手动加载模型的函数)
                caption = generate_image_caption_manual(image_path)

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
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']  #  常见的图片扩展名，可以根据需要添加
    for ext in valid_extensions:
        if filename.lower().endswith(ext):
            return True
    return False


if __name__ == '__main__':
    #  ---  使用示例  ---
    image_folder = "data/k-shot-has-aspect-num-ACL-2023/twitter_2017/images/test"  #  <---  **请替换为你的图片文件夹路径**
    output_json_file = "data/twitter2017_image_captions.json"  #  输出 JSON 文件名 (可以自定义)

    # 处理文件夹并生成字幕 JSON 文件
    process_folder_and_generate_captions(image_folder, output_json_file)

    print("字幕生成和 JSON 文件保存完成！")

