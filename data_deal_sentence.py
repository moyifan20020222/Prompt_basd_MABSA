import json
import os
import json
import os
from collections import Counter # 导入 Counter 类，用于统计词频
import re # 导入正则表达式模块，用于更灵活地分词
from src.data.tokenization_new_for_generated_prompt import ConditionTokenizer
# ... (之前的 generate_image_caption_manual, process_folder_and_generate_captions 函数定义，以及 Blip 模型加载部分保持不变) ...
ABNORMAL_CHAR_PATTERN = r"[^\w\s]{3,}" # 匹配连续出现 3 次以上的非单词字符或非空白符
CHARS_TO_NORMALIZE = ["'", '"', "-"] # 单引号、双引号、破折号等，根据需要添加

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


def is_likely_invalid_caption(caption, min_repeats=2, min_percentage=0.2):
    """
    判断生成的字幕是否可能无效 (通过检测重复单词)。

    Args:
        caption (str): 生成的字幕文本。
        min_repeats (int): 单词连续重复出现的最小次数，超过此次数则视为无效。
        min_percentage (float): 某个单词出现次数占总单词数的最小比例，超过此比例则视为无效。

    Returns:
        bool: 如果字幕可能无效，返回 True，否则返回 False。
    """



    if not caption or len(caption.strip()) < 2: # 空字幕或只有很少字符的字幕也视为无效
        return True
    # 将字幕转换为小写并简单分词 (按非字母数字字符分割)
    words = re.findall(r'\b\w+\b', caption.lower()) # 使用正则表达式找到所有单词 (字母数字序列)

    if not words: # 如果分词后没有单词，也视为无效
        return True
    # print("分词", words)
    if len(words) > 17:
        print("字幕内容", caption)
        return True
    # 1. 检测连续重复单词
    repeat_count = 0
    if len(words) > 1:
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                repeat_count += 1
                if repeat_count >= min_repeats - 1: # 如果连续重复次数达到 min_repeats
                    # print(f"检测到连续重复单词: {words[i]}，次数: {repeat_count + 1}") # 调试打印
                    return True
            else:
                repeat_count = 0 # 重置计数

    # 2. 检测高频重复单词 (某个单词出现次数占总单词数的比例很高)
    word_counts = Counter(words) # 统计每个单词出现的次数
    total_words = len(words)
    min_short_word_len = 2
    min_super_majority_percentage = 0.9
    max_single_word_len = 20
    #  检查是否存在某个短单词出现次数占总单词数的超高比例
    for word, count in word_counts.items():
        if len(word) <= min_short_word_len and count / total_words >= min_super_majority_percentage:  # 如果是短单词且出现比例极高
            print(f"检测到极长重复序列: 短单词 '{word}' 出现次数: {count}, 比例: {count / total_words}")
            return True
    for word, count in word_counts.items():
        if count / total_words >= min_percentage:
            # print(f"检测到高频单词: {word}，出现次数: {count}, 比例: {count/total_words}") # 调试打印
            return True
    # --- 新增：检测单个分词后单词的长度 ---
    for word in words:
        if len(word) > max_single_word_len:
             print(f"检测到超长单词: '{word}', 长度: {len(word)}")
             return True

    return False # 未检测到重复模式，视为有效


def map_polarity(polarity_int):
    """将整数情感极性映射到字符串情感极性"""
    if polarity_int == 0:
        return "NEU"
    elif polarity_int == 1:
        return "POS"
    elif polarity_int == -1:
        return "NEG"
    else:
        return "UNK" # 未知极性，根据需要处理

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

def build_dataset_json(raw_data_path, captions_json_path, output_dataset_json_path):
    """
    从原始数据和图片字幕生成最终的数据集 JSON 文件。

    Args:
        raw_data_path (str): 原始数据文件的本地路径。
        captions_json_path (str): 包含图片字幕的 JSON 文件的本地路径 (之前生成的)。
        output_dataset_json_path (str): 输出最终数据集 JSON 文件的本地路径。
    """

    # 1. 加载图片字幕数据
    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            image_captions = json.load(f)
        print(f"成功加载图片字幕文件: {captions_json_path}")
    except FileNotFoundError:
        print(f"错误: 图片字幕文件未找到: {captions_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析图片字幕文件，请确保它是有效的 JSON 文件: {captions_json_path}")
        return
    except Exception as e:
        print(f"加载图片字幕文件时发生其他错误: {e}")
        return

    # 2. 读取原始数据并解析
    raw_data_blocks = []
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() # 按行读取所有数据

        # 假设每个数据块是 4 行，你需要根据实际情况调整这个数字
        block_size = 4
        for i in range(0, len(lines), block_size):
            block = [line.strip() for line in lines[i:i+block_size]] # 去除每行的空白符
            if len(block) == block_size: # 确保读取了一个完整的数据块
                 # 解析数据块
                raw_text_template = block[0]
                aspect_term = block[1]
                polarity_str = block[2]
                image_id = block[3]

                try:
                    polarity_int = int(polarity_str) # 转换为整数
                except ValueError:
                    print(f"警告: 无法将极性 '{polarity_str}' 转换为整数，跳过此数据块: {block}")
                    continue # 跳过无法解析极性的数据块

                # 将原始数据块信息存储起来
                raw_data_blocks.append({
                    "raw_text_template": raw_text_template,
                    "aspect_term": aspect_term,
                    "polarity_int": polarity_int,
                    "image_id": image_id
                })

            else:
                print(f"警告: 文件末尾可能存在不完整的数据块，跳过: {block}")


    except FileNotFoundError:
        print(f"错误: 原始数据文件未找到: {raw_data_path}")
        return
    except Exception as e:
        print(f"读取或解析原始数据文件时发生其他错误: {e}")
        return

    print(f"成功读取和解析原始数据文件，共 {len(raw_data_blocks)} 个数据块。")


    # 3. 构建最终的 JSON 数据结构
    final_dataset = []
    #  需要一种方式将属于同一张图片的数据块聚合起来
    #  这里使用字典按 image_id 聚合数据块
    aggregated_data_by_image = {}
    for block in raw_data_blocks:
        image_id = block["image_id"]
        if image_id not in aggregated_data_by_image:
            aggregated_data_by_image[image_id] = {
                "image_id": image_id,
                "raw_blocks": [] # 存储属于这个 image_id 的原始数据块
            }
        aggregated_data_by_image[image_id]["raw_blocks"].append(block)

    print(f"原始数据按图片 ID 聚合完成，共 {len(aggregated_data_by_image)} 张独立图片。")


    # 遍历每个图片的聚合数据，构建最终结构
    for image_id, data in aggregated_data_by_image.items():
        # 获取该图片的字幕
        #  使用 image_id (不带扩展名) 作为键查找字幕
        #  假设 image_id 是文件名 (带扩展名)，需要去掉扩展名进行查找
        image_name_without_ext = os.path.splitext(image_id)[0]
        caption = image_captions.get(image_name_without_ext) # 使用 get 方法获取字幕，如果不存在则返回 None
        special_char_pattern = "`"  # <---  **修改这里：替换为你实际观察到的特殊字符的精确字符串**
        special_char_pattern1 = ","
        if caption is not None and isinstance(caption, str):  # 确保 caption 不是 None 且是字符串
            caption = caption.replace(special_char_pattern, "").strip()  # 替换并去除首尾空白符
        if caption is not None and isinstance(caption, str):  # 确保 caption 不是 None 且是字符串
            caption = caption.replace(special_char_pattern1, "").strip()  # 替换并去除首尾空白符
        # 检查字幕是否是无效的 (例如之前标记的 "[INVALID_CAPTION]" 或 None)
        #  这里假设之前无效的字幕被标记为 "[INVALID_CAPTION]"
        if is_likely_invalid_caption(caption, min_repeats=3, min_percentage=0.3):  # 使用重复单词判断逻辑
            print(f"警告: 图片 '{image_id}' 的字幕可能无效 (重复单词或其他模式): '{caption}'，跳过此图片的数据。")
            caption = "<<INVALID_CAPTION>>"
        caption = re.sub(ABNORMAL_CHAR_PATTERN, " ", caption).strip()

        # --- 新增：规范化特定字符 ---
        # 将单引号、双引号、破折号等替换为标准的空格
        # 替换为单个空格，以保持词语分隔
        for char in CHARS_TO_NORMALIZE:
            caption = caption.replace(char, " ")

        # 再次去除多余的空格
        caption = re.sub(r"\s+", " ", caption).strip()  # 将连续的空白符替换为单个空格
        # 遍历该图片下的所有方面数据块
        aspects_list = []
        #  由于原始数据块是独立的，我们需要从中提取 aspect 信息
        #  并找到对应的原始文本，然后计算索引
        #  这里假设所有属于同一个 image_id 的数据块的 "raw_text_template" 是相同的
        #  （即同一个推文模板，只是替换的词和极性不同）
        #  如果同一个图片ID对应不同的推文模板，需要更复杂的处理逻辑

        #  使用第一个数据块的原始文本作为基准，并进行替换和分词
        if not data["raw_blocks"]: # 如果没有属于这张图片的原始数据块，跳过
            continue

        for block in data["raw_blocks"]:
            aspect_term = block["aspect_term"]
            polarity_int = block["polarity_int"]
            # 选取第一个数据块的原始文本模板作为基准
            base_raw_text_template = block["raw_text_template"]
            # 进行分词，以便后续计算索引
            #  这里简单按空格分词，你需要根据实际的文本和分词需求调整
            base_words = base_raw_text_template.split()  # 按空格分词
            #  将方面术语替换回原始文本模板
            #  这里假设 $T$ 占位符只出现一次
            processed_text = base_raw_text_template.replace("$T$", aspect_term).split()

            #  计算方面术语在分词后的文本中的索引
            #  这需要根据你实际的分词方法和原始文本结构来精确计算
            #  这里的索引计算是一个简化的示例，可能需要调整
            start_index = -1
            end_index = -1
            try:
                # 找到方面术语在 base_words 中的位置
                #  需要处理方面术语可能包含多个词的情况
                term_words = aspect_term.split() # 方面术语本身也按空格分词
                # print("总体文本", processed_text)
                # print("部分文本", term_words)
                if term_words:
                    # 查找 term_words 在 base_words 中的起始位置
                    for word_idx in range(len(processed_text) - len(term_words) + 1):
                         if processed_text[word_idx:word_idx + len(term_words)] == term_words:
                             start_index = word_idx
                             end_index = word_idx + len(term_words) # 结束索引是开区间
                             break # 找到后就退出循环

            except Exception as e:
                 print(f"警告: 计算图片 '{image_id}' 方面 '{aspect_term}' 的索引时出错: {e}")
                 # 继续处理下一个方面，或者跳过此图片，根据需求调整

            # 构建 aspect 部分
            aspects_list.append({
                "from": start_index, # 方面术语的起始索引
                "to": end_index,   # 方面术语的结束索引 (开区间)
                "polarity": map_polarity(polarity_int), # 情感极性字符串
                "term": term_words # 方面术语列表
            })

        aspect_term = data["raw_blocks"][0]["aspect_term"]
        processed_text_last = data["raw_blocks"][0]["raw_text_template"].replace("$T$", aspect_term).split()
        scores = get_blip_itm_score(f"data//k-shot-has-aspect-num-ACL-2023/twitter_2017/images/test/{image_id}",
                                    data["raw_blocks"][0]["raw_text_template"].replace("$T$", aspect_term))
        print("score的值", scores)
        #  构建最终的字典结构
        final_item = {
            #  这里的 words 需要是最终替换后的文本，并按空格分词
            #  但是原始数据提供了带有 $T$ 的模板和替换的 term
            #  如果每个数据块的原始文本模板都相同，可以用第一个数据块的替换结果
            #  如果不同，需要更复杂的处理
            "words": processed_text_last, #  使用最后一个方面块替换后的文本进行分词 (假设所有块的模板相同)
            "image_id": image_id,
            "aspects": aspects_list,
            "opinions": [{"term": []}], #  根据你的需求填充或保留空列表
            "caption": caption, #  从图片字幕文件中获取的字幕
            "image_path": f"/images/test/{image_id}", #  根据你的实际文件路径结构调整
            "aspects_num": len(aspects_list),   #  方面数量
            "score": scores
        }

        final_dataset.append(final_item)

    # 4. 将最终的 JSON 数据保存到文件
    try:
        with open(output_dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=4)

        print(f"最终数据集 JSON 文件已保存到: {output_dataset_json_path}")

    except Exception as e:
        print(f"保存最终数据集 JSON 文件时出错: {e}")


# --- 使用示例 ---
if __name__ == '__main__':
    # raw_data_file = "data/IJCAI2019_data/twitter2015/dev.txt" # <--- 替换为你的原始数据文件路径
    # image_captions_file = "data/twitter2015_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    # final_dataset_file = "data/final_multimodal_dataset_twitter2015_dev.json" # 输出最终的数据集 JSON 文件名
    #
    # build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)
    # raw_data_file = "data/IJCAI2019_data/twitter2015/test.txt" # <--- 替换为你的原始数据文件路径
    # image_captions_file = "data/twitter2015_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    # final_dataset_file = "data/final_multimodal_dataset_twitter2015_test.json" # 输出最终的数据集 JSON 文件名
    #
    # build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)
    # raw_data_file = "data/IJCAI2019_data/twitter2015/train.txt" # <--- 替换为你的原始数据文件路径
    # image_captions_file = "data/twitter2015_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    # final_dataset_file = "data/final_multimodal_dataset_twitter2015_train.json" # 输出最终的数据集 JSON 文件名
    #
    # build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)

    raw_data_file = "data/IJCAI2019_data/twitter2017/dev.txt" # <--- 替换为你的原始数据文件路径
    image_captions_file = "data/twitter2017_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    final_dataset_file = "data/final_multimodal_dataset_twitter2017_dev.json" # 输出最终的数据集 JSON 文件名

    build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)
    raw_data_file = "data/IJCAI2019_data/twitter2017/test.txt" # <--- 替换为你的原始数据文件路径
    image_captions_file = "data/twitter2017_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    final_dataset_file = "data/final_multimodal_dataset_twitter2017_test.json" # 输出最终的数据集 JSON 文件名

    build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)
    raw_data_file = "data/IJCAI2019_data/twitter2017/train.txt" # <--- 替换为你的原始数据文件路径
    image_captions_file = "data/twitter2017_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    final_dataset_file = "data/final_multimodal_dataset_twitter2017_train.json" # 输出最终的数据集 JSON 文件名

    build_dataset_json(raw_data_file, image_captions_file, final_dataset_file)

    print("数据集 JSON 文件生成完成！")