import json
import os
from collections import Counter
import re

# ... (之前的 generate_image_caption_manual, process_folder_and_generate_captions 函数定义，以及 Blip 模型加载部分保持不变) ...
#  is_likely_invalid_caption 函数保持不变，但在 build_dataset_json 中使用它来过滤


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


def is_likely_invalid_caption(caption, min_repeats=2, min_percentage=0.5):
    """
    判断生成的字幕是否可能无效 (通过检测重复单词)。

    Args:
        caption (str): 生成的字幕文本。
        min_repeats (int): 单词连续重复出现的最小次数，超过此次数则视为无效。
        min_percentage (float): 某个单词出现次数占总单词数的最小比例，超过此比例则视为无效。

    Returns:
        bool: 如果字幕可能无效，返回 True，否则返回 False。
    """
    print("字幕内容", caption)


    if not caption or len(caption.strip()) < 2: # 空字幕或只有很少字符的字幕也视为无效
        return True
    # 将字幕转换为小写并简单分词 (按非字母数字字符分割)
    words = re.findall(r'\b\w+\b', caption.lower()) # 使用正则表达式找到所有单词 (字母数字序列)

    if not words: # 如果分词后没有单词，也视为无效
        return True
    print("分词", words)
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

    for word, count in word_counts.items():
        if count / total_words >= min_percentage:
            # print(f"检测到高频单词: {word}，出现次数: {count}, 比例: {count/total_words}") # 调试打印
            return True

    return False # 未检测到重复模式，视为有效

def build_dataset_json_final(raw_data_path, captions_json_path, output_dataset_json_path):
    """
    从原始数据和图片字幕生成最终的数据集 JSON 文件，
    统一使用按空格分词，去除字幕中的特殊字符，并根据字幕有效性过滤。

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
    #  这里不按 image_id 聚合，直接处理原始数据块列表
    #  由于原始数据块是按图片ID和方面排列的，我们可以直接遍历 raw_data_blocks

    #  需要记录已经处理过的图片的原始文本模板，以便重复使用和分词
    #  以及已经计算过的图片的字幕
    processed_image_data = {} # 存储已经处理过的图片的原始文本模板和字幕

    for block in raw_data_blocks:
        raw_text_template = block["raw_text_template"]
        aspect_term = block["aspect_term"]
        polarity_int = block["polarity_int"]
        image_id = block["image_id"]

        #  如果当前图片ID的数据是第一次处理
        if image_id not in processed_image_data:
            # 获取该图片的字幕
            image_name_without_ext = os.path.splitext(image_id)[0]
            caption = image_captions.get(image_name_without_ext) # 使用 get 方法获取字幕

            # --- 去除字幕中的特殊字符 ---
            special_char_pattern_backtick = "`" # <---  去除反引号
            special_char_pattern_comma = ","   # <---  去除逗号
            if caption is not None and isinstance(caption, str):
                caption = caption.replace(special_char_pattern_backtick, "").strip()
                caption = caption.replace(special_char_pattern_comma, "").strip()


            # --- 判断字幕是否有效 ---
            # 使用 is_likely_invalid_caption 判断，它会自动处理空字幕等情况
            if is_likely_invalid_caption(caption, min_repeats=2, min_percentage=0.5): #  使用你的阈值 2
                print(f"警告: 图片 '{image_id}' 的字幕可能无效 (重复单词或其他模式): '{caption}'，跳过此图片下的所有相关数据块。")
                #  标记这张图片的数据需要跳过
                processed_image_data[image_id] = {"skip": True}
                continue # 跳过当前数据块

            else:
                 # 字幕有效，存储处理后的信息
                 processed_image_data[image_id] = {
                     "caption": caption,
                     "raw_text_template": raw_text_template, # 存储该图片的原始文本模板
                     "skip": False
                 }

        #  如果这张图片的数据需要跳过，则跳过当前数据块
        if processed_image_data[image_id].get("skip", False):
             continue # 跳过当前数据块


        # --- 构建当前数据块对应的独立 JSON 项 ---

        # 1. 生成最终文本（在 $T$ 处替换方面术语）
        #  使用当前数据块的 raw_text_template
        final_text = raw_text_template.replace("$T$", aspect_term)

        # 2. 使用按空格分词，获取 words 列表
        final_words = final_text.split() # 按空格分词

        # 3. 计算方面术语在 final_text 字符串中的字符位置 (用于后续索引计算)
        #  不是必需，但有助于理解
        aspect_term_start_char_idx = final_text.find(aspect_term)
        aspect_term_end_char_idx = -1
        if aspect_term_start_char_idx != -1:
             aspect_term_end_char_idx = aspect_term_start_char_idx + len(aspect_term)


        # 4. 计算方面术语（按空格分词后的列表）在 final_words (按空格分词后的列表) 中的索引
        term_words = aspect_term.split() # 方面术语本身按空格分词

        start_index = -1
        end_index = -1

        if term_words:
            # 查找 term_words 列表在 final_words 列表中的起始位置
            for word_idx in range(len(final_words) - len(term_words) + 1):
                 # 比较子列表
                 if final_words[word_idx:word_idx + len(term_words)] == term_words:
                     start_index = word_idx
                     end_index = word_idx + len(term_words) # 结束索引是开区间
                     break # 找到后就退出循环

        # 5. 构建 aspect 部分
        aspects_list = []
        aspects_list.append({
            "from": start_index, # 方面术语的起始索引 (按空格分词后)
            "to": end_index,   # 方面术语的结束索引 (开区间) (按空格分词后)
            "polarity": map_polarity(polarity_int),
            "term": term_words # 方面术语列表 (按空格分词后)
        })

        # 6. 构建最终的 JSON 字典结构
        final_item = {
            "words": final_words, #  按空格分词后的 token 列表
            "image_id": image_id,
            "aspects": aspects_list, #  现在 aspects_list 只包含当前方面
            "opinions": [{"term": []}], #  根据你的需求填充或保留空列表
            "caption": processed_image_data[image_id]["caption"], #  使用之前处理好的图片字幕
            "image_path": f"/images/test/{image_id}", #  根据你的实际文件路径结构调整
            "aspects_num": len(aspects_list) #  这里 aspects_num 始终是 1
        }

        final_dataset.append(final_item) # 将这个独立的数据项添加到最终列表

    print(f"成功处理原始数据，共生成 {len(final_dataset)} 个数据项。")


    # 4. 将最终的 JSON 数据保存到文件
    try:
        with open(output_dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=4)

        print(f"最终数据集 JSON 文件已保存到: {output_dataset_json_path}")

    except Exception as e:
        print(f"保存最终数据集 JSON 文件时出错: {e}")


# --- 使用示例 ---
if __name__ == '__main__':
    #  你需要提供原始数据文件路径和输出 JSON 文件路径
    raw_data_file = "data/IJCAI2019_data/twitter2017/dev.txt" # <--- 替换为你的原始数据文件路径
    image_captions_file = "data/twitter2017_image_captions.json" # <--- 替换为你之前生成的图片字幕 JSON 文件路径
    final_dataset_file = "data/final_multimodal_dataset_twitter2017_dev_aspect.json" # 输出最终的数据集 JSON 文件名

    # 调用构建数据集函数
    build_dataset_json_final(raw_data_file, image_captions_file, final_dataset_file)

    print("数据集 JSON 文件生成完成！")