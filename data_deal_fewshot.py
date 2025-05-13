import json


# --- 假设的加载函数，你需要替换成你实际的加载方式 ---
def load_json_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 基本检查，确保加载的是列表，并且列表元素是字典
        if not isinstance(data, list):
            print(f"Warning: Data in {filepath} is not a list. Loaded type: {type(data)}")
            return []
        if data and not isinstance(data[0], dict):
            print(f"Warning: First item in {filepath} is not a dict. Item type: {type(data[0])}")
            # 可能需要进一步处理，例如如果是JSON Lines格式
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return []


# --- 加载数据 ---
my_data_path = "data/twitter_2017/full_data/dev.json"  # 替换为你的文件路径
gmp_data_path = "data/twitter_2017/42/dev.json" # 替换为GMP数据文件路径

# --- 保存生成的数据 ---
output_gmp_with_scores_path = "data/twitter_2017/42/dev_change.json"
output_my_data_updated_path = "data/twitter_2017/full_data/dev_change.json"

my_data_list = load_json_data(my_data_path)
gmp_data_list = load_json_data(gmp_data_path)

if not my_data_list or not gmp_data_list:
    print("One or both datasets could not be loaded. Exiting.")
    exit()

# --- 创建查找表 (方便快速匹配) ---
my_data_map = {}
for i, item in enumerate(my_data_list):
    if not isinstance(item, dict) or 'image_id' not in item:
        print(f"Skipping invalid item in my_data at index {i}: {item}")
        continue
    my_data_map[item['image_id']] = item

gmp_data_map = {}
for i, item in enumerate(gmp_data_list):
    if not isinstance(item, dict) or 'image_id' not in item:
        print(f"Skipping invalid item in gmp_data at index {i}: {item}")
        continue
    gmp_data_map[item['image_id']] = item

# --- 1. 生成增加了score的GMP数据 (gmp_data_with_scores) ---
gmp_data_with_scores = []
scores_added_to_gmp = 0
scores_defaulted_for_gmp = 0

for gmp_item_content in gmp_data_list:  # 确保遍历原始列表以保持顺序
    if not isinstance(gmp_item_content, dict) or 'image_id' not in gmp_item_content:
        # 如果原始gmp_data_list就有问题，这里也跳过，但最好在加载时处理
        continue

    new_gmp_item = gmp_item_content.copy()  # 从GMP数据开始
    gmp_image_id = gmp_item_content['image_id']

    my_corresponding_item = my_data_map.get(gmp_image_id)
    if my_corresponding_item and 'score' in my_corresponding_item:
        new_gmp_item['score'] = my_corresponding_item['score']
        scores_added_to_gmp += 1
    else:
        new_gmp_item['score'] = 0.0  # 默认score
        scores_defaulted_for_gmp += 1

    # caption 和其他字段保持GMP原始的
    gmp_data_with_scores.append(new_gmp_item)

print(f"\n--- GMP Data with Scores ---")
print(f"Total items in gmp_data_with_scores: {len(gmp_data_with_scores)}")
print(f"Scores successfully added from your data: {scores_added_to_gmp}")
print(f"Scores defaulted (not found in your data or no score field): {scores_defaulted_for_gmp}")

# --- 2. 生成被GMP字幕修改了的你的数据 (my_data_captions_updated) ---
my_data_captions_updated = []
captions_updated_in_my_data = 0
captions_kept_original_in_my_data = 0

INVALID_CAPTION_MARKER = "<<INVALID_CAPTION>>"

for my_item_content in my_data_list:  # 确保遍历原始列表以保持顺序
    if not isinstance(my_item_content, dict) or 'image_id' not in my_item_content:
        continue

    new_my_item = my_item_content.copy()  # 从你的数据开始
    my_image_id = my_item_content['image_id']
    my_original_caption = my_item_content.get('caption')

    gmp_corresponding_item = gmp_data_map.get(my_image_id)
    if gmp_corresponding_item:
        gmp_caption = gmp_corresponding_item.get('caption')
        # 条件：GMP有有效caption，并且你的caption是无效标记或为空/None
        if gmp_caption is not None and gmp_caption.strip() != "":  # 确保GMP caption不是空字符串
            if my_original_caption == INVALID_CAPTION_MARKER or \
                    my_original_caption is None or \
                    my_original_caption.strip() == "":
                new_my_item['caption'] = gmp_caption
                captions_updated_in_my_data += 1
            # 如果你的caption有效，但你想比较是否要用GMP的覆盖，可以在这里加逻辑
            # 例如: elif my_original_caption != gmp_caption:
            #           print(f"For {my_image_id}, your_caption='{my_original_caption}', gmp_caption='{gmp_caption}'. Keeping yours for now.")
            #           captions_kept_original_in_my_data += 1
            else:  # 你的caption有效，保留你的
                captions_kept_original_in_my_data += 1
        else:  # GMP没有有效caption，保留你的原始caption
            captions_kept_original_in_my_data += 1
    else:  # GMP数据中没有这个image_id，保留你的原始caption
        captions_kept_original_in_my_data += 1

    # 确保score字段是你原始的
    # new_my_item['score'] 已经是你原始的了，因为是从my_item_content复制的

    my_data_captions_updated.append(new_my_item)

print(f"\n--- Your Data with Captions Updated/Verified ---")
print(f"Total items in my_data_captions_updated: {len(my_data_captions_updated)}")
print(f"Captions in your data updated/filled by GMP's caption: {captions_updated_in_my_data}")
print(
    f"Captions in your data kept original (yours was valid or GMP had no valid caption): {captions_kept_original_in_my_data}")


try:
    with open(output_gmp_with_scores_path, "w", encoding="utf-8") as f:
        json.dump(gmp_data_with_scores, f, indent=4)
    print(f"\nGMP data with scores saved to {output_gmp_with_scores_path}")
except Exception as e:
    print(f"Error saving gmp_data_with_scores: {e}")

try:
    with open(output_my_data_updated_path, "w", encoding="utf-8") as f:
        json.dump(my_data_captions_updated, f, indent=4)
    print(f"Your data with updated captions saved to {output_my_data_updated_path}")
except Exception as e:
    print(f"Error saving my_data_captions_updated: {e}")
