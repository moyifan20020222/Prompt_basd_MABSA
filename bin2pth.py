# from transformers import AutoModel, ResNetForImageClassification, AutoConfig, ResNetConfig
# import torch
#
# # 从 Hugging Face 加载模型（示例）
# model = ResNetForImageClassification.from_pretrained("data/nf_Resnet50")
# print(model)
# config = ResNetConfig.from_pretrained("data/nf_Resnet50")
# print(config.model_type)
# # 保存为 PyTorch 格式
# output_path = "data/nf_Resnet50/nf_resnet50.pth"
# torch.save(model.state_dict(), output_path)
import torch
from safetensors import safe_open
import timm
import os

from src.model.modeling_bart import BartModel

# # 设置镜像站（可选）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# # 1. 加载模型架构（移除分类头）
# model = timm.create_model("nf_resnet50", pretrained=False, num_classes=0)
#
# # 2. 验证文件存在性
# checkpoint_path = r"data/nf_Resnet50/model.safetensors"
# assert os.path.exists(checkpoint_path), f"文件不存在: {checkpoint_path}"
#
# # 3. 使用 safetensors 读取权重
# with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
#     state_dict = {k: f.get_tensor(k) for k in f.keys()}
#
# # 4. 过滤分类头参数（可选）
# filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.fc")}
#
# # 5. 加载权重
# model.load_state_dict(filtered_state_dict, strict=False)
# # print(model)
# print("模型加载成功！")
#
# model.eval()  # 设置为评估模式
#
# # 生成随机测试图像（模拟输入）
# fake_image = torch.randn(1, 3, 256, 256)  # (batch, channels, height, width)
#
# # 前向传播
# with torch.no_grad():
#     features = model(fake_image)
#
# # 验证输出
# print("特征形状:", features.shape)  # 应输出 torch.Size([1, 2048])
# print("特征示例:", features[0, :5])  # 应输出非全零的浮点数
#
# import torch
# from PIL import Image
# from torchvision import transforms
# import timm
#
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 加载测试图像
# image_path = r"data/test_img.jpg"
# img = Image.open(image_path).convert('RGB')
# input_tensor = transform(img).unsqueeze(0)  # (1, 3, 256, 256)
#
# # 获取特征
# with torch.no_grad():
#     features = model(input_tensor)
#
# # 验证结果
# print("特征范数:", torch.norm(features))  # 应输出10-100之间的数值
# print("特征形状:", features.shape)        # 应输出 torch.Size([1, 2048])
#
# print("@@@@@@@@@@@")
# from transformers import BartConfig
#
# config = BartConfig()
# print(hasattr(config, "static_position_embeddings"))  # 旧版本应为 True，新版本应为 False
#
#
# from safetensors import safe_open
#
# local_checkpoint_path = "D:\\Desktop\\研一内容\\论文对应代码\\MPLMM-main (处理多模态缺失)\\vit_base_patch32_224\\model.safetensors" #  <---  **替换为你的路径**
#
# try:
#     with safe_open(local_checkpoint_path, framework="pt", device="cpu") as f:
#         keys = f.keys()
#         print(f"成功使用 safe_open 打开 safetensors 文件！")
#         print("safetensors 文件中的键:", keys) # 打印键以检查是否加载了 *内容*
# except Exception as e:
#     print(f"使用 safe_open 打开 safetensors 文件时出错: {e}")
#     print("safetensors 文件本身或路径有问题。")
#
#
# import timm
#
# image_model_name = 'vit_base_patch32_224'
# image_encoder_timm_arch = timm.create_model(image_model_name, pretrained=False, num_classes=0)
# expected_keys = image_encoder_timm_arch.state_dict().keys()
# print("来自 timm 的 vit_base_patch32_224 架构的预期键:")
# for key in expected_keys:
#     print(key)
#
# import timm
#
# image_model_name = 'vit_base_patch32_224'
# try:
#     image_encoder_online = timm.create_model(image_model_name, pretrained=True, num_classes=0)
#     print(f"成功在线加载 timm 模型: {image_model_name}")
# except Exception as e:
#     print(f"在线加载 timm 模型 {image_model_name} 失败: {e}")


# from src.model.modeling_bart import BartModel
#
# model = BartModel.from_pretrained("data/bart-base")
#
#
# encoder = model.encoder
#
# print("Encoder object:", encoder)
# print("Attributes of encoder:", dir(encoder))
#
# print("Encoder config object:", len(encoder.layers))
#
# print("选择特定层",encoder.embed_tokens)



from PIL import Image
#
# image_path = "data/IJCAI2019_data/twitter2015_images/245131.jpg" # <--- 替换为你的图片路径
#
# try:
#     img = Image.open(image_path)
#     img.verify() # 验证文件完整性
#     print(f"使用 Pillow 成功打开并验证图片: {image_path}")
# except Exception as e:
#     print(f"使用 Pillow 打开或验证图片时出错: {image_path}, 错误: {e}")

# from collections import defaultdict
#
# def create_C1(transactions):
#     """
#     构建候选1项集 C1
#     transactions: 事务列表，每个事务是一个项的集合
#     """
#     C1 = []
#     for transaction in transactions:
#         for item in transaction:
#             if [item] not in C1: # 避免重复添加
#                 C1.append([item])
#     C1.sort() # 排序，方便后续处理
#     return list(map(frozenset, C1)) # 将每个项集转换为 frozenset，因为 frozenset 可以作为字典的键
#
# def scan_D(D, Ck, min_support):
#     """
#     从候选k项集 Ck 中生成频繁k项集 Lk，并计算支持度
#     D: 事务数据集
#     Ck: 候选k项集
#     min_support: 最小支持度阈值 (例如 0.5 表示 50%)
#     """
#     support_count = defaultdict(int) # 使用 defaultdict 存储项集计数，默认值为 0
#     for tid in D: # 遍历每个事务
#         for can in Ck: # 遍历每个候选k项集
#             if can.issubset(tid): # 检查候选集是否是事务的子集
#                 support_count[can] += 1 # 如果是子集，计数加1
#
#     num_items = float(len(D)) # 事务总数
#     Lk = [] # 频繁k项集列表
#     support_data = {} # 存储每个频繁项集的支持度
#
#     for key in support_count: # 遍历计数
#         support = support_count[key] / num_items # 计算支持度 (相对支持度)
#         if support >= min_support: # 判断是否满足最小支持度
#             Lk.insert(0, key) # 将频繁项集添加到 Lk 列表的头部 (可以保持项集按支持度降序排列，虽然这里不是严格排序，但可以提高效率)
#             support_data[key] = support # 存储支持度
#     return Lk, support_data
#
# def apriori_gen(Lk_prev, k):
#     """
#     基于频繁k-1项集 Lk_prev 生成候选k项集 Ck
#     Lk_prev: 频繁k-1项集
#     k: 当前要生成的项集大小 k
#     """
#     Ck = set() # 使用 set 避免重复
#     len_Lk_prev = len(Lk_prev)
#     for i in range(len_Lk_prev):
#         for j in range(i+1, len_Lk_prev): # 两两组合
#             L1 = list(Lk_prev[i])[:k-2] # 取前 k-2 个项
#             L2 = list(Lk_prev[j])[:k-2]
#             L1.sort()
#             L2.sort()
#             if L1 == L2: # 如果前 k-2 个项相同，则可以连接
#                 Ck.add(Lk_prev[i] | Lk_prev[j]) # 合并生成候选k项集
#     return list(Ck) # 将 set 转换为 list
#
# def apriori(transactions, min_support=0.5):
#     """
#     Apriori 算法主函数
#     transactions: 事务数据集
#     min_support: 最小支持度阈值
#     """
#     C1 = create_C1(transactions) # 构建候选1项集
#     D = list(map(set, transactions)) # 将事务列表转换为 set 列表，方便后续子集判断
#     L1, support_data = scan_D(D, C1, min_support) # 生成频繁1项集 L1 和支持度数据
#     L = [L1] # L 存储所有频繁项集，L[k-1] 存储频繁k项集
#     k = 2
#     while(len(L[k-2]) > 0): # 当还存在频繁k-1项集时，继续迭代
#         Ck = apriori_gen(L[k-2], k) # 生成候选k项集
#         Lk, sup_k = scan_D(D, Ck, min_support) # 生成频繁k项集和支持度
#         support_data.update(sup_k) # 更新支持度数据
#         L.append(Lk) # 将频繁k项集添加到 L
#         k += 1
#     return L, support_data
#
# def generate_rules(L, support_data, min_confidence=0.7):
#     """
#     从频繁项集中生成关联规则
#     L: 频繁项集列表，L[i] 表示频繁 (i+1) 项集
#     support_data: 频繁项集的支持度字典
#     min_confidence: 最小置信度阈值
#     """
#     rule_list = []
#     for i in range(1, len(L)): # 从频繁2项集开始
#         for freq_set in L[i]:
#             H1 = [frozenset([item]) for item in freq_set] # 初始规则后件为单个项
#             if i > 1:
#                 rules_from_conseq(freq_set, H1, support_data, rule_list, min_confidence)
#             else:
#                 calc_confidence(freq_set, H1, support_data, rule_list, min_confidence)
#     return rule_list
#
# def rules_from_conseq(freq_set, H, support_data, rule_list, min_confidence):
#     """
#     递归生成关联规则
#     freq_set: 频繁项集
#     H: 规则后件候选集
#     support_data: 频繁项集的支持度字典
#     rule_list: 规则列表
#     min_confidence: 最小置信度阈值
#     """
#     m = len(H[0])
#     if (len(freq_set) > (m + 1)): # 可以继续生成更长的后件
#         H_next = apriori_gen(H, m+1) # 生成新的后件候选集
#         H_next = calc_confidence(freq_set, H_next, support_data, rule_list, min_confidence)
#         if (len(H_next) > 1): # 如果还有满足置信度的后件，则递归调用
#             rules_from_conseq(freq_set, H_next, support_data, rule_list, min_confidence)
#
# def calc_confidence(freq_set, H, support_data, rule_list, min_confidence):
#     """
#     计算规则置信度，并筛选满足最小置信度的规则
#     freq_set: 频繁项集 (规则前件+后件)
#     H: 规则后件候选集
#     support_data: 频繁项集的支持度字典
#     rule_list: 规则列表
#     min_confidence: 最小置信度阈值
#     """
#     pruned_H = [] # 存储满足置信度的后件
#     for conseq in H:
#         conf = support_data[freq_set] / support_data[freq_set - conseq] # 计算置信度
#         if conf >= min_confidence:
#             print(f"{tuple(freq_set - conseq)} --> {tuple(conseq)} confidence:{conf:.2f}") # 打印规则
#             rule_list.append(((freq_set - conseq), conseq, conf)) # 存储规则 (前件, 后件, 置信度)
#             pruned_H.append(conseq)
#     return pruned_H
#
#
# # 示例数据
# transactions = [
#     ['A', 'B', 'C', 'D'],
#     ['B', 'C', 'E'],
#     ['A', 'B', 'C', 'E'],
#     ['B', 'D', 'E']
# ]
#
# min_support = 0.5  # 最小支持度阈值 50%
# min_confidence = 0.7 # 最小置信度阈值 70%
#
# # 运行 Apriori 算法
# L, support_data = apriori(transactions, min_support)
# print("\n频繁项集 L:")
# for i in range(len(L)):
#     print(f"L{i+1}: {L[i]}")
#
# print("\n频繁项集及其支持度 support_data:")
# for key in support_data:
#     print(f"{tuple(key)}: {support_data[key]:.2f}")
#
# # 生成关联规则
# print("\n关联规则 (confidence >= 0.7):")
# rules = generate_rules(L, support_data, min_confidence)
# print("\n生成的规则列表 (rule_list):")
# for rule in rules:
#     antecedent, consequent, confidence = rule
#     print(f"规则: {tuple(antecedent)} --> {tuple(consequent)}, 置信度: {confidence:.2f}")


from senticnet.senticnet import SenticNet
import re

# 创建 SenticNet 实例
sn = SenticNet()

# 定义一些测试用的词语或概念
test_concepts = [
    "love",
    "happy",
    "beautiful",
    "good",
    "excellent",
    "hate",
    "sad",
    "ugly",
    "bad",
    "terrible",
    "war",
    "peace",
    "computer",       # 通常是中性的
    "buy_gift",       # SenticNet 的概念通常用下划线连接多词
    "lose_weight",
    "not_good",       # SenticNet可能不直接处理否定，其概念是预定义的
    "very_happy",     # SenticNet可能不直接处理程度副词
    "a_little_sad"
]

print("--- Testing SenticNet Polarity ---")
for concept_str in test_concepts:
    print(f"\nConcept: '{concept_str}'")
    try:
        # 获取概念的完整信息
        concept_info = sn.concept(concept_str)
        # concept()方法内部会调用 polarity_value() 和 polarity_label()

        polarity_value_str = concept_info.get("polarity_value") # 从字典获取
        polarity_label = concept_info.get("polarity_label")

        if polarity_value_str is not None:
            polarity_value = float(polarity_value_str) # SenticNet 返回的是字符串形式的数值
            print(f"  Polarity Value: {polarity_value:.4f}")
            print(f"  Polarity Label: {polarity_label}")

            # 我们可以根据极性值给出一个更直观的判断
            if polarity_value > 0.1: # 阈值可以调整
                print(f"  Interpreted Sentiment: Positive")
            elif polarity_value < -0.1: # 阈值可以调整
                print(f"  Interpreted Sentiment: Negative")
            else:
                print(f"  Interpreted Sentiment: Neutral")

        else:
            print(f"  Could not retrieve polarity value for '{concept_str}'.")

        # 你也可以测试其他信息
        # sentics = concept_info.get("sentics")
        # if sentics:
        #     print(f"  Sentics: {sentics}")

    except KeyError:
        # KeyError 意味着这个概念（或者其下划线形式）不在 SenticNet 的数据中
        print(f"  Concept '{concept_str}' (or its formatted version) not found in SenticNet data.")
    except Exception as e:
        print(f"  An error occurred while processing '{concept_str}': {e}")

print("\n--- Test Finished ---")

# 额外测试，如果你的SenticNet类中的方法是直接可用的：
print("\n--- Direct Method Call Test (if applicable) ---")
try:
    concept_to_test_direct = "joy"
    value = sn.polarity_value(concept_to_test_direct)
    label = sn.polarity_label(concept_to_test_direct)
    print(f"Direct call for '{concept_to_test_direct}': Value='{value}', Label='{label}'")
    if value is not None:
        print(f"  Interpreted Sentiment: {'Positive' if float(value) > 0.1 else ('Negative' if float(value) < -0.1 else 'Neutral')}")
except KeyError:
    print(f"Direct call for '{concept_to_test_direct}': Concept not found.")
except Exception as e:
    print(f"Direct call for '{concept_to_test_direct}': Error - {e}")

import spacy
spacy.load("en_core_web_sm")


from senticnet.senticnet import SenticNet # 确保类名正确

sn = SenticNet()

# 定义否定词列表 (可以扩展)
negation_words = {"not", "no", "never", "n't", "isnt", "arent", "dont", "doesnt", "didnt", "wasnt", "werent"}

def get_word_sentiment_with_negation_check(word, previous_word=None):
    """
    获取单个词的SenticNet情感极性，并进行简单的否定检查。
    """
    polarity_value = None
    polarity_label = "neutral" # 默认
    original_polarity_value_for_debug = None

    try:
        # SenticNet通常对小写、下划线连接的格式更友好
        formatted_word = word.lower().replace(" ", "_")
        concept_info = sn.concept(formatted_word)
        # print(f"Debug: Querying '{formatted_word}', Info: {concept_info}") # 调试用

        if concept_info and 'polarity_value' in concept_info and concept_info['polarity_value'] is not None:
            polarity_value_str = concept_info['polarity_value']
            original_polarity_value_for_debug = float(polarity_value_str)
            polarity_value = float(polarity_value_str)
            polarity_label = concept_info.get('polarity_label', 'neutral')

            # 简单否定检查
            if previous_word and previous_word.lower() in negation_words:
                polarity_value = -polarity_value # 反转极性
                # 相应地调整标签 (简化)
                if polarity_value > 0:
                    polarity_label = "positive (negated)"
                elif polarity_value < 0:
                    polarity_label = "negative (negated)"
                else:
                    polarity_label = "neutral (negated)"
        else:
            # 如果词本身找不到，可以尝试不替换空格（如果SenticNet库支持）
            # 或者尝试其词根（lemmatization），但这会增加复杂性
            pass


    except KeyError:
        # print(f"Debug: Concept '{formatted_word}' not found in SenticNet data.") # 调试用
        pass # 概念未找到，polarity_value 保持为 None
    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        pass

    return polarity_value, polarity_label, original_polarity_value_for_debug


# --- 测试句子 ---
test_sentences = [
    "This is a very good and beautiful product, I love it.",
    "I am not happy with the terrible service.",
    "The food was delicious, although the service is poor.",
    "This movie is not bad at all.", # 双重否定或复杂否定，简单规则可能失效
    "The war situation is dire." # "war" 和 "dire" 可能不在SenticNet中
]

print("\n--- Testing Sentiment for Words in Sentences (with simple negation check) ---")
for sentence in test_sentences:
    print(f"\nSentence: \"{sentence}\"")
    # 简单分词 (实际应用中你会用BART的分词器)
    words = sentence.lower().replace('.', '').replace(',', '').split()
    previous_word_token = None
    for word_token in words:
        # 实际应用中，word_token 会是BART tokenizer处理后的token
        # 你需要将subword组合回完整的词，或者将词的情感传播给其所有subword
        # 这里为了演示，我们假设 word_token 就是一个完整的词

        p_value, p_label, orig_p_value = get_word_sentiment_with_negation_check(word_token, previous_word_token)

        if p_value is not None:
            interpretation = "Neutral"
            if p_value > 0.1: interpretation = "Positive"
            elif p_value < -0.1: interpretation = "Negative"
            print(f"  Word: '{word_token}' -> Polarity Value: {p_value:.4f} (Original: {orig_p_value if orig_p_value is not None else 'N/A'}), Label: {p_label}, Interpreted: {interpretation}")
        else:
            print(f"  Word: '{word_token}' -> SenticNet data not found or no polarity.")
        previous_word_token = word_token
