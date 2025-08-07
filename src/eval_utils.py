import numpy as np
import torch
import torch.nn as nn


def eval(args, model, loader, metric, device):
    model.eval()
    # 在分类下构建新的指标计算

    confusion_matrix = np.zeros((3, 3), dtype=int)
    all_num = 0
    all_right = 0
    tp = np.zeros(3, dtype=int)
    fp = np.zeros(3, dtype=int)
    fn = np.zeros(3, dtype=int)
    all_y_true = []
    all_y_scores = []
    for i, batch in enumerate(loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
        # import ipdb; ipdb.set_trace()
        predict, senti_label, sentiment_labels_mapped, y_scores_list_flat = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos, 
            aspects_num=batch['aspects_num'],
            sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
            image_mask=batch['my_image_mask'],
            mlm_message=batch['MLM'],
            image_caption_valid=batch['image_caption_valid'],
            image_caption_mask=batch['image_caption_mask'],
            score=batch['score'],
            caption_nouns=batch['caption_nouns'],
            sentence_nouns=batch['sentence_nouns']
        )
        # 返回结果为序列生成的结果。
        # print('predict is {}'.format(predict))

        if args.is_classifier:
            for idx in range(len(senti_label)):
                true_label = sentiment_labels_mapped[idx]
                pred_label = senti_label[idx]
                confusion_matrix[true_label][pred_label] += 1
                all_num += 1
                if true_label == pred_label:
                    tp[pred_label] += 1
                    all_right += 1
                else:
                    fp[pred_label] += 1
                    fn[true_label] += 1

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))

        batch_logits_tensor = torch.tensor(y_scores_list_flat,
                                           device=sentiment_labels_mapped.device)  # Shape: [num_aspects_in_batch, 3]

        # ii. 计算概率分数
        batch_scores_tensor = torch.softmax(batch_logits_tensor, dim=1)

        # iii. 将计算好的分数和真实标签转换为Python list
        batch_scores_list_processed = batch_scores_tensor.cpu().numpy().tolist()
        batch_true_list_processed = sentiment_labels_mapped.cpu().numpy().tolist()

        # e. *** 使用 .extend() 将当前批次的结果追加到总列表中 ***
        all_y_true.extend(batch_true_list_processed)
        all_y_scores.extend(batch_scores_list_processed)

        # break
    mAP, class_aps = plot_pr_curve_and_map(all_y_true, all_y_scores)
    print(mAP)
    Micro_f1, Micro_rec, Micro_pre, acc = 0.0, 0.0, 0.0, 0.0
    if args.is_classifier:
        pos_tp, pos_fp, pos_fn = tp[0], fp[0], fn[0]
        pos_pre = pos_tp / (pos_fp + pos_tp + 1e-13)
        pos_rec = pos_tp / (pos_fn + pos_tp + 1e-13)
        pos_f1 = round(2 * pos_rec * pos_pre / (pos_pre + pos_rec + 1e-13) * 100, 2)

        neu_tp, neu_fp, neu_fn = tp[1], fp[1], fn[1]
        neu_pre = neu_tp / (neu_fp + neu_tp + 1e-13)
        neu_rec = neu_tp / (neu_fn + neu_tp + 1e-13)
        neu_f1 = round(2 * neu_rec * neu_pre / (neu_pre + neu_rec + 1e-13) * 100, 2)

        neg_tp, neg_fp, neg_fn = tp[2], fp[2], fn[2]
        neg_pre = neg_tp / (neg_fp + neg_tp + 1e-13)
        neg_rec = neg_tp / (neg_fn + neg_tp + 1e-13)
        neg_f1 = round(2 * neg_rec * neg_pre / (neg_pre + neg_rec + 1e-13) * 100, 2)
        print("各个类别的F1值 积极 中性 消极", pos_f1, neu_f1, neg_f1)
        avg_pre = (pos_pre + neu_pre + neg_pre) / 3
        avg_rec = (pos_rec + neu_rec + neg_rec) / 3

        Micro_f1 = round(2 * avg_pre * avg_rec / (avg_rec + avg_pre + 1e-12) * 100, 2)
        print("message", avg_pre, avg_rec, Micro_f1)
        # round(2 * avg_pre_custom * avg_rec_custom / (avg_pre_custom + avg_rec_custom + 1e-12) * 100, 2)
        Micro_pre = round(avg_pre * 100, 2)
        Micro_rec = round(avg_rec * 100, 2)
        acc = round(1.0 * all_right / (all_num + 1e-12) * 100, 2)
    res = metric.get_metric()
    res_classifer = {}
    res_classifer['sc_f'] = Micro_f1
    res_classifer['sc_pre'] = Micro_pre
    res_classifer['sc_rec'] = Micro_rec
    res_classifer['sc_acc'] = acc
    res_classifer['confusion_matrix'] = confusion_matrix
    model.train()
    return res, res_classifer, mAP

def plot_pr_curve_and_map(y_true, y_scores):
    """
    绘制多分类任务的PR曲线，并计算每个类别的AP和mAP。
    此版本已适配PyTorch Tensor和Python列表作为输入。

    Args:
        y_true (list or np.array or torch.Tensor): 真实标签列表，一维。
        y_scores (list of lists or np.array or torch.Tensor): 预测分数列表，形状为 [n_samples, n_classes]。
        class_names (list of str): 类别名称列表，例如 ['Positive', 'Neutral', 'Negative']。
        save_path (str, optional): 图像保存路径。如果为None，则直接显示图像。
        title_suffix (str, optional): 添加到图像标题的后缀，例如模型名称或epoch号。
    """

    num_classes = 3  # 你的类别数
    class_names = ['Positive', 'Neutral', 'Negative']

    # 调用新的、无需sklearn的计算函数
    mAP, class_aps = calculate_map_without_sklearn(
        y_true,
        y_scores,
        num_classes=num_classes
    )

    # 打印结果
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    for i, class_name in enumerate(class_names):
        # .get(i, 0.0) 是为了安全地处理某个类别可能没有AP的情况
        print(f"  - AP for {class_name}: {class_aps.get(i, 0.0):.4f}")
    return mAP, class_aps

def calculate_map_without_sklearn(y_true, y_scores, num_classes):
    """
    计算mAP (mean Average Precision)，不依赖任何外部包（除了PyTorch）。
    此版本已适配Python列表和PyTorch Tensor作为输入。

    Args:
        y_true (list, np.array, or torch.Tensor): 一维的真实标签。
        y_scores (list of lists, np.array, or torch.Tensor): 预测分数，形状 [n_samples, n_classes]。
        num_classes (int): 类别的总数。

    Returns:
        float: mAP (mean Average Precision) 的值。
        dict: 每个类别的AP值。
    """
    # --- 1. 将输入统一转换为 PyTorch Tensor ---
    # 检查 y_true 的类型
    if not isinstance(y_true, torch.Tensor):
        # 假设是 list 或 numpy array
        y_true = torch.tensor(y_true, dtype=torch.long)

    # 检查 y_scores 的类型
    if not isinstance(y_scores, torch.Tensor):
        # 假设是 list of lists 或 numpy array
        y_scores = torch.tensor(y_scores, dtype=torch.float)

    # 确保张量在同一个设备上，以y_true为准
    device = y_true.device
    y_scores = y_scores.to(device)

    # 确保输入不为空
    if y_true.numel() == 0 or y_scores.numel() == 0:
        print("Warning: y_true or y_scores is empty. Cannot calculate mAP.")
        return 0.0, {c: 0.0 for c in range(num_classes)}

    average_precisions = {}

    # 遍历每个类别，采用"一对多"(One-vs-Rest)策略计算AP
    for c in range(num_classes):
        # 2. 准备当前类别的真实标签和预测分数
        y_true_c = (y_true == c).int()
        y_scores_c = y_scores[:, c]

        total_positives = y_true_c.sum()
        if total_positives == 0:
            average_precisions[c] = 0.0
            continue

        # 3. 按预测分数从高到低排序
        indices = torch.argsort(y_scores_c, descending=True)
        y_true_c_sorted = y_true_c[indices]

        # 4. 计算Precision和Recall列表
        tp_cumulative = torch.cumsum(y_true_c_sorted, dim=0)
        # (TP + FP) 就是当前扫描到的样本数量 (从1开始)
        precision_points = tp_cumulative / (torch.arange(len(y_true_c_sorted), device=device) + 1)
        recall_points = tp_cumulative / total_positives

        # 只在正样本的位置计算precision，这是计算AP的标准方法
        precision_at_recalls = precision_points[y_true_c_sorted == 1]

        # 如果没有正样本被召回，AP为0
        if precision_at_recalls.numel() == 0:
            ap = 0.0
        else:
            ap = precision_at_recalls.mean().item()

        average_precisions[c] = ap

    # 6. 计算mAP (mean Average Precision)
    # 只对在数据中实际出现过的类别计算平均值
    unique_classes = torch.unique(y_true)
    valid_aps = [ap for c, ap in average_precisions.items() if c in unique_classes]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

    return mAP, average_precisions

