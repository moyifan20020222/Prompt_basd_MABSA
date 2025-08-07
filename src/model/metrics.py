from collections import Counter
import numpy as np
import torch


class AESCSpanMetric(object):
    def __init__(self,
                 eos_token_id,
                 num_labels,
                 conflict_id,
                 opinion_first=False):
        super(AESCSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels + 2

        self.aesc_fp = 0
        self.aesc_tp = 0
        self.aesc_fn = 0
        self.ae_fp = 0
        self.ae_tp = 0
        self.ae_fn = 0
        self.sc_fp = Counter()
        self.sc_tp = Counter()
        self.sc_fn = Counter()
        self.sc_right = 0
        self.sc_all_num = 0

        self.em = 0
        self.total = 0
        self.invalid = 0
        self.conflict_id = conflict_id
        # assert opinion_first is False, "Current metric only supports aspect first"
        # 混淆矩阵初始化
        self.confusion_matrix = np.zeros((3, 3), dtype=int)

    def evaluate(self, aesc_target_span, pred, tgt_tokens):
        '''
        aesc_target_span [[7, 8, 3]]
        原始数据信息 tensor([0, 2, 2, 1, 1, 3, 1, 1, 1, 1], device='cuda:0')
        tgt_tokens tensor([0, 2, 2, 7, 8, 3, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
        '''
        # print('aesc_target_span', aesc_target_span[0])
        # print("原始数据信息", pred[0])
        # print('tgt_tokens', tgt_tokens[0])
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(
            pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(
            target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        flag = True
        for i, (ts, ps) in enumerate(zip(aesc_target_span, pred.tolist())):
            # print("原始数据信息", ps)
            em = 0
            assert ps[0] == tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(
                    pred[i, :target_seq_len[i]]).sum().item() ==
                         target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair) != 3 or cur_pair[0] > cur_pair[1]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())

            # print(pred_spans)
            self.invalid += invalid

            aesc_target_counter = Counter()
            aesc_pred_counter = Counter()
            ae_target_counter = Counter()
            ae_pred_counter = Counter()
            conflicts = set()
            # if flag:
            #     print(tgt_tokens[0])
            #     print(pred[0])
            #     print(ts)
            #     print(pairs)
            #     flag = False
            for t in ts:
                ae_target_counter[(t[0], t[1])] = 1
                if t[2] != self.conflict_id:
                    aesc_target_counter[(t[0], t[1])] = t[2]
                else:
                    conflicts.add((t[0], t[1]))

            for p in pairs:
                ae_pred_counter[(p[0], p[1])] = 1
                if (p[0], p[1]) not in conflicts and p[-1] not in (
                        0, 1, self.conflict_id):
                    aesc_pred_counter[(p[0], p[1])] = p[-1]

            # 这里相同的pair会被计算多次 一个pair的结构就是三元组， 首索引 尾索引和情绪极性
            tp, fn, fp = _compute_tp_fn_fp(
                [(key[0], key[1], value)
                 for key, value in aesc_pred_counter.items()],
                [(key[0], key[1], value)
                 for key, value in aesc_target_counter.items()])
            self.aesc_fn += fn
            self.aesc_fp += fp
            self.aesc_tp += tp
            # print("预测结果", [(key[0], key[1], value)
            #      for key, value in aesc_pred_counter.items()])
            # print("真实结果", [(key[0], key[1], value)
            #                    for key, value in aesc_target_counter.items()])

            tp, fn, fp = _compute_tp_fn_fp(list(aesc_pred_counter.keys()),
                                           list(aesc_target_counter.keys()))
            self.ae_fn += fn
            self.ae_fp += fp
            self.ae_tp += tp

            # sorry, this is a very wrongdoing, but to make it comparable with previous work, we have to stick to the
            #   error
            # SC 任务的统计和混淆矩阵填充 只计算识别成功后，再识别情绪
            # 这里的 for 循环只考虑预测的 Aspect Span 成功匹配到真实 Aspect Span 的情况
            for key in aesc_pred_counter:  # key 是 (span_start, span_end)
                if key not in aesc_target_counter:  # 如果预测的 Aspect Span 没有对应的真实 Span
                    continue  # 则跳过，不计入 SC 评估和混淆矩阵

                # 获得真实和预测的情感 ID
                true_sentiment_id = aesc_target_counter[key]
                predicted_sentiment_id = aesc_pred_counter[key]
                # print("id", true_sentiment_id, predicted_sentiment_id)
                # 确保这些情感 ID 是我们关注的有效分类类别
                # if true_sentiment_id in self.sentiment_id_to_idx and \
                #         predicted_sentiment_id in self.sentiment_id_to_idx:
                #     true_idx = self.sentiment_id_to_idx[true_sentiment_id]
                #     pred_idx = self.sentiment_id_to_idx[predicted_sentiment_id]

                # **核心：更新混淆矩阵**
                self.confusion_matrix[true_sentiment_id - 3][predicted_sentiment_id - 3] += 1

                self.sc_all_num += 1  # 统计 SC 评估的总次数
                if true_sentiment_id == predicted_sentiment_id:
                    self.sc_tp[predicted_sentiment_id] += 1  # 统计正确预测的类别
                    self.sc_right += 1  # 统计总的正确预测数
                    # aesc_target_counter.pop(key) # 这一行可能会影响后续对 aesc_target_counter 的使用，
                    # 如果没有其他地方用到，可以保留，否则建议删除或谨慎处理
                else:
                    self.sc_fp[predicted_sentiment_id] += 1  # 统计预测为该类别的FP
                    self.sc_fn[true_sentiment_id] += 1  # 统计真实为该类别的FN

            # for key in aesc_pred_counter:
            #     if key not in aesc_target_counter:
            #         continue
            #     self.sc_all_num += 1
            #     if aesc_target_counter[key] == aesc_pred_counter[key]:
            #         self.sc_tp[aesc_pred_counter[key]] += 1
            #         self.sc_right += 1
            #         aesc_target_counter.pop(key)
            #     else:
            #         self.sc_fp[aesc_pred_counter[key]] += 1
            #         self.sc_fn[aesc_target_counter[key]] += 1

    def pri(self):
        print('aesc_fp tp fn', self.aesc_fp, self.aesc_tp, self.aesc_fn)
        print('ae_fp tp fn', self.ae_fp, self.ae_tp, self.ae_fn)

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.aesc_tp, self.aesc_fn,
                                         self.aesc_fp)
        res['aesc_f'] = round(f * 100, 2)
        res['aesc_rec'] = round(rec * 100, 2)
        res['aesc_pre'] = round(pre * 100, 2)

        f, pre, rec = _compute_f_pre_rec(1, self.ae_tp, self.ae_fn, self.ae_fp)
        res['ae_f'] = round(f * 100, 2)
        res['ae_rec'] = round(rec * 100, 2)
        res['ae_pre'] = round(pre * 100, 2)

        tags = set(self.sc_tp.keys())
        tags.update(set(self.sc_fp.keys()))
        tags.update(set(self.sc_fn.keys()))
        f_sum = 0
        pre_sum = 0
        rec_sum = 0
        # ----
        # 计算一下加权的F1值，给不平衡的MASC任务一个好一点的指标
        # 用于 Macro-F1 计算 (你现有的逻辑)
        macro_f_sum = 0  # 未直接使用，但可以保留用于计算另一种 Macro-F1
        macro_pre_sum = 0
        macro_rec_sum = 0

        # 用于 Weighted-F1 计算
        weighted_f1_numerator = 0
        total_support_for_sc = 0  # 所有类别的总支持度 (用于加权平均)

        valid_tags_count = 0  # 计算有效参与平均的类别数量

        # ---
        for tag in tags:
            assert tag not in (0, 1, self.conflict_id), (tag, self.conflict_id)
            tp = self.sc_tp[tag]
            fn = self.sc_fn[tag]
            fp = self.sc_fp[tag]
            # 计算当前类别的支持度 (真实样本中该类别的数量)

            f, pre, rec = _compute_f_pre_rec(1, tp, fn, fp)
            # ----
            # 计算一下加权的F1值，给不平衡的MASC任务一个好一点的指标
            support_tag = tp + fn
            if support_tag > 0:  # 只对有真实样本的类别进行累加，避免干扰平均值
                # Macro-F1 的累加 (你现有的)
                macro_pre_sum += pre
                macro_rec_sum += rec
                valid_tags_count += 1  # 只有当一个类别有真实样本时，它才应该参与宏平均的分母

                # Weighted-F1 的累加
                weighted_f1_numerator += (f * support_tag)
                total_support_for_sc += support_tag
            # -------
            f_sum += f
            pre_sum += pre
            rec_sum += rec

        rec_sum /= (len(tags) + 1e-12)
        pre_sum /= (len(tags) + 1e-12)
        res['sc_f'] = round(
            2 * pre_sum * rec_sum / (pre_sum + rec_sum + 1e-12) * 100, 2)
        res['sc_rec'] = round(rec_sum * 100, 2)
        res['sc_pre'] = round(pre_sum * 100, 2)
        res['sc_acc'] = round(
            1.0 * self.sc_right / (self.sc_all_num + 1e-12) * 100, 2)
        res['sc_all_num'] = self.sc_all_num
        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)

        # --- SC 指标计算 ---
        tags = set(self.sc_tp.keys())
        tags.update(set(self.sc_fp.keys()))
        tags.update(set(self.sc_fn.keys()))

        # 用于 Macro-average 和您自定义 F1 计算的累加器
        # f_sum_for_custom_f1 = 0 # 您原始代码中的 f_sum
        pre_sum_for_custom_f1 = 0
        rec_sum_for_custom_f1 = 0

        # 用于标准的 Macro-average F1, P, R 计算的累加器
        per_class_f1_scores_list = []
        per_class_precisions_list = []
        per_class_recalls_list = []

        # 用于 Weighted-average F1 计算的累加器
        weighted_f1_numerator = 0
        total_support_for_sc = 0  # 所有有真实样本的类别的总支持度

        # 打印混淆矩阵 (如果希望在 get_metric 中打印)
        print("\nSentiment Classification Confusion Matrix (from AESCSpanMetric):")
        print(self.confusion_matrix)  # 假设 confusion_matrix 是最新的

        print("\n--- Per-Class Sentiment Metrics (from sc_tp/fp/fn Counters) ---")
        for tag in sorted(list(tags)):  # 对 tags 进行排序，确保输出顺序一致
            # 您的断言，确保 tag 是有效的情感 ID (例如 3, 4, 5)
            # 您需要根据实际情况调整 (0, 1, self.conflict_id)
            # 如果您的有效情感 ID 就是 3,4,5，那么这个断言可能需要调整
            # 例如，如果 self.conflict_id 不是情感标签，但 0,1 是，那这里就有问题
            # 假设您的 tags 集合已经只包含有效的情感 ID
            # assert tag not in (0, 1, self.conflict_id), (f"Tag: {tag}, Conflict ID: {self.conflict_id}")

            tp = self.sc_tp.get(tag, 0)  # 使用 .get(tag, 0) 避免 KeyError
            fn = self.sc_fn.get(tag, 0)
            fp = self.sc_fp.get(tag, 0)

            f, pre, rec = _compute_f_pre_rec(1, tp, fn, fp)  # 计算当前类别的 P, R, F1

            # **【新增】保存每个类别的 P, R, F1 到结果字典**
            # 您需要一种方式将 tag (例如 3, 4, 5) 映射到可读的类别名称 (POS, NEU, NEG)
            # 假设您有一个 sentiment_id_to_name_map = {3: 'POS', 4: 'NEU', 5: 'NEG'}
            # 如果没有，就直接用 tag ID 作为 key
            sentiment_name = f"Class_{tag}"  # 默认使用 tag ID
            # if hasattr(self, 'sentiment_id_to_name_map') and tag in self.sentiment_id_to_name_map:
            #     sentiment_name = self.sentiment_id_to_name_map[tag]

            res[f'sc_f1_{sentiment_name}'] = round(f * 100, 2)
            res[f'sc_precision_{sentiment_name}'] = round(pre * 100, 2)
            res[f'sc_recall_{sentiment_name}'] = round(rec * 100, 2)

            print(f"  {sentiment_name}: P={pre:.4f}, R={rec:.4f}, F1={f:.4f} (TP={tp}, FP={fp}, FN={fn})")

            # --- 累加用于后续平均计算 ---
            # 累加用于您自定义的基于平均 P 和 R 的 F1
            pre_sum_for_custom_f1 += pre
            rec_sum_for_custom_f1 += rec
            # f_sum_for_custom_f1 += f # 您原始代码的 f_sum，用于标准 Macro F1

            # 累加用于标准的 Macro P, R, F1
            per_class_precisions_list.append(pre)
            per_class_recalls_list.append(rec)
            per_class_f1_scores_list.append(f)

            # 累加用于 Weighted F1
            support_tag = tp + fn  # 当前类别的真实样本数 (支持度)
            if support_tag > 0:
                weighted_f1_numerator += (f * support_tag)
                total_support_for_sc += support_tag

        # --- 计算并保存宏平均指标 ---
        # 方式1: 您原始的基于平均 P 和 R 计算 F1 (可以保留，但命名为 custom 或 avg_pr_f1)
        num_valid_tags_for_custom = len(tags) if len(tags) > 0 else 1  # 防止除以零
        avg_pre_custom = pre_sum_for_custom_f1 / num_valid_tags_for_custom
        avg_rec_custom = rec_sum_for_custom_f1 / num_valid_tags_for_custom
        if avg_pre_custom + avg_rec_custom == 0:
            res['sc_f_custom_avg_pr'] = 0.0
        else:
            res['sc_f_custom_avg_pr'] = round(
                2 * avg_pre_custom * avg_rec_custom / (avg_pre_custom + avg_rec_custom + 1e-12) * 100, 2)
        res['sc_rec_avg_pr'] = round(avg_rec_custom * 100, 2)  # 这是 Macro Recall
        res['sc_pre_avg_pr'] = round(avg_pre_custom * 100, 2)  # 这是 Macro Precision
        print("旧指标 Micro f1 pre rec", res['sc_f_custom_avg_pr'], res['sc_pre_avg_pr'], res['sc_rec_avg_pr'])
        # 方式2: 标准的 Macro-F1, Macro-P, Macro-R (推荐报告这个)
        if per_class_f1_scores_list:  # 确保列表不为空
            res['sc_macro_f1'] = round(np.mean(per_class_f1_scores_list) * 100, 2)
            res['sc_macro_pre'] = round(np.mean(per_class_precisions_list) * 100, 2)
            res['sc_macro_rec'] = round(np.mean(per_class_recalls_list) * 100, 2)
        else:
            res['sc_macro_f1'] = 0.0
            res['sc_macro_pre'] = 0.0
            res['sc_macro_rec'] = 0.0

        # --- 计算并保存加权平均 F1 ---
        if total_support_for_sc > 0:
            res['sc_weighted_f1'] = round((weighted_f1_numerator / total_support_for_sc) * 100, 2)
        else:
            res['sc_weighted_f1'] = 0.0

        # --- 计算并保存 Micro-F1 (Overall Accuracy) ---
        # 您的 sc_acc 和 sc_all_num 是基于之前循环中的统计
        # 为了与混淆矩阵一致，最好也从混淆矩阵重新计算 Micro-F1 / Accuracy

        # 从混淆矩阵计算 Micro 指标
        cm_TPs = np.diag(self.confusion_matrix)
        cm_FPs = np.sum(self.confusion_matrix, axis=0) - cm_TPs
        # cm_FNs = np.sum(self.confusion_matrix, axis=1) - cm_TPs # FN 和 FP 在这里是相同的

        total_tp_from_cm = np.sum(cm_TPs)
        total_fp_from_cm = np.sum(cm_FPs)  # 等于 total_fn_from_cm

        micro_f1_from_cm = total_tp_from_cm / (total_tp_from_cm + total_fp_from_cm) \
            if (total_tp_from_cm + total_fp_from_cm) > 0 else 0
        res['sc_micro_f1'] = round(micro_f1_from_cm * 100, 2)
        res['sc_acc_from_cm'] = res['sc_micro_f1']  # Micro F1 通常等同于多分类的准确率

        # 保留您原有的 sc_acc 和 sc_all_num，但建议优先看从混淆矩阵计算的
        res['sc_acc_original_calc'] = round(
            1.0 * self.sc_right / (self.sc_all_num + 1e-12) * 100, 2)
        res['sc_all_num_original_calc'] = self.sc_all_num

        # 其他指标
        res['confusion_matrix_sc'] = self.confusion_matrix.tolist()  # 将 numpy array 转为 list 方便 json 序列化
        print("新指标 f1,pre,rec:",
              res['sc_macro_f1'],
              res['sc_macro_pre'],
              res['sc_macro_rec']
              )
        # --- 计算 Macro-F1 (你现有的逻辑，稍作调整确保分母正确) ---
        if valid_tags_count > 0:
            avg_macro_rec = macro_rec_sum / valid_tags_count
            avg_macro_pre = macro_pre_sum / valid_tags_count
            if avg_macro_pre + avg_macro_rec == 0:  # 防止除零
                res['sc_f_macro'] = 0.0
            else:
                res['sc_f_macro'] = round(
                    2 * avg_macro_pre * avg_macro_rec / (avg_macro_pre + avg_macro_rec + 1e-12) * 100, 2)
            res['sc_rec_macro'] = round(avg_macro_rec * 100, 2)
            res['sc_pre_macro'] = round(avg_macro_pre * 100, 2)
        else:  # 如果没有任何一个类别有真实样本 (不太可能，但作为边界处理)
            res['sc_f_macro'] = 0.0
            res['sc_rec_macro'] = 0.0
            res['sc_pre_macro'] = 0.0
        res['confusion_matrix'] = self.confusion_matrix
        # --- 计算 Weighted-F1 ---
        if total_support_for_sc > 0:
            res['sc_f_weighted'] = round((weighted_f1_numerator / total_support_for_sc) * 100, 2)
        else:  # 如果总支持度为0 (不太可能)
            res['sc_f_weighted'] = 0.0
        # -------------

        if reset:
            self.aesc_fp = 0
            self.aesc_tp = 0
            self.aesc_fn = 0
            self.ae_fp = 0
            self.ae_tp = 0
            self.ae_fn = 0
            self.sc_all_num = 0
            self.sc_right = 0
            self.sc_fp = Counter()
            self.sc_tp = Counter()
            self.sc_fn = Counter()
            self.confusion_matrix = np.zeros((3, 3), dtype=int)
        return res


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    # print(ts)
    # print(ps)
    if isinstance(ts, (list, set)):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, (list, set)):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        # print(key)
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        # print(p_num, t_num)
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        # print(fp, tp, fn)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    # print(fp, tp, fn)
    return tp, fn, fp


class OESpanMetric(object):
    def __init__(self, eos_token_id, num_labels, opinion_first=True):
        super(OESpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels + 2

        self.oe_fp = 0
        self.oe_tp = 0
        self.oe_fn = 0
        self.em = 0
        self.total = 0
        self.invalid = 0
        # assert opinion_first is False, "Current metric only supports aspect first"

        self.opinin_first = opinion_first

    def evaluate(self, oe_target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(
            pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(
            target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        flag = True
        for i, (ts, ps) in enumerate(zip(oe_target_span, pred.tolist())):
            em = 0
            assert ps[0] == tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(
                    pred[i, :target_seq_len[i]]).sum().item() ==
                         target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps, start=1):
                    if index % 2 == 0:
                        cur_pair.append(j)
                        if cur_pair[0] > cur_pair[1] or cur_pair[0] < self.word_start_index \
                                or cur_pair[1] < self.word_start_index:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            self.invalid += invalid

            oe_target_counter = Counter([tuple(t) for t in ts])
            oe_pred_counter = Counter(pairs)
            # if flag:
            #     print(tgt_tokens[0])
            #     print(pred[0])
            #     print(ts)
            #     print(pairs)
            #     flag = False
            # 这里相同的pair会被计算多次
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_pred_counter.keys())),
                                           set(list(oe_target_counter.keys())))
            self.oe_fn += fn
            self.oe_fp += fp
            self.oe_tp += tp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.oe_tp, self.oe_fn, self.oe_fp)

        res['oe_f'] = round(f * 100, 2)
        res['oe_rec'] = round(rec * 100, 2)
        res['oe_pre'] = round(pre * 100, 2)

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.oe_fp = 0
            self.oe_tp = 0
            self.oe_fn = 0

        return res

# metric = AESCSpanMetric(1, 3, -1)

# spans = [[(6, 7, 3), (9, 10, 4)]]
# pred = torch.tensor([[0, 2, 2, 6, 7, 3, 9, 9, 4, 1, 1]])
# print(pred.size())
# tgt = torch.tensor([[0, 2, 2, 6, 7, 3, 9, 10, 4, 1, 1]])
# metric.evaluate(spans, pred, tgt)

# metric.pri()
