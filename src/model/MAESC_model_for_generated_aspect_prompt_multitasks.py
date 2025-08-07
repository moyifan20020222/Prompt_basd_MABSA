from collections import defaultdict
from typing import Optional, Tuple
from fastNLP.modules.torch.encoder import Seq2SeqEncoder
from fastNLP.modules.torch.decoder import Seq2SeqDecoder
from fastNLP.modules.torch import State
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import cosine_similarity

from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from transformers import BartTokenizer

from src.model.config import MultiModalBartConfig
# from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules_for_prompt_multitasks import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss, \
    MultiModalBartEncoder_for_Generating_aspect_prompt, MultiModalBartDecoder_generate_aspect_prompt, \
    MultiModalBartDecoder_MLM
from src.model.modules_for_prompt_multitasks import MultiModalBartDecoder_aspects_num


class MultiModalBartModel_AESC(PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder_for_generated_aspect_prompt = MultiModalBartEncoder(config, encoder,
                                                                               tokenizer.img_feat_id,
                                                                               tokenizer.cls_token_id,
                                                                               args.num_image_tokens)

        multimodal_encoder = MultiModalBartEncoder_for_Generating_aspect_prompt(
            use_generated_prompt=args.use_generated_prompt,
            config=config,
            encoder=encoder,
            img_feat_id=tokenizer.img_feat_id,
            aspect_prompt_token_id=tokenizer.aspect_prompt_token_id,
            senti_prompt_token_id=tokenizer.senti_prompt_token_id,
            cls_token_id=tokenizer.cls_token_id,
            num_image_tokens=args.num_image_tokens,
            use_different_aspect_prompt=args.use_different_aspect_prompt,
            aspect_prompt_token_front_id=tokenizer.aspect_prompt_token_front_id,
            aspect_prompt_token_end_id=tokenizer.aspect_prompt_token_end_id,
            is_few_shot=args.is_few_shot
        )
        # 这里做出一个小修改，为了保证模型可塑性，我们需要返回当前所用模型的encoder层数信息， 用于在APD模块中使用，
        return multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder, decoder, encoder

    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        label_ids = sorted(label_ids)
        multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder, share_decoder, encoder = self.build_model(
            args, bart_model, self.tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.use_multitasks = args.use_multitasks
        self.loss_lambda = args.loss_lambda
        self.num_image_tokens = args.num_image_tokens

        self.aspect_prompt_encoder = multimodal_encoder_for_generated_aspect_prompt
        self.encoder = multimodal_encoder
        self.bart_encoder = encoder
        only_sc = False
        # need_tag = True  #if predict the sentiment or not
        if args.task == 'twitter_ae':
            need_tag = False
        else:
            need_tag = True
            # if args.task == 'twitter_sc':
            #     only_sc = True
        # 这个是APD模块信息。 多增加一个参数，用于获取encoder的信息
        self.prompt_decoder = MultiModalBartDecoder_generate_aspect_prompt(config=self.config, decoder=share_decoder,
                                                                           encoder=encoder,
                                                                           prompt_pool_num=args.Prompt_Pool_num,
                                                                           diversity_loss_weight=args.diversity_loss_weight,
                                                                           l2_reg_weight=args.l2_reg_weight,
                                                                           is_few_shot=args.is_few_shot)

        if self.use_multitasks:  # 这个的意思是 AND 模块信息
            self.aspect_num_decoder = MultiModalBartDecoder_aspects_num(config=self.config, decoder=share_decoder,
                                                                        encoder=encoder,
                                                                        prompt_pool_num=args.Prompt_Pool_num,
                                                                        diversity_loss_weight=args.diversity_loss_weight,
                                                                        l2_reg_weight=args.l2_reg_weight,
                                                                        is_few_shot=args.is_few_shot)
        # 这是最后生成结果的序列的 decoder
        self.decoder = MultiModalBartDecoder_span(self.config,
                                                  self.tokenizer,
                                                  share_decoder,
                                                  self.tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  num_image_tokens=self.num_image_tokens,
                                                  need_tag=need_tag,
                                                  only_sc=False)
        self.span_loss_fct = Span_loss(args)

        # MLM 损失， 用在测试阶段对模型进行微调，它的微调只对Aspect索引的token信息创建的参数使用。 所以他需要在Aspect索引decoder之后
        # 对他的参数进行修正
        self.mlm_loss_module = MultiModalBartDecoder_MLM(self.config, self.prompt_decoder.decoder)
        # 同样的，下方是全数据集上测试较好的结果
        if not args.is_few_shot:
            # 字幕与文本的相关度计算 6.1 8->4
            self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8,
                                                                      batch_first=True, dropout=0.2)
            # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
            self.text_caption_attn_output_projection = nn.Sequential(
                nn.Linear(768 + 1, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            # 字幕与文本的相关度计算 第二轮的encoder的
            self.text_caption_cross_attention_second = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
                                                                             num_heads=8,
                                                                             batch_first=True, dropout=0.2)
            # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
            self.text_caption_attn_output_projection_second = nn.Sequential(
                nn.Linear(768 + 1, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            self.nouns_cross_attention = nn.MultiheadAttention(768, 4, batch_first=True)
            self.gate_proj_nouns = nn.Linear(768 * 2, 1)
            self.nouns_cross_attention_image = nn.MultiheadAttention(768, 4, batch_first=True)
            self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)
        # 少样本下的模型架构：
        else:
            # 字幕与文本的相关度计算 5.31 4->2
            self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=4,
                                                                      batch_first=True, dropout=0.2)
            # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
            # self.text_caption_attn_output_projection = nn.Linear(768 + 1, 1)
            self.text_caption_attn_output_projection = nn.Sequential(
                nn.Linear(768 + 1, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            # 字幕与文本的相关度计算 第二轮的encoder的
            self.text_caption_cross_attention_second = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
                                                                             num_heads=4,
                                                                             batch_first=True, dropout=0.2)
            # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
            # self.text_caption_attn_output_projection_second = nn.Linear(768 + 1, 1)
            self.text_caption_attn_output_projection_second = nn.Sequential(
                nn.Linear(768 + 1, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            self.nouns_cross_attention = nn.MultiheadAttention(768, 2, batch_first=True)
            self.gate_proj_nouns = nn.Linear(768 * 2, 1)
            self.nouns_cross_attention_image = nn.MultiheadAttention(768, 2, batch_first=True)
            self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)

        # 定义相关度阈值  可以 把阈值变成可学习的参数， 与固定的阈值做一个比较，
        if args.dataset[0][0] == 'twitter15':
            self.threshold = nn.Parameter(torch.tensor(0.7))
        elif args.dataset[0][0] == 'twitter17':
            self.threshold = nn.Parameter(torch.tensor(0.7))

        # 温度参数，用于调整sigmoid的陡峭程度
        self.temperature = nn.Parameter(torch.tensor(5.0))

        # 定义字幕名词和文本名词的相关度阈值， 当低于这个阈值的时候，同样可以认为， 图像与文本并不一致，说明图片信息将是干扰。
        if args.dataset[0][0] == 'twitter15':
            self.cosine_threshold = 0.9
        elif args.dataset[0][0] == 'twitter17':
            self.cosine_threshold = 0.9
        self.hidden_size = 768
        self.mate_projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)  # 投影到128维
        )
        # MATE对比学习的温度
        self.mate_contrastive_temperature = 0.1
        self.noun_cache = defaultdict(lambda: None)  # 名词嵌入缓存
        self.span_augmentor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.2)  # Dropout是实现数据增强的关键
        )

    def prepare_state(self,
                      input_ids,
                      image_features,
                      attention_mask=None,
                      aesc_infos=None,
                      aspects_num=None,
                      sentence_mask=None,
                      image_mask=None,
                      mlm_message=None,
                      image_caption_valid=None,
                      image_caption_mask=None,
                      score=None,
                      caption_nouns=None,
                      sentence_nouns=None,
                      Training=None,
                      first=None):
        ##generate prompt for each instance

        prompt_attention_mask = attention_mask
        if self.num_image_tokens == 0:
            end_index = 62
            begin_index = 22
        elif self.num_image_tokens == 1:
            end_index = 63
            begin_index = 23
        elif self.num_image_tokens == 2:
            end_index = 64
            begin_index = 24
        elif self.num_image_tokens == 3:
            end_index = 65
            begin_index = 25
        elif self.num_image_tokens == 4:
            end_index = 66
            begin_index = 26
        elif self.num_image_tokens == 5:
            end_index = 67
            begin_index = 27
        elif self.num_image_tokens == 6:
            end_index = 68
            begin_index = 28
        elif self.num_image_tokens == 7:
            end_index = 69
            begin_index = 29

        for i in range(len(prompt_attention_mask)):
            mask = prompt_attention_mask[i]
            mask[begin_index:end_index] = torch.zeros_like(mask[begin_index:end_index])  ##26:66 是aspect提示的位置
            prompt_attention_mask[i] = mask

        ''' aspects_prompt '''
        # 这里在与两个任务一起上出现了 不同， 这个encoder的结果 是否共用也可以是一个测试的部分。
        dict_for_prompt, loss, gating_scores, hard_mask = self.aspect_prompt_encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
            caption_mask=image_caption_mask,
            sentence_mask=sentence_mask,
            image_mask=image_mask,
            image_valid=image_caption_valid,
            score=score,
            return_dict=True)

        #  通过计算字幕与文本的相关度，让图片特征的信息相应的减少
        # print("image_caption_mask:", image_caption_mask)
        image_caption_mask_expanded_for_image = image_caption_mask.unsqueeze(-1).to(
            attention_mask.device)  # 为图像 mask 扩展维度
        image_caption_embeddings = dict_for_prompt.last_hidden_state * image_caption_mask_expanded_for_image.float()  # 提取图像嵌入
        image_caption = image_caption_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value

        # 文本嵌入
        sentence_mask_expanded_for_image = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        text_embeddings = dict_for_prompt.last_hidden_state * sentence_mask_expanded_for_image.float()  # 提取图像嵌入
        text = text_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # 图像嵌入
        image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        image_embeddings = dict_for_prompt.last_hidden_state * image_mask_expanded_for_image.float()  # 提取图像嵌入
        image = image_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        relevance_scores = torch.zeros(image_caption_mask.size(0), 1, device=input_ids.device)  # 初始化相关度得分张量 (形状 [b, 1])
        # print("image_caption_valid 维度", image_caption_mask.size())
        # print("图片特征长度", len(image_features))
        loss_crd = torch.tensor(0.0, dtype=torch.float)
        valid_indices = torch.where(image_caption_valid)[0]
        # print("valid_indices", valid_indices.size())
        # print("score", score.size())
        num_valid = len(valid_indices)
        batch_size = image_caption_mask.size(0)
        # 使用 logits 计算损失，使用 scores 进行门控
        crd_logits_all = torch.zeros(batch_size, 1, device=attention_mask.device)  # 默认logit为0，对应sigmoid(0)=0.5
        # relevance_scores_all = torch.full((batch_size, 1), 0.5, device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
        relevance_scores_all = score.unsqueeze(-1).to(attention_mask.device)  # 对于没有字幕就用预训练模型给出的参数
        # -------------------
        # 第一轮相关度计算
        # if num_valid > 0:
        #     # 2. 提取有效样本的数据
        #     valid_text_emb = text[valid_indices].to(attention_mask.device)  # [num_valid, seq_len_text, hidden_size]
        #     valid_caption_emb = image_caption[valid_indices].to(
        #         attention_mask.device)  # [num_valid, max_len, hidden_size]
        #     valid_text_mask = sentence_mask[valid_indices].to(attention_mask.device)  # [num_valid, max_len]
        #     # **关键: MultiheadAttention 的 key_padding_mask 需要 True 表示 Padding 位置**
        #     # 假设你的 mask 是 1 表示有效, 0 表示 padding, 需要转换
        #     valid_caption_padding_mask = (
        #             image_caption_mask[valid_indices] == 0).to(
        #         attention_mask.device)  # [num_valid, max_len], True for padding
        #
        #     # 3. 批处理交叉注意力 (假设 batch_first=True)
        #     # query: text, key/value: caption
        #     attn_output_valid, attn_weights = self.text_caption_cross_attention(
        #         query=valid_text_emb,
        #         key=valid_caption_emb,
        #         value=valid_caption_emb,
        #         key_padding_mask=valid_caption_padding_mask  # ****** 提供正确的 Mask ******
        #     )  # Output: [num_valid, max_len, hidden_size]
        #
        #     # 提取最高的几个注意力权重，表示最相关的部分
        #     top_k_weights, _ = torch.topk(attn_weights, k=min(5, attn_weights.size(-1)), dim=-1)
        #     attention_focus = top_k_weights.mean(dim=-1).mean(dim=-1).unsqueeze(1)  # [num_valid, 1]
        #
        #     # 4. 批处理屏蔽平均池化
        #     # 将 padding 位置的 attention output 置零
        #     attn_output_valid_masked = attn_output_valid * valid_text_mask.unsqueeze(-1).float().to(
        #         attention_mask.device)
        #     # 计算每个样本的有效长度
        #     text_lengths_valid = valid_text_mask.sum(dim=1, keepdim=True).float().to(
        #         attention_mask.device)  # [num_valid, 1]
        #     # 对有效位置求和
        #     sum_attn_output_valid = attn_output_valid_masked.sum(dim=1).to(
        #         attention_mask.device)  # [num_valid, hidden_size]
        #     # 计算平均值，防止除零
        #     mean_attn_output_valid = sum_attn_output_valid / torch.clamp(text_lengths_valid,
        #                                                                  min=1e-9)  # [num_valid, hidden_size]
        #
        #     # 计算文本和图像表示的余弦相似度
        #     text_pooled = valid_text_emb.sum(dim=1) / torch.clamp(valid_text_mask.sum(dim=1, keepdim=True), min=1e-9)
        #     caption_pooled = valid_caption_emb.sum(dim=1) / torch.clamp(
        #         (~valid_caption_padding_mask).sum(dim=1, keepdim=True), min=1e-9)
        #     cosine_sim = F.cosine_similarity(text_pooled, caption_pooled, dim=1, eps=1e-8).unsqueeze(
        #         1)  # [num_valid, 1]
        #
        #     # 5. 多特征融合
        #     # 将注意力焦点和余弦相似度与平均表示拼接
        #     combined_features = torch.cat([
        #         mean_attn_output_valid,  # 交叉注意力特征
        #         cosine_sim,  # 余弦相似度特征
        #     ], dim=1)
        #     # 5. 批处理线性投影，得到 Logits
        #     logits_valid = self.text_caption_attn_output_projection(combined_features)  # [num_valid, 1]
        #
        #     # 6. 计算有效样本的相关性得分 (用于门控)
        #     scores_valid = torch.sigmoid(logits_valid).to(attention_mask.device)  # [num_valid, 1]
        #
        #     # 7. 计算 CRD 损失 (仅针对有效样本)
        #     target_labels_valid = score[valid_indices].unsqueeze(1).to(attention_mask.device)  # [num_valid, 1]
        #     # 再转为硬标签
        #     hard_labels = (target_labels_valid > self.threshold).float()
        #
        #     criterion_crd = nn.BCELoss()
        #     loss_crd_hard = criterion_crd(scores_valid, hard_labels)
        #     # 就正常的使用软标签
        #     loss_crd_soft = F.binary_cross_entropy(scores_valid, target_labels_valid)  # 无需阈值处理
        #     # loss_crd = 0.6 * loss_crd_soft + 0.4 * loss_crd_hard
        #     loss_crd = loss_crd_soft
        #     # 8. 更新完整批次的 logits 和 scores
        #     # 使用 scatter_ 或 index_put_ 更新特定索引的值
        #     crd_logits_all[valid_indices] = logits_valid
        #     relevance_scores_all[valid_indices] = scores_valid
        #
        #     # 9. 使用相关性得分调整 image_features (向量化)
        #     # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
        #     # 9. 使用相关性得分调整 image_features (向量化)
        #     # 应用学习型阈值和温度参数
        # threshold = torch.sigmoid(self.threshold)
        # temperature = torch.abs(self.temperature) + 1.0  # 确保温度参数为正
        #
        # # 平滑阈值处理
        # adjusted_scores = torch.sigmoid((relevance_scores_all - threshold) * temperature)
        #
        # # 硬阈值+软权重结合
        # hard_mask = (relevance_scores_all > adjusted_scores).bool()
        # gating_scores = relevance_scores_all * hard_mask
        # gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)
        # # 软权重
        # # gating_scores = relevance_scores_all.unsqueeze(1).to(attention_mask.device)  # 调整为 [batch_size, 1, 1] 以便广播
        # # 或者如果 image_features 是 [batch_size, img_hidden]
        # # gating_scores = relevance_scores_all # 调整为 [batch_size, 1]
        # # print("gating_scores", gating_scores)
        # # print("image_features", )
        #
        # # ------------------------
        # # 一种方法是 原始信息就不更改了，只有用在信息融合的时候再修改，即在下面具体每一个decoder中
        # # 还有是只在原始信息上更改，后续融合的时候正常融合，让模型决定融合权重
        # # for i in range(batch_size):
        # #     image_features[i] = gating_scores[i] * image_features[i]
        hard_mask = hard_mask.to(image_caption_mask.device)
        # weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # # image_mask = image_mask * hard_mask
        # # print("image的维度", image.size())
        # # print("计算的token结果", weighted_image_tokens.size())
        # # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        # encoder_outputs = dict_for_prompt.last_hidden_state
        # mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_prompt.last_hidden_state = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     weighted_image_tokens[mask_expanded_encoder]
        # )
        #
        # # 字幕修正
        # weight_image_caption = image_caption * gating_scores
        # image_caption_mask = image_caption_mask.to(input_ids.device)
        image_caption_mask = image_caption_mask * hard_mask
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_prompt.last_hidden_state = dict_for_prompt.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )

        # ---------------------
        loss_crd = loss
        # print("计算的 权重 ： ", relevance_weights)
        # TODO: 有个问题， 后续的字幕信息我们也一起作为了encoder的输入， 它的信息如何处理， 方案1、因为这两者 表达内容接近，可以采用一个门控机制学习
        # 2、直接认为这个字幕信息就是一个中间内容，计算出图片和文本的相关度后就可以丢弃了， 那为了保证结构统一，可以直接在此乘0，或是mask为0即可。

        # --------------------
        # 增加一个备选名词集合的部分，他将会作为候选的aspect，增强模型对方面术语的感受

        noun_embeds, noun_mask = self.process_batch(caption_nouns, sentence_nouns, attention_mask.device)
        # print("noun_embeds min/max:", noun_embeds.min(), noun_embeds.max())
        # -------------------

        # --------------

        aspect_prompt_decoder_input_ids, aspect_prompt_decoder_attention_mask = [
            aesc_infos['aspect_prompt_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspect_prompt_decoder_attention_mask'].to(input_ids.device)]
        # print("aspect_prompt_decoder_input_ids", aspect_prompt_decoder_input_ids.shape)
        generated_prompt, sparsity_loss_layers, sparsity_loss_image = self.prompt_decoder(
            encoder_outputs=dict_for_prompt.last_hidden_state,
            attention_mask=attention_mask,
            decoder_input_ids=aspect_prompt_decoder_input_ids,
            decoder_attention_mask=aspect_prompt_decoder_attention_mask,
            sentence_mask=sentence_mask,
            image_mask=image_mask,
            encoder_outputs_all=dict_for_prompt.hidden_states,
            nouns_embeds=noun_embeds,
            nouns_mask=noun_mask,
            image_caption_valid=image_caption_valid,
            image_caption_mask=image_caption_mask,
            score=gating_scores)
        # 生成的用于生成索引Prompt 的 总索引Prompt 这个时候还没有开始构建Prompt，所以AND的判别数目在下面
        generated_prompt = generated_prompt[:, 1:, :]  ##(batch_size, 2, 768)
        pseudo_loss = torch.tensor(0.0, dtype=torch.float)
        # 添加一个MLM损失，这个是一个伪CTTA方法，这个损失只在测试阶段使用，它的作用能迫使模型更好地理解测试数据的语言分布
        # mlm_labels = mlm_message['mlm_labels'].to(input_ids.device)
        # mlm_decoder_input_ids = mlm_message['mlm_decoder_input_ids'].to(input_ids.device)
        # mlm_decoder_attention_mask = mlm_message['mlm_decoder_attention_mask'].to(input_ids.device)
        # # mlm_inputs_id, mlm_labels = self.prepare_mlm(input_ids, attention_mask, self.tokenizer)
        # pseudo_loss = self.mlm_loss_module(labels=mlm_labels, input_ids=input_ids,
        #                                    encoder_outputs=dict_for_prompt.last_hidden_state,
        #                                    attention_mask=attention_mask,
        #                                    decoder_input_ids=mlm_decoder_input_ids,
        #                                    decoder_attention_mask=mlm_decoder_attention_mask)

        '''aspects_num'''
        aspects_num_decoder_input_ids, aspects_num_decoder_attention_mask = [
            aesc_infos['aspects_num_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspects_num_decoder_attention_mask'].to(input_ids.device)]
        sparsity_loss_layers_aspect = torch.tensor(0.0, dtype=torch.float)
        sparsity_loss_image_aspect = torch.tensor(0.0, dtype=torch.float)
        # import ipdb; ipdb.set_trace()
        if self.use_multitasks:  # 用于预测Aspect的数量
            aspects_num_loss, predict_aspects_num_logits, sparsity_loss_layers_aspect, sparsity_loss_image_aspect = \
                self.aspect_num_decoder(aspects_num_labels=aspects_num,
                                        encoder_outputs=dict_for_prompt[0],
                                        attention_mask=attention_mask,
                                        aspects_num_decoder_input_ids=aspects_num_decoder_input_ids,
                                        sentence_mask=sentence_mask,
                                        image_mask=image_mask,
                                        encoder_outputs_all=dict_for_prompt.hidden_states,
                                        nouns_embeds=noun_embeds,
                                        nouns_mask=noun_mask,
                                        image_caption_valid=image_caption_valid,
                                        image_caption_mask=image_caption_mask,
                                        score=gating_scores)

            predict_aspects_num = torch.argmax(predict_aspects_num_logits, dim=1)
            new_predict_aspects_num = predict_aspects_num + torch.ones_like(predict_aspects_num)
        else:
            # 这边的aspect 数量提升 由5 -> 6
            aspects_num_loss = 0
            new_predict_aspects_num = []
            predict_aspects_num = []
            for i in range(len(input_ids)):
                # new_predict_aspects_num.append(5)
                # predict_aspects_num.append(4)
                new_predict_aspects_num.append(6)
                predict_aspects_num.append(5)
            new_predict_aspects_num = torch.tensor(new_predict_aspects_num)
            predict_aspects_num = torch.tensor(predict_aspects_num)
        # 把融合放到每一个具体Aspect的计算中
        dict = self.encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            generated_prompt=generated_prompt,
            aspects_num=new_predict_aspects_num,
            sentence_mask=sentence_mask,
            image_mask=image_mask,
            encoder_outputs_all=dict_for_prompt.hidden_states,
            encoder_output=dict_for_prompt.last_hidden_state,
            image_caption_valid=image_caption_valid,
            image_caption_mask=image_caption_mask,
            output_hidden_states=True,
            return_dict=True)
        # -------------- 在第二次的encoder结果中同样使用一样的修正部分，前面的修正部分，只为了aspect部分的信息获取。
        loss_crd_second = torch.tensor(0.0, dtype=torch.float)
        # 6.1 20.43去除
        # image_caption_mask_expanded_for_image = image_caption_mask.unsqueeze(-1).to(
        #     attention_mask.device)  # 为图像 mask 扩展维度
        # image_caption_embeddings = dict.last_hidden_state * image_caption_mask_expanded_for_image.float()  # 提取图像嵌入
        # image_caption = image_caption_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        #
        # # 文本嵌入
        # sentence_mask_expanded_for_image = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        # text_embeddings = dict.last_hidden_state * sentence_mask_expanded_for_image.float()  # 提取图像嵌入
        # text = text_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # # 图像嵌入
        # image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        # image_embeddings = dict.last_hidden_state * image_mask_expanded_for_image.float()  # 提取图像嵌入
        # image = image_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # relevance_scores = torch.zeros(image_caption_mask.size(0), 1, device=input_ids.device)  # 初始化相关度得分张量 (形状 [b, 1])
        # # print("image_caption_valid 维度", image_caption_mask.size())
        # # print("图片特征长度", len(image_features))
        # loss_crd_second = torch.tensor(0.0, dtype=torch.float)
        # valid_indices = torch.where(image_caption_valid)[0]
        # # print("valid_indices", valid_indices.size())
        # # print("score", score.size())
        # num_valid = len(valid_indices)
        # batch_size = image_caption_mask.size(0)
        # # 使用 logits 计算损失，使用 scores 进行门控
        # crd_logits_all = torch.zeros(batch_size, 1, device=attention_mask.device)  # 默认logit为0，对应sigmoid(0)=0.5
        # # relevance_scores_all = torch.full((batch_size, 1), 0.5, device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
        # relevance_scores_all = score.unsqueeze(-1).to(attention_mask.device)  # 对于没有字幕就用预训练模型给出的参数

        # 第二轮的相关度计算：
        # ----------------------------
        # if num_valid > 0:
        #     # 2. 提取有效样本的数据
        #     valid_text_emb = text[valid_indices].to(attention_mask.device)  # [num_valid, seq_len_text, hidden_size]
        #     valid_caption_emb = image_caption[valid_indices].to(
        #         attention_mask.device)  # [num_valid, max_len, hidden_size]
        #     valid_text_mask = sentence_mask[valid_indices].to(attention_mask.device)  # [num_valid, max_len]
        #     # **关键: MultiheadAttention 的 key_padding_mask 需要 True 表示 Padding 位置**
        #     # 假设你的 mask 是 1 表示有效, 0 表示 padding, 需要转换
        #     valid_caption_padding_mask = (
        #             image_caption_mask[valid_indices] == 0).to(
        #         attention_mask.device)  # [num_valid, max_len], True for padding
        #
        #     # 3. 批处理交叉注意力 (假设 batch_first=True)
        #     # query: text, key/value: caption
        #     attn_output_valid, _ = self.text_caption_cross_attention_second(
        #         query=valid_text_emb,
        #         key=valid_caption_emb,
        #         value=valid_caption_emb,
        #         key_padding_mask=valid_caption_padding_mask  # ****** 提供正确的 Mask ******
        #     )  # Output: [num_valid, max_len, hidden_size]
        #
        #     # 4. 批处理屏蔽平均池化
        #     # 将 padding 位置的 attention output 置零
        #     attn_output_valid_masked = attn_output_valid * valid_text_mask.unsqueeze(-1).float().to(
        #         attention_mask.device)
        #     # 计算每个样本的有效长度
        #     text_lengths_valid = valid_text_mask.sum(dim=1, keepdim=True).float().to(
        #         attention_mask.device)  # [num_valid, 1]
        #     # 对有效位置求和
        #     sum_attn_output_valid = attn_output_valid_masked.sum(dim=1).to(
        #         attention_mask.device)  # [num_valid, hidden_size]
        #     # 计算平均值，防止除零
        #     mean_attn_output_valid = sum_attn_output_valid / torch.clamp(text_lengths_valid,
        #                                                                  min=1e-9)  # [num_valid, hidden_size]
        #     # 计算文本和图像表示的余弦相似度
        #     text_pooled = valid_text_emb.sum(dim=1) / torch.clamp(valid_text_mask.sum(dim=1, keepdim=True), min=1e-9)
        #     caption_pooled = valid_caption_emb.sum(dim=1) / torch.clamp(
        #         (~valid_caption_padding_mask).sum(dim=1, keepdim=True), min=1e-9)
        #     cosine_sim = F.cosine_similarity(text_pooled, caption_pooled, dim=1, eps=1e-8).unsqueeze(
        #         1)  # [num_valid, 1]
        #
        #     # 多特征融合
        #     combined_features = torch.cat([
        #         mean_attn_output_valid,  # 交叉注意力特征
        #         cosine_sim,  # 余弦相似度特征
        #     ], dim=1)
        #
        #     # 5. 批处理线性投影，得到 Logits  mean_attn_output_valid -> combined_features
        #     logits_valid = self.text_caption_attn_output_projection_second(combined_features)  # [num_valid, 1]
        #
        #     # 6. 计算有效样本的相关性得分 (用于门控)
        #     scores_valid = torch.sigmoid(logits_valid).to(attention_mask.device)  # [num_valid, 1]
        #
        #     # 7. 计算 CRD 损失 (仅针对有效样本)
        #     target_labels_valid = score[valid_indices].unsqueeze(1).to(attention_mask.device)  # [num_valid, 1]
        #     # 再转为硬标签
        #     hard_labels = (target_labels_valid > self.threshold).float()
        #
        #     criterion_crd = nn.BCELoss()
        #     loss_crd_hard_second = criterion_crd(scores_valid, hard_labels)
        #     # 就正常的使用软标签
        #     loss_crd_soft_second = F.binary_cross_entropy(scores_valid, target_labels_valid)  # 无需阈值处理
        #     loss_crd_second = 0.6 * loss_crd_soft_second + 0.4 * loss_crd_hard_second
        #     # 8. 更新完整批次的 logits 和 scores
        #     # 使用 scatter_ 或 index_put_ 更新特定索引的值
        #     crd_logits_all[valid_indices] = logits_valid
        #     relevance_scores_all[valid_indices] = scores_valid
        #
        #     # 9. 使用相关性得分调整 image_features (向量化)
        #     # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
        #     # 9. 使用相关性得分调整 image_features (向量化)
        #     # 应用学习型阈值和温度参数
        # threshold = torch.sigmoid(self.threshold)
        # temperature = torch.abs(self.temperature) + 1.0  # 确保温度参数为正
        #
        # # 平滑阈值处理
        # adjusted_scores = torch.sigmoid((relevance_scores_all - threshold) * temperature)
        #
        # # 硬阈值+软权重结合
        # hard_mask = (relevance_scores_all > adjusted_scores).float()
        # gating_scores = relevance_scores_all * hard_mask
        # gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)        # 图像特征后续不用了，就扔掉了。
        #
        # weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # # print("image的维度", image.size())
        # # print("计算的token结果", weighted_image_tokens.size())
        # # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        # encoder_outputs = dict.last_hidden_state
        # mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict.last_hidden_state = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     weighted_image_tokens[mask_expanded_encoder]
        # )
        # # 字幕修正
        # weight_image_caption = image_caption * gating_scores
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_prompt.last_hidden_state = dict_for_prompt.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )

        # # ------ 名词修正同样来一次。
        # encoder_outputs = dict.last_hidden_state
        #
        # mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, encoder_outputs.size(2)).to(attention_mask.device)
        # # 使用广播机制提取文本特征 [b,s,768]
        # text_embeddings = encoder_outputs * mask_expanded.float()
        #
        # # --- 新增：图像嵌入交叉注意力计算 ---
        # # 现在利用相关度对 图像信息进行了修正， 其实可以把这个图像融合也放回来试试效果
        # image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        # image_embeddings = encoder_outputs * image_mask_expanded_for_image.float()  # 提取图像嵌入
        # image_mask_expanded_for_image = image_mask_expanded_for_image.repeat(1, 1, text_embeddings.size(2)).to(
        #     attention_mask.device)
        # # 再做任何操作前，先和名词做一个交叉注意力计算，让文本嵌入与更加关注特定的Aspect，也就是我们的目标
        # nouns_attn_output, attn_weights = self.nouns_cross_attention(
        #     query=text_embeddings,
        #     key=noun_embeds,
        #     value=noun_embeds,
        #     key_padding_mask=noun_mask,  # True 表示需要被掩码的位置
        #     need_weights=True  # 返回注意力权重以便调试
        # )
        # # 直接相加 or 门控融合
        # # text_embeddings = text_embeddings + nouns_attn_output
        #
        # gate_nouns = torch.sigmoid(self.gate_proj_nouns(torch.cat([text_embeddings, nouns_attn_output], dim=-1)))
        # text_embeddings = gate_nouns * text_embeddings + (1 - gate_nouns) * nouns_attn_output
        # # 同理对图像嵌入也做同样操作
        #
        # nouns_attn_output_image, attn_weights = self.nouns_cross_attention_image(
        #     query=image_embeddings,
        #     key=noun_embeds,
        #     value=noun_embeds,
        #     key_padding_mask=noun_mask,  # True 表示需要被掩码的位置
        #     need_weights=True  # 返回注意力权重以便调试
        # )
        # gate_nouns_image = torch.sigmoid(
        #     self.gate_proj_nouns_image(torch.cat([image_embeddings, nouns_attn_output_image], dim=-1)))
        # image_embeddings = gate_nouns_image * image_embeddings + (1 - gate_nouns_image) * nouns_attn_output_image
        # #  把计算结果放回去，
        # dict.last_hidden_state = encoder_outputs.masked_scatter(
        #     image_mask_expanded_for_image,
        #     image_embeddings[image_mask_expanded_for_image]
        # )
        # #  把计算结果放回去，
        # dict.last_hidden_state = dict.last_hidden_state.masked_scatter(
        #     mask_expanded_encoder,
        #     text_embeddings[mask_expanded_encoder]
        # )

        # -----
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]
        state = BartState(
            encoder_outputs,
            encoder_mask,
            input_ids[:,
            end_index:],  # the text features start from index 38, the front are image features.
            first,
            src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        loss_crd = loss_crd + loss_crd_second
        loss = [sparsity_loss_layers, sparsity_loss_layers_aspect, loss_crd]
        hidden_state, logits = [], []
        return state, aspects_num_loss, predict_aspects_num, pseudo_loss, loss, hidden_state, logits, encoder_outputs

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            aesc_infos=None,
            aspects_num=None,
            sentence_mask=None,
            image_mask=None,
            mlm_message=None,
            image_caption_valid=None,
            image_caption_mask=None,
            score=None,
            caption_nouns=None,
            sentence_nouns=None,
            training=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        ### for prompt
        # import ipdb; ipdb.set_trace()

        ## for aspect-spans

        aspects_num = torch.tensor(aspects_num).to(input_ids.device)
        state, aspects_num_loss, predict_aspects_num, pseudo_loss, loss, _, _, encoder_outputs = self.prepare_state(
            input_ids,
            image_features,
            attention_mask,
            aesc_infos,
            aspects_num,
            sentence_mask,
            image_mask,
            mlm_message,
            image_caption_valid,
            image_caption_mask,
            score,
            caption_nouns,
            sentence_nouns,
            Training=training
            )
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]
        sentence_mask = sentence_mask.to(input_ids.device)
        hidden_state, logits = self.decoder(spans, state)  ## spans: (2, 13) logits: (2, 12, 40)
        # mate_loss = self.calculate_mate_contrastive_loss(encoder_outputs, spans, sentence_mask, self.tokenizer)
        # print("mate对比损失", mate_loss.item())
        if training:
            span_loss, per_token_loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])
        else:
            span_loss, per_token_loss = self.span_loss_fct(spans[:, :], logits, span_mask[:, :])
        all_loss = span_loss + self.loss_lambda * aspects_num_loss
        for i in range(len(loss)):
            all_loss = all_loss + loss[i]
        return all_loss, predict_aspects_num, pseudo_loss

    # 在你的主模型 forward 方法中调用的函数
    def calculate_mate_contrastive_loss(self, encoder_outputs, aspect_spans, sentence_mask, tokenizer):
        """
        计算MATE任务的噪声对比学习损失。
        负样本是在嵌入空间中直接生成的随机向量。
        """
        device = encoder_outputs.device
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        # print("size", encoder_outputs.size())
        # 1. 提取所有正样本span的表示 (Anchors & Positives)
        #    这部分和之前完全一样，通过数据增强生成
        anchor_features = []
        positive_features = []
        # 假设 self.span_augmentor 是一个 nn.Sequential(nn.Linear(D, D), nn.Dropout(0.1))
        # print("aspect_spans", aspect_spans, aspect_spans.size())
        # print("input_ids", sentence_mask, sentence_mask.size())
        num_aspect = (aspect_spans.size(1) - 4) / 2 - 1
        for i in range(batch_size):
            for mask in sentence_mask:
                st = torch.nonzero(mask)[0].item()
            for aspect_spans_tmp in aspect_spans:
                for j in range(int(num_aspect)):
                    start = aspect_spans_tmp[3 + 2 * j].item()
                    end = aspect_spans_tmp[4 + 2 * j].item()
                    if start == 1:
                        continue
                    start -= 6
                    end -= 6  # 6 是len(self.mapping2targetid) + 2
                    start += st
                    end += st
                    # print("下标", start, end, st)
                    span_vector = encoder_outputs[i, start:end + 1, :].mean(dim=0)
                    # print("span", span_vector.size(), encoder_outputs[i, start:end + 1, :].size())
                    self.span_augmentor.train()
                    anchor_z = self.span_augmentor(span_vector)
                    positive_z = self.span_augmentor(span_vector)

                    anchor_features.append(anchor_z)
                    positive_features.append(positive_z)

        if not anchor_features:
            return torch.tensor(0.0, device=device)

        anchors = torch.stack(anchor_features)
        positives = torch.stack(positive_features)

        # --- 1. 批量生成所有需要的随机Token ID序列 ---
        num_neg_per_anchor = 10
        min_len, max_len = 10, 20
        num_total_anchors = len(anchors)
        total_neg_samples = num_total_anchors * num_neg_per_anchor

        # a. 确定每个负样本的长度
        #    生成一个包含 total_neg_samples 个随机长度的tensor
        neg_lengths = torch.randint(min_len, max_len + 1, (total_neg_samples,), device=device)

        # b. 创建一个大的padding张量来存放所有的随机ID
        #    形状是 [total_neg_samples, max_len]
        #    假设你的padding_id是0
        padding_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
        neg_input_ids = torch.full((total_neg_samples, max_len), padding_id, dtype=torch.long, device=device)

        # c. 填充随机ID
        #    我们只在每个序列的有效长度内填充随机ID
        #    创建一个mask来帮助我们做这件事
        arange_mask = torch.arange(max_len, device=device)[None, :] < neg_lengths[:, None]
        # arange_mask 的形状是 [total_neg_samples, max_len]，True代表需要填充随机ID

        # 从词汇表中随机采样足够多的ID
        # 假设 self.tokenizer.vocab_size 是你的词汇表大小
        num_tokens_to_generate = arange_mask.sum()
        random_ids = torch.randint(0, self.tokenizer.vocab_size, (num_tokens_to_generate,), device=device)

        # 将随机ID填充到正确的位置
        neg_input_ids[arange_mask] = random_ids

        # --- 2. 批量编码所有负样本 ---
        with torch.no_grad():
            # a. 创建 attention_mask
            #    arange_mask 本身就可以作为attention_mask（需要转换类型）
            neg_attention_mask = arange_mask.int()

            # b. 将随机ID序列送入Encoder
            #    假设 self.bart_encoder 是你的BART Encoder模块
            encoder_outputs = self.bart_encoder(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask
            )
            neg_embeds = encoder_outputs[0]  # Shape: [N * 10, max_len, D]

            # c. 进行批量平均池化，得到最终的负样本向量
            mask_expanded = neg_attention_mask.unsqueeze(-1).expand(neg_embeds.size()).float()
            sum_embeds = torch.sum(neg_embeds * mask_expanded, dim=1)
            # 使用我们之前生成的neg_lengths来做分母
            sum_mask = torch.clamp(neg_lengths.unsqueeze(-1), min=1e-9)
            neg_vectors_flat = sum_embeds / sum_mask  # Shape: [N * 10, D]

        # --- 3. 重塑负样本以匹配anchors ---
        negatives = neg_vectors_flat.view(
            num_total_anchors,
            num_neg_per_anchor,
            -1  # feature_dim
        )  # Final shape: [N, 10, D]

        # # a. 为每个负样本序列随机确定一个长度
        # neg_lengths = torch.randint(min_neg_len, max_neg_len + 1,
        #                             (len(anchors), num_neg_samples),
        #                             device=device)
        #
        # # b. 直接生成随机的嵌入向量，而不是随机ID
        # #    torch.randn会生成一个服从标准正态分布的随机张量
        # #    其形状模拟了(总aspect数, 负样本数, 最大长度, 特征维度)
        # random_embeds = torch.randn(len(anchors), num_neg_samples, max_neg_len, hidden_dim,
        #                             device=device)
        #
        # # c. 像之前一样，进行可变长度的平均池化
        # neg_mask = torch.arange(max_neg_len, device=device)[None, None, :] < neg_lengths[:, :, None]
        # random_embeds *= neg_mask.unsqueeze(-1)
        #
        # negatives = random_embeds.sum(dim=2) / torch.clamp(neg_lengths.unsqueeze(-1), min=1)
        # negatives 的形状是 [Total_Aspects, N_neg, D]

        # 3. 后续步骤完全一样：投影 -> 归一化 -> InfoNCE损失
        anchors_proj = self.mate_projection_head(anchors)
        positives_proj = self.mate_projection_head(positives)

        negatives_flat = negatives.view(-1, hidden_dim)
        negatives_proj_flat = self.mate_projection_head(negatives_flat)
        negatives_proj = negatives_proj_flat.view(len(anchors), num_neg_per_anchor, -1)

        anchors_norm = F.normalize(anchors_proj)
        positives_norm = F.normalize(positives_proj)
        negatives_norm = F.normalize(negatives_proj, dim=-1)

        l_pos = torch.einsum('nc,nc->n', [anchors_norm, positives_norm]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nsc->ns', [anchors_norm, negatives_norm])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.mate_contrastive_temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def _get_noun_embedding(self, noun: str) -> torch.Tensor:
        """将名词转换为嵌入向量

        Args:
            noun (str): 输入名词（如 "person", "object"）

        Returns:
            torch.Tensor: 嵌入向量 [hidden_size]
        """
        # Step 1: 分词
        noun_tokens = self.tokenizer._base_tokenizer.tokenize(noun, add_prefix_space=True)

        # Step 2: 处理空token（如特殊符号）
        if not noun_tokens:
            return torch.zeros(768, device=self.device)

        # Step 3: 转换为token IDs
        input_ids = self.tokenizer._base_tokenizer.convert_tokens_to_ids(noun_tokens)
        input_ids = torch.tensor([input_ids], device=self.device)  # [1, seq_len]

        # # Step 4: 截断超长序列
        # max_length = self.bart_encoder.config.max_position_embeddings
        # input_ids = input_ids[:, :max_length]

        # Step 5: 生成注意力掩码
        attention_mask = torch.ones_like(input_ids, device=self.device)

        # Step 6: 编码获取隐藏状态
        with torch.no_grad():
            outputs = self.bart_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs[0]  # [1, seq_len, hidden_size]

        # Step 7: 池化（取均值）
        return hidden_states.mean(dim=1).squeeze(0)  # [hidden_size]

    # def batch_encode_nouns(self, noun_list, device):
    #     """批量编码名词"""
    #     unique_nouns = list({noun for sublist in noun_list for noun in sublist})
    #     inputs = self.tokenizer._base_tokenizer(
    #         unique_nouns,
    #         add_prefix_space=True,
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(device)
    #
    #     with torch.no_grad():
    #         outputs = self.bart_encoder(**inputs)
    #         embeddings = outputs[0].mean(dim=1)
    #
    #     # 更新缓存
    #     for noun, emb in zip(unique_nouns, embeddings):
    #         self.noun_cache[noun] = emb.detach()
    #
    #     return {noun: self.noun_cache[noun] for noun in unique_nouns}
    #
    # def process_batch(self, caption_nouns, sentence_nouns, device):
    #     # 预加载默认名词嵌入
    #     for noun in ["person", "object"]:
    #         if noun not in self.noun_cache:
    #             self.noun_cache[noun] = self._get_noun_embedding(noun)
    #
    #     all_nouns = [cap + sent for cap, sent in zip(caption_nouns, sentence_nouns)]
    #     if all_nouns is not None:
    #         batch_embeddings = self.batch_encode_nouns(all_nouns, device)
    #     # 批量处理逻辑
    #     merged_nouns = []
    #     for cap_nouns, sent_nouns in zip(caption_nouns, sentence_nouns):
    #         # 从缓存获取嵌入
    #         cap_embs = [self.noun_cache[n] for n in cap_nouns if self.noun_cache[n] is not None]
    #         sent_embs = [self.noun_cache[n] for n in sent_nouns if self.noun_cache[n] is not None]
    #
    #         # 批量计算相似度矩阵
    #         if cap_embs and sent_embs:
    #             cap_matrix = torch.stack(cap_embs).to(device)
    #             sent_matrix = torch.stack(sent_embs).to(device)
    #             sim_matrix = cosine_similarity(
    #                 cap_matrix.unsqueeze(1),
    #                 sent_matrix.unsqueeze(0),
    #                 dim=-1
    #             )
    #             mean_sims = sim_matrix.mean(dim=1)
    #             valid_mask = mean_sims >= self.cosine_threshold
    #             keep_cap = [cap_nouns[i] for i, m in enumerate(valid_mask) if m]
    #         else:
    #             keep_cap = []
    #
    #         # 修正1：固定填充两个默认名词
    #         merged = list(set(keep_cap + sent_nouns))
    #         if not merged:
    #             merged = ["person", "object"]  # 直接赋值
    #
    #         merged_nouns.append(merged)
    #
    #     # 动态计算最大长度
    #     max_len = max(len(nouns) for nouns in merged_nouns) or 2  # 保证最小长度
    #
    #     # 修正2：安全初始化张量
    #     noun_embeds = torch.randn(  # 随机初始化代替全零
    #         len(merged_nouns), max_len, 768,
    #         device=device
    #     )
    #     noun_mask = torch.ones(
    #         len(merged_nouns), max_len,
    #         dtype=torch.bool, device=device
    #     )
    #
    #     # 修正3：带缓存的嵌入生成
    #     for i, nouns in enumerate(merged_nouns):
    #         current_embs = []
    #         for n in nouns:
    #             if n not in self.noun_cache:
    #                 self.noun_cache[n] = self._get_noun_embedding(n)
    #             current_embs.append(self.noun_cache[n])
    #
    #         if current_embs:
    #             stacked = torch.stack(current_embs)
    #             noun_embeds[i, :len(stacked)] = stacked
    #             noun_mask[i, :len(stacked)] = False
    #         else:  # 双重保障
    #             noun_embeds[i, 0] = self.noun_cache["person"]
    #             noun_mask[i, 0] = False
    #             noun_embeds[i, 1] = self.noun_cache["object"]
    #             noun_mask[i, 1] = False
    #
    #     return noun_embeds, noun_mask
    def batch_encode_nouns(self, all_nouns, device):
        """批量编码名词，确保安全处理空值"""
        # 检查输入是否为空
        if not all_nouns or all(not nouns for nouns in all_nouns):
            # 如果全部为空，返回默认名词的嵌入
            default_nouns = ["person", "object"]
            default_embeds = torch.stack([self.noun_cache[n] for n in default_nouns]).to(device)
            return default_embeds.unsqueeze(0)  # [1, 2, 768]

        # 收集所有唯一名词
        unique_nouns = set()
        for nouns in all_nouns:
            unique_nouns.update(nouns)

        # 确保默认名词在缓存中
        for noun in ["person", "object"]:
            if noun not in self.noun_cache:
                self.noun_cache[noun] = self._get_noun_embedding(noun)
            unique_nouns.add(noun)  # 确保默认名词在列表中

        unique_nouns = list(unique_nouns)

        # 安全检查：确保列表不为空
        if not unique_nouns:
            unique_nouns = ["person", "object"]

        try:
            # 批量编码所有唯一名词
            inputs = self.tokenizer._base_tokenizer(
                unique_nouns,
                add_prefix_space=True,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.bart_encoder(**inputs)
                embeddings = outputs[0].mean(dim=1)  # [num_unique, hidden_size]

            # 更新缓存
            for noun, emb in zip(unique_nouns, embeddings):
                self.noun_cache[noun] = emb.detach().cpu()  # 存储在CPU上节省GPU内存

            # 构建批次嵌入
            batch_size = len(all_nouns)
            max_nouns = max(len(nouns) if nouns else 2 for nouns in all_nouns)  # 至少为2，容纳默认名词
            batch_embeddings = torch.zeros((batch_size, max_nouns, 768), device=device)

            for i, nouns in enumerate(all_nouns):
                if not nouns:  # 如果为空，使用默认名词
                    batch_embeddings[i, 0] = self.noun_cache["person"].to(device)
                    batch_embeddings[i, 1] = self.noun_cache["object"].to(device)
                else:
                    for j, noun in enumerate(nouns[:max_nouns]):
                        batch_embeddings[i, j] = self.noun_cache[noun].to(device)

            return batch_embeddings

        except Exception as e:
            print(f"批量编码名词时出错: {e}")
            # 出错时返回默认嵌入
            default_embeds = torch.stack([
                self.noun_cache["person"].to(device),
                self.noun_cache["object"].to(device)
            ])  # [2, 768]
            return default_embeds.unsqueeze(0).expand(len(all_nouns), 2, 768)  # [batch_size, 2, 768]

    def process_batch(self, caption_nouns, sentence_nouns, device):
        """处理一批名词，确保安全处理空值"""
        # 确保输入不为None
        if caption_nouns is None or sentence_nouns is None:
            # 使用默认名词
            batch_size = 1
            if caption_nouns is not None:
                batch_size = len(caption_nouns)
            elif sentence_nouns is not None:
                batch_size = len(sentence_nouns)

            # 创建默认嵌入和掩码
            for noun in ["person", "object"]:
                if noun not in self.noun_cache:
                    self.noun_cache[noun] = self._get_noun_embedding(noun)

            default_embeds = torch.stack([
                self.noun_cache["person"].to(device),
                self.noun_cache["object"].to(device)
            ])  # [2, 768]

            batch_embeds = default_embeds.unsqueeze(0).expand(batch_size, 2, 768)  # [batch_size, 2, 768]
            batch_mask = torch.zeros((batch_size, 2), dtype=torch.bool, device=device)  # 不掩码，全部可见
            return batch_embeds, batch_mask

        # 标准化输入，确保它们是列表的列表
        batch_size = len(caption_nouns)
        safe_caption_nouns = []
        safe_sentence_nouns = []

        for i in range(batch_size):
            cap_nouns = caption_nouns[i] if i < len(caption_nouns) and caption_nouns[i] is not None else []
            sent_nouns = sentence_nouns[i] if i < len(sentence_nouns) and sentence_nouns[i] is not None else []

            # 过滤掉None和空字符串
            cap_nouns = [n for n in cap_nouns if n is not None and n != ""]
            sent_nouns = [n for n in sent_nouns if n is not None and n != ""]

            safe_caption_nouns.append(cap_nouns)
            safe_sentence_nouns.append(sent_nouns)

        # 合并名词并确保默认名词在缓存中
        for noun in ["person", "object"]:
            if noun not in self.noun_cache:
                self.noun_cache[noun] = self._get_noun_embedding(noun)

        # 处理每个样本的名词
        merged_nouns = []
        for cap_nouns, sent_nouns in zip(safe_caption_nouns, safe_sentence_nouns):
            # 合并并去重
            merged = list(set(cap_nouns + sent_nouns))

            # 如果合并后为空，使用默认名词
            if not merged:
                merged = ["person", "object"]

            merged_nouns.append(merged)

        # 获取嵌入
        batch_embeddings = self.batch_encode_nouns(merged_nouns, device)

        # 创建掩码 (False表示有效位置，True表示需要被掩码的位置)
        max_nouns = batch_embeddings.size(1)
        batch_mask = torch.ones((batch_size, max_nouns), dtype=torch.bool, device=device)

        for i, nouns in enumerate(merged_nouns):
            # 有效位置设为False
            batch_mask[i, :min(len(nouns), max_nouns)] = False

        return batch_embeddings, batch_mask


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new
