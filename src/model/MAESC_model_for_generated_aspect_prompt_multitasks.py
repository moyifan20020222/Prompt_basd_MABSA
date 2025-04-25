from typing import Optional, Tuple
from fastNLP.modules.torch.encoder import Seq2SeqEncoder
from fastNLP.modules.torch.decoder import Seq2SeqDecoder
from fastNLP.modules.torch import State
import torch
import torch.nn.functional as F
from torch import nn
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
            aspect_prompt_token_end_id=tokenizer.aspect_prompt_token_end_id
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

        only_sc = False
        # need_tag = True  #if predict the sentiment or not
        if args.task == 'twitter_ae':
            need_tag = False
        else:
            need_tag = True
            # if args.task == 'twitter_sc':
            #     only_sc = True
        # 这个是APD模块信息。 多增加一个参数，用于获取encoder的信息
        self.prompt_decoder = MultiModalBartDecoder_generate_aspect_prompt(self.config, share_decoder, encoder,
                                                                           args.Prompt_Pool_num,
                                                                           args.diversity_loss_weight,
                                                                           args.l2_reg_weight)

        if self.use_multitasks:  # 这个的意思是 AND 模块信息
            self.aspect_num_decoder = MultiModalBartDecoder_aspects_num(self.config, share_decoder, encoder,
                                                                        args.Prompt_Pool_num,
                                                                        args.diversity_loss_weight,
                                                                        args.l2_reg_weight)
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
        self.span_loss_fct = Span_loss()

        # MLM 损失， 用在测试阶段对模型进行微调，它的微调只对Aspect索引的token信息创建的参数使用。 所以他需要在Aspect索引decoder之后
        # 对他的参数进行修正
        self.mlm_loss_module = MultiModalBartDecoder_MLM(self.config, self.prompt_decoder.decoder)
        # 字幕与文本的相关度计算
        self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8,
                                                                  batch_first=True)
        self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        # 定义相关度阈值
        if args.dataset[0][0] == 'twitter15':
            self.threshold = 0.5
        elif args.dataset[0][0] == 'twitter17':
            self.threshold = 0.7

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
        dict_for_prompt = self.aspect_prompt_encoder(input_ids=input_ids,
                                                     image_features=image_features,
                                                     attention_mask=prompt_attention_mask,
                                                     output_hidden_states=True,
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
        relevance_scores_all = torch.full((batch_size, 1), 0.5,
                                          device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
        if num_valid > 0:
            # 2. 提取有效样本的数据
            valid_text_emb = text[valid_indices].to(attention_mask.device)  # [num_valid, seq_len_text, hidden_size]
            valid_caption_emb = image_caption[valid_indices].to(
                attention_mask.device)  # [num_valid, max_len, hidden_size]
            valid_text_mask = sentence_mask[valid_indices].to(attention_mask.device)  # [num_valid, max_len]
            # **关键: MultiheadAttention 的 key_padding_mask 需要 True 表示 Padding 位置**
            # 假设你的 mask 是 1 表示有效, 0 表示 padding, 需要转换
            valid_caption_padding_mask = (
                    image_caption_mask[valid_indices] == 0).to(
                attention_mask.device)  # [num_valid, max_len], True for padding

            # 3. 批处理交叉注意力 (假设 batch_first=True)
            # query: text, key/value: caption
            attn_output_valid, _ = self.text_caption_cross_attention(
                query=valid_text_emb,
                key=valid_caption_emb,
                value=valid_caption_emb,
                key_padding_mask=valid_caption_padding_mask  # ****** 提供正确的 Mask ******
            )  # Output: [num_valid, max_len, hidden_size]

            # 4. 批处理屏蔽平均池化
            # 将 padding 位置的 attention output 置零
            attn_output_valid_masked = attn_output_valid * valid_text_mask.unsqueeze(-1).float().to(
                attention_mask.device)
            # 计算每个样本的有效长度
            text_lengths_valid = valid_text_mask.sum(dim=1, keepdim=True).float().to(
                attention_mask.device)  # [num_valid, 1]
            # 对有效位置求和
            sum_attn_output_valid = attn_output_valid_masked.sum(dim=1).to(
                attention_mask.device)  # [num_valid, hidden_size]
            # 计算平均值，防止除零
            mean_attn_output_valid = sum_attn_output_valid / torch.clamp(text_lengths_valid,
                                                                         min=1e-9)  # [num_valid, hidden_size]

            # 5. 批处理线性投影，得到 Logits
            logits_valid = self.text_caption_attn_output_projection(mean_attn_output_valid)  # [num_valid, 1]

            # 6. 计算有效样本的相关性得分 (用于门控)
            scores_valid = torch.sigmoid(logits_valid).to(attention_mask.device)  # [num_valid, 1]

            # 7. 计算 CRD 损失 (仅针对有效样本)
            target_labels_valid = score[valid_indices].unsqueeze(1).to(attention_mask.device)  # [num_valid, 1]
            # binary_scores_valid = (target_labels_valid > self.threshold).float().unsqueeze(-1).to(attention_mask.device)
            # binary_logits_valid = (scores_valid > self.threshold).float().unsqueeze(-1).to(attention_mask.device)
            # print("二元判别：标注 and 计算的", binary_scores_valid, binary_logits_valid)
            # criterion_crd = nn.BCEWithLogitsLoss()
            criterion_crd = nn.MSELoss()
            loss_crd = criterion_crd(scores_valid, target_labels_valid)

            # 8. 更新完整批次的 logits 和 scores
            # 使用 scatter_ 或 index_put_ 更新特定索引的值
            crd_logits_all[valid_indices] = logits_valid
            relevance_scores_all[valid_indices] = scores_valid

        # 9. 使用相关性得分调整 image_features (向量化)
        # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
        gating_scores = relevance_scores_all.unsqueeze(1).to(attention_mask.device)  # 调整为 [batch_size, 1, 1] 以便广播
        # 或者如果 image_features 是 [batch_size, img_hidden]
        # gating_scores = relevance_scores_all # 调整为 [batch_size, 1]
        # print("gating_scores", gating_scores)
        # print("image_features", )
        for i in range(batch_size):
            image_features[i] = gating_scores[i] * image_features[i]

        weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # print("image的维度", image.size())
        # print("计算的token结果", weighted_image_tokens.size())
        # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        encoder_outputs = dict_for_prompt.last_hidden_state
        mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        dict_for_prompt.last_hidden_state = encoder_outputs.masked_scatter(
            mask_expanded_encoder,
            weighted_image_tokens[mask_expanded_encoder]
        )
        # print("计算的 权重 ： ", relevance_weights)
        # TODO: 有个问题， 后续的字幕信息我们也一起作为了encoder的输入， 它的信息如何处理， 方案1、因为这两者 表达内容接近，可以采用一个门控机制学习
        # 2、直接认为这个字幕信息就是一个中间内容，计算出图片和文本的相关度后就可以丢弃了， 那为了保证结构统一，可以直接在此乘0，或是mask为0即可。

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
            encoder_outputs_all=dict_for_prompt.hidden_states)
        # 生成的用于生成索引Prompt 的 总索引Prompt 这个时候还没有开始构建Prompt，所以AND的判别数目在下面
        generated_prompt = generated_prompt[:, 1:, :]  ##(batch_size, 2, 768)
        pseudo_loss = torch.tensor(0.0, dtype=torch.float)
        # 添加一个MLM损失，这个是一个伪CTTA方法，这个损失只在测试阶段使用，它的作用能迫使模型更好地理解测试数据的语言分布
        mlm_labels = mlm_message['mlm_labels'].to(input_ids.device)
        mlm_decoder_input_ids = mlm_message['mlm_decoder_input_ids'].to(input_ids.device)
        mlm_decoder_attention_mask = mlm_message['mlm_decoder_attention_mask'].to(input_ids.device)
        # mlm_inputs_id, mlm_labels = self.prepare_mlm(input_ids, attention_mask, self.tokenizer)
        pseudo_loss = self.mlm_loss_module(labels=mlm_labels, input_ids=input_ids,
                                           encoder_outputs=dict_for_prompt.last_hidden_state,
                                           attention_mask=attention_mask,
                                           decoder_input_ids=mlm_decoder_input_ids,
                                           decoder_attention_mask=mlm_decoder_attention_mask)

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
                                        encoder_outputs_all=dict_for_prompt.hidden_states)

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

        dict = self.encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            generated_prompt=generated_prompt,
            aspects_num=new_predict_aspects_num,
            output_hidden_states=True,
            return_dict=True)

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
        loss = [sparsity_loss_layers, sparsity_loss_layers_aspect, loss_crd]
        return state, aspects_num_loss, predict_aspects_num, pseudo_loss, loss

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
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        ### for prompt
        # import ipdb; ipdb.set_trace()

        ## for aspect-spans

        aspects_num = torch.tensor(aspects_num).to(input_ids.device)
        state, aspects_num_loss, predict_aspects_num, pseudo_loss, loss = self.prepare_state(input_ids, image_features,
                                                                                             attention_mask,
                                                                                             aesc_infos, aspects_num,
                                                                                             sentence_mask,
                                                                                             image_mask, mlm_message,
                                                                                             image_caption_valid,
                                                                                             image_caption_mask,
                                                                                             score)
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]

        logits = self.decoder(spans, state)  ## spans: (2, 13) logits: (2, 12, 40)

        span_loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

        all_loss = span_loss + self.loss_lambda * aspects_num_loss
        for i in range(len(loss)):
            all_loss = all_loss + loss[i]
        return all_loss, predict_aspects_num, pseudo_loss


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
