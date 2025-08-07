from collections import defaultdict
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
    MultiModalBartEncoder_for_Generating_Dual_prompts, MultiModalBartDecoder_generate_sentiment_prompt, \
    MultiModalBartDecoder_generate_aspect_prompt, MultiModalBartDecoder_MLM, ImageEmbedding, image_encoder_type, \
    image_model_name
from src.model.modules_for_prompt_multitasks import MultiModalBartDecoder_aspects_num


# 添加一个共享解码器基类
class SharedBartDecoder(nn.Module):
    """共享的BART解码器基类，用于减少参数量"""

    _instance = None  # 单例模式

    @classmethod
    def get_instance(cls, config, decoder):
        """获取共享解码器实例"""
        if cls._instance is None:
            cls._instance = decoder
        return cls._instance

    def __init__(self, config, decoder):
        super().__init__()
        self.decoder = self.get_instance(config, decoder)
        self.config = config


# 新增共享组件管理类
class SharedComponents:
    """管理模型中共享的组件，避免重复创建相同的模块"""

    def __init__(self):
        self.components = {}

    def get_or_create(self, component_type, component_name, create_fn, *args, **kwargs):
        """获取已存在的组件或创建新组件

        Args:
            component_type: 组件类型
            component_name: 组件名称
            create_fn: 创建组件的函数
            *args, **kwargs: 传递给create_fn的参数

        Returns:
            已存在或新创建的组件
        """
        key = f"{component_type}_{component_name}"
        if key not in self.components:
            self.components[key] = create_fn(*args, **kwargs)
        return self.components[key]


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

        # 考虑共用的 而非 各自编码 因为实验条件不允许构建太多的BART，这里共用一个刚好ok，如果你的环境ok 就分开吧
        # multimodal_encoder_for_generated_aspect_prompt = MultiModalBartEncoder(config, encoder,
        #                                                                        tokenizer.img_feat_id,
        #                                                                        tokenizer.cls_token_id,
        #                                                                        args.num_image_tokens)
        #
        # multimodal_encoder_for_generated_aspects_num = MultiModalBartEncoder(config, encoder,
        #                                                                      tokenizer.img_feat_id,
        #                                                                      tokenizer.cls_token_id,
        #                                                                      args.num_image_tokens)
        # 分为两种。 一种是大家共用一个encoder， 另一个是Aspect 和 Emotion拆开

        # multimodal_encoder_for_generated_senti_prompt = MultiModalBartEncoder(config, encoder,
        #                                                                       tokenizer.img_feat_id,
        #                                                                       tokenizer.cls_token_id,
        #                                                                       args.num_image_tokens)
        # multimodal_encoder_for_generated_aspect_prompt = MultiModalBartEncoder(config, encoder,
        #                                                                        tokenizer.img_feat_id,
        #                                                                        tokenizer.cls_token_id,
        #                                                                        args.num_image_tokens)
        multimodal_encoder_for_generated_aspect_prompt = MultiModalBartEncoder(config, encoder,
                                                                               tokenizer.img_feat_id,
                                                                               tokenizer.cls_token_id,
                                                                               args.num_image_tokens)

        multimodal_encoder_for_generated_aspects_num = multimodal_encoder_for_generated_aspect_prompt

        multimodal_encoder_for_generated_senti_prompt = multimodal_encoder_for_generated_aspect_prompt

        multimodal_encoder = MultiModalBartEncoder_for_Generating_Dual_prompts(
            use_generated_aspect_prompt=args.use_generated_aspect_prompt,
            use_generated_senti_prompt=args.use_generated_senti_prompt,
            config=config,
            encoder=encoder,
            img_feat_id=tokenizer.img_feat_id,
            aspect_prompt_token_id=tokenizer.aspect_prompt_token_id,
            senti_prompt_token_id=tokenizer.senti_prompt_token_id,
            cls_token_id=tokenizer.cls_token_id,
            num_image_tokens=args.num_image_tokens,
            use_different_aspect_prompt=args.use_different_aspect_prompt,
            use_different_senti_prompt=args.use_different_senti_prompt,
            NEU_id=tokenizer.neu_token_id,
            POS_id=tokenizer.pos_token_id,
            NEG_id=tokenizer.neg_token_id,
            aspect_prompt_token_front_id=tokenizer.aspect_prompt_token_front_id,
            aspect_prompt_token_end_id=tokenizer.aspect_prompt_token_end_id,
            is_few_shot=args.is_few_shot,
            prompt_pool_num=args.Prompt_Pool_num,
            diversity_loss_weight=args.diversity_loss_weight,
            l2_reg_weight=args.l2_reg_weight,
            args=args
        )
        return (multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder_for_generated_aspects_num,
                multimodal_encoder_for_generated_senti_prompt, multimodal_encoder, decoder, encoder)

    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.senti_prompt_token_id = tokenizer.senti_prompt_token_id
        label_ids = sorted(label_ids)
        multimodal_encoder_for_generated_aspect_prompt, multimodal_encoder_for_generated_aspects_num, multimodal_encoder_for_generated_senti_prompt, multimodal_encoder, share_decoder, encoder = self.build_model(
            args, bart_model, self.tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)

        self.aspect_prompt_encoder = multimodal_encoder_for_generated_aspect_prompt
        self.aspects_num_encoder = multimodal_encoder_for_generated_aspects_num
        self.senti_prompt_encoder = multimodal_encoder_for_generated_senti_prompt
        self.encoder = multimodal_encoder
        self.use_generated_senti_prompt = args.use_generated_senti_prompt
        self.use_multitasks = args.use_multitasks
        self.num_image_tokens = args.num_image_tokens
        self.loss_lambda = args.loss_lambda
        self.bart_encoder = encoder
        only_sc = False
        # need_tag = True  #if predict the sentiment or not
        if args.task == 'twitter_ae':
            need_tag = False
        else:
            need_tag = True
            # if args.task == 'twitter_sc':
            #     only_sc = True
        shared_bart_decoder = SharedBartDecoder.get_instance(self.config, share_decoder)
        # 创建共享的解码器组件
        shared_decoder_params = {
            'config': self.config,
            'decoder': shared_bart_decoder,
            'prompt_pool_num': args.Prompt_Pool_num,
            'diversity_loss_weight': args.diversity_loss_weight,
            'l2_reg_weight': args.l2_reg_weight
        }

        # self.aspect_prompt_decoder = MultiModalBartDecoder_generate_aspect_prompt(self.config, share_decoder, encoder,
        #                                                                           args.Prompt_Pool_num,
        #                                                                           args.diversity_loss_weight,
        #                                                                           args.l2_reg_weight,
        #                                                                           args.is_few_shot)
        #
        #
        # if self.use_generated_senti_prompt:
        #     self.senti_prompt_decoder = MultiModalBartDecoder_generate_sentiment_prompt(self.config, share_decoder,
        #                                                                                 args.Prompt_Pool_num,
        #                                                                                 args.diversity_loss_weight,
        #                                                                                 args.l2_reg_weight)
        # if self.use_multitasks:
        #     self.aspect_num_decoder = MultiModalBartDecoder_aspects_num(self.config, share_decoder, encoder,
        #                                                                 args.Prompt_Pool_num,
        #                                                                 args.diversity_loss_weight,
        #                                                                 args.l2_reg_weight,
        #                                                                 args.is_few_shot)

        # 创建方面提示解码器（添加encoder参数）
        self.aspect_prompt_decoder = MultiModalBartDecoder_generate_aspect_prompt(
            **shared_decoder_params,
            encoder=encoder,
            is_few_shot=args.is_few_shot
        )

        # 创建情感提示解码器（如果需要）
        if self.use_generated_senti_prompt:
            # 重用相同的解码器参数，但不传递encoder
            self.senti_prompt_decoder = MultiModalBartDecoder_generate_sentiment_prompt(
                **shared_decoder_params,
                is_few_shot=args.is_few_shot
            )

        # 创建方面数量解码器（如果需要）
        if self.use_multitasks:
            self.aspect_num_decoder = MultiModalBartDecoder_aspects_num(
                **shared_decoder_params,
                encoder=encoder,
                is_few_shot=args.is_few_shot
            )

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
        # self.aspect_num_linear = nn.Linear(768, 5) ##max_aspects_num=5
        self.mlm_loss_module = MultiModalBartDecoder_MLM(self.config, self.aspect_prompt_decoder.decoder)
        # 同样的，下方是全数据集上测试较好的结果
        # if not args.is_few_shot:
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8,
        #                                                               batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     self.text_caption_cross_attention_second = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                      num_heads=8,
        #                                                                      batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection_second = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #     self.nouns_cross_attention = nn.MultiheadAttention(768, 4, batch_first=True)
        #     self.gate_proj_nouns = nn.Linear(768 * 2, 1)
        #     self.nouns_cross_attention_image = nn.MultiheadAttention(768, 4, batch_first=True)
        #     self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)
        #
        #     # 因为原文为三个任务分别构建了三个encoder，（虽然三者共用的是一个encoder，但是我们还是拆分了三份，所以为了对应，
        #     # 同样的注意力机制我们也是用了三个，后续不用的话 就注释掉下文的注意力和线性层） 上面的是用与aspect识别的部分
        #     # 下面分别适应于情绪Prompt和 Aspect数量的encoder结果
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention_senti = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                     num_heads=8,
        #                                                                     batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection_senti = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     # self.text_caption_cross_attention_second_senti = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #     #                                                                        num_heads=8,
        #     #                                                                        batch_first=True, dropout=0.2)
        #     # # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     # self.text_caption_attn_output_projection_second_senti = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )
        #     # Aspect 数量的
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention_num = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                   num_heads=8,
        #                                                                   batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection_num = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     # self.text_caption_cross_attention_second_num = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #     #                                                                      num_heads=8,
        #     #                                                                      batch_first=True, dropout=0.2)
        #     # # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        #     # self.text_caption_attn_output_projection_second_num = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )
        #
        # # 少样本下的模型架构：
        # else:
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=4,
        #                                                               batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size + 1, 1)  # 示例：将池化后的输出投影到 1 维
        #     # self.text_caption_attn_output_projection = nn.Linear(768 + 1, 1)
        #     self.text_caption_attn_output_projection = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     self.text_caption_cross_attention_second = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                      num_heads=4,
        #                                                                      batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection_second = nn.Linear(self.config.hidden_size + 1, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection_second = nn.Linear(768 + 1, 1)
        #     # self.text_caption_attn_output_projection_second = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )
        #     self.nouns_cross_attention = nn.MultiheadAttention(768, 2, batch_first=True)
        #     self.gate_proj_nouns = nn.Linear(768 * 2, 1)
        #     self.nouns_cross_attention_image = nn.MultiheadAttention(768, 2, batch_first=True)
        #     self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)
        #
        #     # 因为原文为三个任务分别构建了三个encoder，（虽然三者共用的是一个encoder，但是我们还是拆分了三份，所以为了对应，
        #     # 同样的注意力机制我们也是用了三个，后续不用的话 就注释掉下文的注意力和线性层） 上面的是用与aspect识别的部分
        #     # 下面分别适应于情绪Prompt和 Aspect数量的encoder结果
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention_senti = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                     num_heads=4,
        #                                                                     batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection_senti = nn.Linear(self.config.hidden_size + 1, 1)  # 示例：将池化后的输出投影到 1 维
        #     self.text_caption_attn_output_projection_senti = nn.Sequential(
        #         nn.Linear(768 + 1, 256),
        #         nn.LayerNorm(256),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 1),
        #     )
        #
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     # self.text_caption_cross_attention_second_senti = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #     #                                                                        num_heads=2,
        #     #                                                                        batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection_second_senti = nn.Linear(self.config.hidden_size + 1,
        #     #                                                                   1)  # 示例：将池化后的输出投影到 1 维
        #
        #     # self.text_caption_attn_output_projection_second_senti = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )
        #     # Aspect 数量的
        #     # 字幕与文本的相关度计算
        #     self.text_caption_cross_attention_num = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #                                                                   num_heads=4,
        #                                                                   batch_first=True, dropout=0.2)
        #     self.text_caption_attn_output_projection_num = nn.Linear(self.config.hidden_size + 1,
        #                                                              1)  # 示例：将池化后的输出投影到 1 维
        #     # self.text_caption_attn_output_projection_num = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )
        #     # 字幕与文本的相关度计算 第二轮的encoder的
        #     # self.text_caption_cross_attention_second_num = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
        #     #                                                                      num_heads=2,
        #     #                                                                      batch_first=True, dropout=0.2)
        #     # self.text_caption_attn_output_projection_second_num = nn.Linear(self.config.hidden_size + 1,
        #     #                                                                 1)  # 示例：将池化后的输出投影到 1 维
        #
        #     # self.text_caption_attn_output_projection_second_num = nn.Sequential(
        #     #     nn.Linear(768 + 1, 256),
        #     #     nn.LayerNorm(256),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.2),
        #     #     nn.Linear(256, 1),
        #     # )

        # 定义相关度阈值  可以 把阈值变成可学习的参数， 与固定的阈值做一个比较，
        if args.dataset[0][0] == 'twitter15':
            self.threshold = nn.Parameter(torch.tensor(0.7))
        elif args.dataset[0][0] == 'twitter17':
            self.threshold = nn.Parameter(torch.tensor(0.5))

        # 温度参数，用于调整sigmoid的陡峭程度
        self.temperature = nn.Parameter(torch.tensor(5.0))

        # 定义字幕名词和文本名词的相关度阈值， 当低于这个阈值的时候，同样可以认为， 图像与文本并不一致，说明图片信息将是干扰。
        if args.dataset[0][0] == 'twitter15':
            self.cosine_threshold = 0.9
        elif args.dataset[0][0] == 'twitter17':
            self.cosine_threshold = 0.9

        self.noun_cache = defaultdict(lambda: None)  # 名词嵌入缓存

        # 原型表示 提升情绪识别部分：
        self.prototypes = nn.Parameter(torch.randn(3, 768))
        self.sentiment_id_to_idx_map = {3: 0, 4: 1, 5: 2}
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(),
            nn.Linear(768 // 2, 768)
        )

        # **新增：用于原型空间内特征增强的模块**
        # 例如，一个小型 MLP 或注意力机制，用于融合投影特征和其对应的原型
        self.prototype_enhancement_mlp = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),  # 输入是拼接的 (projected_feat, prototype)
            nn.ReLU(),
            nn.Linear(768 * 2, 768)  # 输出增强后的特征，维度与原型一致
        )
        # 或者使用注意力：让投影特征去关注对应的原型
        # self.prototype_enhancement_attention = nn.MultiheadAttention(embed_dim=prototype_dim, num_heads=...)

        self.prototype_to_encoder_projection = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(),
            nn.Linear(768 // 2, 768)
        )
        self.proto_fusion_gate_linear = nn.Linear(768 * 2, 1)
        self.temperature_prototype = 0.5
        self.prototype_enhancement_attention = nn.MultiheadAttention(768, 4, 0.2)
        if args.dataset[0][0] == 'twitter15':
            self.contrastive_loss = SupConLoss(0.05, 0.05)
        else:
            self.contrastive_loss = SupConLoss(0.1, 0.1)
        self.datasets = args.dataset[0][0]
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),  # BatchNorm is often helpful
            nn.ReLU(inplace=True),
            nn.Linear(768, 768)
        )
        # e. 用于自适应温度的移动平均值
        self.register_buffer('running_loss_mean', torch.tensor(0.0))
        self.register_buffer('running_loss_std', torch.tensor(1.0))
        self.ema_decay = 0.99

    def prepare_state(self,
                      input_ids,
                      image_features,
                      attention_mask=None,
                      aesc_infos=None,
                      aspects_nums=None,
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
        # import ipdb; ipdb.set_trace()
        # tokenization_new_for_generated_prompt_multitasks.py 的571 和 582行定义了Prompt的最大长度，
        # 总体为 img的起始和结束token + num_image_token个数 + 图像描述(20) + Aspect所有内容的部分(40) 这里最多就是35 还没有超
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
        '''dict_for_senti_prompt'''
        dict_for_aspect_prompt, loss_aspect, aspect_scores, hard_mask_aspect = self.aspect_prompt_encoder(
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

        # dict_for_aspects_num = self.aspects_num_encoder(input_ids=input_ids,
        #                                                 image_features=image_features,
        #                                                 attention_mask=prompt_attention_mask,
        #                                                 output_hidden_states=True,
        #                                                 return_dict=True)
        #
        dict_for_senti_prompt, loss_senti, senti_scores, hard_mask_senti = self.senti_prompt_encoder(
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
        hard_mask_aspect = hard_mask_aspect.to(image_caption_mask.device)
        image_caption_mask_aspect = image_caption_mask * hard_mask_aspect
        hard_mask_senti = hard_mask_senti.to(image_caption_mask.device)
        image_caption_mask_senti = image_caption_mask * hard_mask_senti
        #  通过计算字幕与文本的相关度，让图片特征的信息相应的减少
        # print("image_caption_mask:", image_caption_mask)
        image_caption_mask_expanded_for_image = image_caption_mask.unsqueeze(-1).to(
            attention_mask.device)  # 为图像 mask 扩展维度
        image_caption_embeddings = dict_for_aspect_prompt.last_hidden_state * image_caption_mask_expanded_for_image.float()  # 提取图像嵌入
        image_caption = image_caption_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value

        # 文本嵌入
        sentence_mask_expanded_for_image = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        text_embeddings = dict_for_aspect_prompt.last_hidden_state * sentence_mask_expanded_for_image.float()  # 提取图像嵌入
        text = text_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # 图像嵌入
        image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        image_embeddings = dict_for_aspect_prompt.last_hidden_state * image_mask_expanded_for_image.float()  # 提取图像嵌入
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
        # TODO: 5.15 在MABSA任务上有些难度，我们从头开始添加模块，先把相关度拿掉，看看是不是相关度限制了模型性能

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
        #     attn_output_valid, _ = self.text_caption_cross_attention(
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
        #     # loss_crd = 0.9 * loss_crd_soft + 0.1 * loss_crd_hard
        #     loss_aspect = loss_crd_soft
        #     # 8. 更新完整批次的 logits 和 scores
        #     # 使用 scatter_ 或 index_put_ 更新特定索引的值
        #     crd_logits_all[valid_indices] = logits_valid
        #     relevance_scores_all[valid_indices] = scores_valid
        #
        # # 9. 使用相关性得分调整 image_features (向量化)
        # # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
        # threshold = torch.sigmoid(self.threshold)
        # temperature = torch.abs(self.temperature) + 1.0  # 确保温度参数为正
        #
        # # 平滑阈值处理
        # adjusted_scores = torch.sigmoid((relevance_scores_all - threshold) * temperature)
        #
        # # 硬阈值+软权重结合
        # hard_mask = (relevance_scores_all > adjusted_scores).float()
        # gating_scores = relevance_scores_all * hard_mask
        # gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)
        # # gating_scores = relevance_scores_all.unsqueeze(1).to(attention_mask.device)  # 调整为 [batch_size, 1, 1] 以便广播
        # # 或者如果 image_features 是 [batch_size, img_hidden]
        # # gating_scores = relevance_scores_all # 调整为 [batch_size, 1]
        # # print("gating_scores", gating_scores)
        # # print("image_features", )
        # for i in range(batch_size):
        #     image_features[i] = gating_scores[i] * image_features[i]
        # gating_scores = aspect_scores
        # weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # # # print("image的维度", image.size())
        # # # print("计算的token结果", weighted_image_tokens.size())
        # # # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        # encoder_outputs = dict_for_aspect_prompt.last_hidden_state
        # mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_aspect_prompt.last_hidden_state = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     weighted_image_tokens[mask_expanded_encoder]
        # )
        #
        # # 字幕修正
        # weight_image_caption = image_caption * gating_scores
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(
        #     attention_mask.device)
        # dict_for_aspect_prompt.last_hidden_state = dict_for_aspect_prompt.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )
        #
        # # -------------------------------
        #
        #
        # 同样的操作为情绪的encoder结果 和 Aspect num的encoder结果操作一次。
        image_caption_mask_expanded_for_image = image_caption_mask.unsqueeze(-1).to(
            attention_mask.device)  # 为图像 mask 扩展维度
        image_caption_embeddings = dict_for_senti_prompt.last_hidden_state * image_caption_mask_expanded_for_image.float()  # 提取图像嵌入
        image_caption = image_caption_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value

        # 文本嵌入
        sentence_mask_expanded_for_image = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        text_embeddings = dict_for_senti_prompt.last_hidden_state * sentence_mask_expanded_for_image.float()  # 提取图像嵌入
        text = text_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # 图像嵌入
        image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        image_embeddings = dict_for_senti_prompt.last_hidden_state * image_mask_expanded_for_image.float()  # 提取图像嵌入
        image = image_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # relevance_scores = torch.zeros(image_caption_mask.size(0), 1, device=input_ids.device)  # 初始化相关度得分张量 (形状 [b, 1])
        # # print("image_caption_valid 维度", image_caption_mask.size())
        # # print("图片特征长度", len(image_features))
        # loss_crd_senti = torch.tensor(0.0, dtype=torch.float)
        # valid_indices = torch.where(image_caption_valid)[0]
        # # print("valid_indices", valid_indices.size())
        # # print("score", score.size())
        # num_valid = len(valid_indices)
        # batch_size = image_caption_mask.size(0)
        # # 使用 logits 计算损失，使用 scores 进行门控
        # crd_logits_all = torch.zeros(batch_size, 1, device=attention_mask.device)  # 默认logit为0，对应sigmoid(0)=0.5
        # relevance_scores_all = torch.full((batch_size, 1), 0.5,
        #                                   device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
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
        #     attn_output_valid, _ = self.text_caption_cross_attention_senti(
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
        #     logits_valid = self.text_caption_attn_output_projection_senti(combined_features)  # [num_valid, 1]
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
        #     # 5.29 权重0.6 0.4 -> 0.9 0.1
        #     # loss_crd_senti = 0.9 * loss_crd_soft + 0.1 * loss_crd_hard
        #     loss_senti = loss_crd_soft
        #     # 8. 更新完整批次的 logits 和 scores
        #     # 使用 scatter_ 或 index_put_ 更新特定索引的值
        #     crd_logits_all[valid_indices] = logits_valid
        #     relevance_scores_all[valid_indices] = scores_valid
        #
        # # 9. 使用相关性得分调整 image_features (向量化)
        # # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
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
        # 直接使用ITM结果
        # print("wd", score.size())
        # gating_scores = score.unsqueeze(-1).unsqueeze(-1).to(attention_mask.device)
        # print("wd", score.size())
        # gating_scores = relevance_scores_all.unsqueeze(1).to(attention_mask.device)  # 调整为 [batch_size, 1, 1] 以便广播
        # 或者如果 image_features 是 [batch_size, img_hidden]
        # gating_scores = relevance_scores_all # 调整为 [batch_size, 1]
        # print("gating_scores", gating_scores)
        # print("image_features", )
        # for i in range(batch_size):
        #     image_features[i] = gating_scores[i] * image_features[i]
        # hard_mask = hard_mask.to(image_mask.device)

        # gating_scores = senti_scores
        # weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # # image_mask = image_mask * hard_mask
        # # print("image的维度", image.size())
        # # print("计算的token结果", weighted_image_tokens.size())
        # # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        # encoder_outputs = dict_for_senti_prompt.last_hidden_state
        # mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_senti_prompt.last_hidden_state = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     weighted_image_tokens[mask_expanded_encoder]
        # )
        #
        # # 字幕修正
        # weight_image_caption = image_caption * gating_scores
        # # image_caption_mask = image_caption_mask * hard_mask
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(
        #     attention_mask.device)
        # dict_for_senti_prompt.last_hidden_state = dict_for_senti_prompt.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )

        #
        # # 最后是对Aspect——num的进行操作
        # image_caption_mask_expanded_for_image = image_caption_mask.unsqueeze(-1).to(
        #     attention_mask.device)  # 为图像 mask 扩展维度
        # image_caption_embeddings = dict_for_aspects_num.last_hidden_state * image_caption_mask_expanded_for_image.float()  # 提取图像嵌入
        # image_caption = image_caption_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        #
        # # 文本嵌入
        # sentence_mask_expanded_for_image = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        # text_embeddings = dict_for_aspects_num.last_hidden_state * sentence_mask_expanded_for_image.float()  # 提取图像嵌入
        # text = text_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # # 图像嵌入
        # image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        # image_embeddings = dict_for_aspects_num.last_hidden_state * image_mask_expanded_for_image.float()  # 提取图像嵌入
        # image = image_embeddings.to(attention_mask.device)  # [s, b, 768]  图像字幕嵌入，准备作为 Key/Value
        # relevance_scores = torch.zeros(image_caption_mask.size(0), 1, device=input_ids.device)  # 初始化相关度得分张量 (形状 [b, 1])
        # # print("image_caption_valid 维度", image_caption_mask.size())
        # # print("图片特征长度", len(image_features))
        # loss_crd_aspect_num = torch.tensor(0.0, dtype=torch.float)
        # valid_indices = torch.where(image_caption_valid)[0]
        # # print("valid_indices", valid_indices.size())
        # # print("score", score.size())
        # num_valid = len(valid_indices)
        # batch_size = image_caption_mask.size(0)
        # # 使用 logits 计算损失，使用 scores 进行门控
        # crd_logits_all = torch.zeros(batch_size, 1, device=attention_mask.device)  # 默认logit为0，对应sigmoid(0)=0.5
        # relevance_scores_all = torch.full((batch_size, 1), 0.5,
        #                                   device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
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
        #     attn_output_valid, _ = self.text_caption_cross_attention(
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
        #     loss_crd_aspect_num = 0.6 * loss_crd_soft + 0.4 * loss_crd_hard
        #     # 8. 更新完整批次的 logits 和 scores
        #     # 使用 scatter_ 或 index_put_ 更新特定索引的值
        #     crd_logits_all[valid_indices] = logits_valid
        #     relevance_scores_all[valid_indices] = scores_valid
        #
        # # 9. 使用相关性得分调整 image_features (向量化)
        # # 确定广播维度，假设 image_features 是 [batch_size, num_patches, img_hidden]
        # threshold = torch.sigmoid(self.threshold)
        # temperature = torch.abs(self.temperature) + 1.0  # 确保温度参数为正
        #
        # # 平滑阈值处理
        # adjusted_scores = torch.sigmoid((relevance_scores_all - threshold) * temperature)
        #
        # # 硬阈值+软权重结合
        # hard_mask = (relevance_scores_all > adjusted_scores).float()
        # gating_scores = relevance_scores_all * hard_mask
        # gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)
        # # gating_scores = relevance_scores_all.unsqueeze(1).to(attention_mask.device)  # 调整为 [batch_size, 1, 1] 以便广播
        # # 或者如果 image_features 是 [batch_size, img_hidden]
        # # gating_scores = relevance_scores_all # 调整为 [batch_size, 1]
        # # print("gating_scores", gating_scores)
        # # print("image_features", )
        # for i in range(batch_size):
        #     image_features[i] = gating_scores[i] * image_features[i]
        #
        # weighted_image_tokens = image * gating_scores  # 逐元素相乘进行加权
        # # print("image的维度", image.size())
        # # print("计算的token结果", weighted_image_tokens.size())
        # # print("dict_for_prompt.last_hidden_state", dict_for_prompt.last_hidden_state.size())
        # encoder_outputs = dict_for_aspects_num.last_hidden_state
        # mask_expanded_encoder = image_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(attention_mask.device)
        # dict_for_aspects_num.last_hidden_state = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     weighted_image_tokens[mask_expanded_encoder]
        # )
        #
        # # 字幕修正
        # weight_image_caption = image_caption * gating_scores
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(
        #     attention_mask.device)
        # dict_for_aspects_num.last_hidden_state = dict_for_aspects_num.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )

        # print("计算的 权重 ： ", relevance_weights)
        # TODO: 有个问题， 后续的字幕信息我们也一起作为了encoder的输入， 它的信息如何处理， 方案1、因为这两者 表达内容接近，可以采用一个门控机制学习
        # 2、直接认为这个字幕信息就是一个中间内容，计算出图片和文本的相关度后就可以丢弃了， 那为了保证结构统一，可以直接在此乘0，或是mask为0即可。

        # --------------
        noun_embeds, noun_mask = self.process_batch(caption_nouns, sentence_nouns, attention_mask.device)

        '''generated_aspect_prompt'''
        aspect_prompt_decoder_input_ids, aspect_prompt_decoder_attention_mask = [
            aesc_infos['aspect_prompt_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspect_prompt_decoder_attention_mask'].to(input_ids.device)]

        generated_aspect_prompt, sparsity_loss_layers_aspect, sparsity_loss_image_aspect = self.aspect_prompt_decoder(
            encoder_outputs=dict_for_aspect_prompt.last_hidden_state,
            attention_mask=attention_mask,
            decoder_input_ids=aspect_prompt_decoder_input_ids,
            decoder_attention_mask=aspect_prompt_decoder_attention_mask,
            sentence_mask=sentence_mask,
            image_mask=image_mask,
            encoder_outputs_all=dict_for_aspect_prompt.hidden_states,
            nouns_embeds=noun_embeds,
            nouns_mask=noun_mask,
            image_caption_valid=image_caption_valid,
            image_caption_mask=image_caption_mask,
            score=aspect_scores)
        generated_aspect_prompt = generated_aspect_prompt[:, 1:, :]  ##(batch_size, 2, 768)

        # APD的数据只有两个。 就是用于生成索引的Prompt 只有一个， 但是为了生成不同Aspect的索引Prompt可以使用多个MLP
        # 所以我们的Prompt是使用在这个总索引Prompt 的生成上， 而非下一个阶段Encoder生成对应于每一个Aspect的索引Prompt上
        pseudo_loss = torch.tensor(0.0, dtype=torch.float)
        # 添加一个MLM损失，这个是一个伪CTTA方法，这个损失只在测试阶段使用，它的作用能迫使模型更好地理解测试数据的语言分布
        # mlm_labels = mlm_message['mlm_labels'].to(input_ids.device)
        # mlm_decoder_input_ids = mlm_message['mlm_decoder_input_ids'].to(input_ids.device)
        # mlm_decoder_attention_mask = mlm_message['mlm_decoder_attention_mask'].to(input_ids.device)
        # # mlm_inputs_id, mlm_labels = self.prepare_mlm(input_ids, attention_mask, self.tokenizer)
        # pseudo_loss = self.mlm_loss_module(labels=mlm_labels, input_ids=input_ids,
        #                                    encoder_outputs=dict_for_aspect_prompt.last_hidden_state,
        #                                    attention_mask=attention_mask,
        #                                    decoder_input_ids=mlm_decoder_input_ids,
        #                                    decoder_attention_mask=mlm_decoder_attention_mask)
        # import ipdb; ipdb.set_trace()
        '''aspects_num'''
        aspects_num_decoder_input_ids, aspects_num_decoder_attention_mask = [
            aesc_infos['aspects_num_decoder_input_ids'].to(input_ids.device),
            aesc_infos['aspects_num_decoder_attention_mask'].to(input_ids.device)]

        # import ipdb; ipdb.set_trace()
        if self.use_multitasks:
            aspects_num_loss, predict_aspects_num_logits, sparsity_loss_layers_aspect_num, sparsity_loss_image_aspect_num = self.aspect_num_decoder(
                aspects_num_labels=aspects_nums,
                # encoder_outputs=dict_for_aspects_num[0],
                # 共用一个会更好
                encoder_outputs=dict_for_aspect_prompt[0],
                attention_mask=attention_mask,
                aspects_num_decoder_input_ids=aspects_num_decoder_input_ids,
                sentence_mask=sentence_mask,
                image_mask=image_mask,
                # 共用一个编码结果
                # encoder_outputs_all=dict_for_aspects_num.hidden_states,
                encoder_outputs_all=dict_for_aspect_prompt.hidden_states,
                nouns_embeds=noun_embeds,
                nouns_mask=noun_mask,
                image_caption_valid=image_caption_valid,
                image_caption_mask=image_caption_mask_aspect,
                score=senti_scores)

            predict_aspects_num = torch.argmax(predict_aspects_num_logits, dim=1)
            new_predict_aspects_num = predict_aspects_num + torch.ones_like(predict_aspects_num)
        else:
            aspects_num_loss = 0
            new_predict_aspects_num = []
            predict_aspects_num = []
            for i in range(len(input_ids)):
                new_predict_aspects_num.append(5)
                predict_aspects_num.append(4)
            new_predict_aspects_num = torch.tensor(new_predict_aspects_num)
            predict_aspects_num = torch.tensor(predict_aspects_num)

        generated_senti_prompt = None

        if self.use_generated_senti_prompt:
            senti_prompt_decoder_input_ids, senti_prompt_decoder_attention_mask = [
                aesc_infos['senti_prompt_decoder_input_ids'].to(input_ids.device),
                aesc_infos['senti_prompt_decoder_attention_mask'].to(input_ids.device)]
            generated_senti_prompt, diversity_loss, l2_reg_loss = self.senti_prompt_decoder(
                encoder_outputs=dict_for_senti_prompt.last_hidden_state,
                # 共用同一个编码器的输出结果
                # encoder_outputs=dict_for_aspect_prompt.last_hidden_state,
                attention_mask=attention_mask,
                decoder_input_ids=senti_prompt_decoder_input_ids,
                decoder_attention_mask=senti_prompt_decoder_attention_mask,
                sentence_mask=sentence_mask,
                image_mask=image_mask,
                noun_embeds=noun_embeds,
                noun_mask=noun_mask,
                image_caption_valid=image_caption_valid,
                image_caption_mask=image_caption_mask_senti,
            )

            generated_senti_prompt = generated_senti_prompt[:, 1:, :]  ##(batch_size, 2, 768)

        dict, inter_task_contrastive_loss = self.encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            generated_aspect_prompt=generated_aspect_prompt,
            generated_senti_prompt=generated_senti_prompt,
            aspects_num=new_predict_aspects_num,
            output_hidden_states=True,
            sentence_mask=sentence_mask,
            image_mask=image_mask,
            encoder_outputs=dict_for_aspect_prompt.last_hidden_state,
            image_caption_valid=image_caption_valid,
            image_caption_mask=image_caption_mask,
            encoder_outputs_all=dict_for_aspect_prompt.hidden_states,
            encoder_outputs_senti=dict_for_senti_prompt.last_hidden_state,
            return_dict=True)
        # -------------- 在第二次的encoder结果中同样使用一样的修正部分，前面的修正部分，只为了aspect部分的信息获取。
        loss_crd_second = torch.tensor(0.0, dtype=torch.float)
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
        # # TODO: 同样的，先暂时去除相关度计算部分代码，先验证Prompt的修改能提升模型性能吧、
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
        #     loss_crd_second = 0.9 * loss_crd_soft_second + 0.1 * loss_crd_hard_second
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
        # gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)  # 图像特征后续不用了，就扔掉了。
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
        # caption_mask_expanded_encoder = image_caption_mask_expanded_for_image.repeat(1, 1, image.size(2)).to(
        #     attention_mask.device)
        # dict.last_hidden_state = dict.last_hidden_state.masked_scatter(
        #     caption_mask_expanded_encoder,
        #     weight_image_caption[caption_mask_expanded_encoder]
        # )
        # --------------------
        # --------------
        senti_prompt_mask = (input_ids == self.senti_prompt_token_id)  # 指示整个输入文本的需要放置情绪Prompt的位置

        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        # print("num_token, end_index", self.num_image_tokens, end_index)
        src_embed_outputs = hidden_states[0]  # 第零层的输出。
        enhanced_features_for_encoder, contrastive_loss = self._get_emotion_prototype(encoder_outputs,
                                                                                      senti_prompt_mask,
                                                                                      aesc_infos['labels'].to(
                                                                                          input_ids.device),
                                                                                      aspects_nums)
        state = BartState(
            # encoder_outputs,
            enhanced_features_for_encoder,
            encoder_mask,
            input_ids[:,
            end_index:],  # the text features start from index 38, the front are image features.
            first,
            src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]
        tmp_spans = spans
        # print("spans", spans, spans.size())
        hidden_state, logits = self.decoder(spans, state)  ## spans: (2, 13) logits: (2, 12, 40)
        # print("Training", Training)
        if Training:
            spans = spans[:, 1:]
        # 单单把MASC任务 拆为分类任务，所以我们从源码中的各自三元组中拿出情绪标签部分
        pos_id, neu_id, neg_id = 3, 4, 5
        is_pos = (spans == 3)
        is_neu = (spans == 4)
        is_neg = (spans == 5)
        is_sentiment_token = is_pos | is_neu | is_neg
        # print("sds", is_sentiment_token.size(), hidden_state.size(), logits.size(), aspects_num)
        sentiment_vectors = hidden_state[is_sentiment_token]  # 特征部分
        sentiment_labels_original = spans[is_sentiment_token]
        sentiment_label_map = {pos_id: 0, neu_id: 1, neg_id: 2}
        sentiment_labels_mapped = torch.tensor(
            [sentiment_label_map[token_id.item()] for token_id in sentiment_labels_original],
            device=input_ids.device
        )

        sentiment_vectors = self.sentiment_head(sentiment_vectors)
        if Training:
            loss, per_token_loss = self.span_loss_fct(tmp_spans[:, 1:], logits, span_mask[:, 1:])
        else:
            loss, per_token_loss = self.span_loss_fct(tmp_spans[:, :], logits, span_mask[:, :])
        # 3. 计算自适应温度
        # print("loss", per_token_loss, per_token_loss.size())
        # with torch.no_grad():
        #
        #     # a. 创建一个mask，找出哪些样本的损失是大于0的
        #     valid_loss_mask = per_token_loss > 0
        #
        #     # b. 如果整个batch都没有有效的损失，直接返回一个默认温度
        #     if not valid_loss_mask.any():
        #         adaptive_temperature_scalar = 0.1  # 返回一个默认或中间值
        #     else:
        #         # c. 只选择那些非零的损失值
        #         effective_losses = per_token_loss[valid_loss_mask]
        #
        #         # d. 用这些有效的损失值来更新移动平均和标准差
        #         if self.training:
        #             batch_mean = effective_losses.mean()
        #             batch_std = effective_losses.std()
        #             if not torch.isnan(batch_std):
        #                 self.running_loss_mean = self.ema_decay * self.running_loss_mean + (
        #                             1 - self.ema_decay) * batch_mean
        #                 self.running_loss_std = self.ema_decay * self.running_loss_std + (
        #                             1 - self.ema_decay) * batch_std
        #
        #         # e. 计算所有样本（包括损失为0的）的自适应温度
        #         #    对于损失为0的样本，我们给它一个默认的中间温度，因为它既不难也不简单
        #         normalized_loss = (per_token_loss - self.running_loss_mean) / (self.running_loss_std + 1e-6)
        #         difficulty_score = torch.sigmoid(normalized_loss)
        #
        #         t_min, t_max = 0.05, 0.3
        #         per_sample_temperatures = t_min + (t_max - t_min) * difficulty_score
        #
        #         # f. 【可选但推荐】对于损失为0的样本，其温度可以设为默认值，因为难度未知
        #         default_temp = (t_min + t_max) / 2
        #         per_sample_temperatures.masked_fill_(~valid_loss_mask, default_temp)
        #
        #         # g. 计算最终的平均温度
        #         #    这里我们可以选择只对有损失的样本的温度求平均，这样更准确
        #         adaptive_temperatures = per_sample_temperatures[valid_loss_mask].mean().item()
        # 确保至少有两个样本才能进行对比
        if sentiment_vectors.shape[0] < 2:
            sup_con_loss = torch.tensor(0.0, device=sentiment_vectors.device)
        else:
            sup_con_loss = self.contrastive_loss(sentiment_vectors, sentiment_labels_mapped)
        if not Training:
            sup_con_loss = torch.tensor(0.0, device=attention_mask.device)
        if self.datasets == 'twitter15':
            sup_con_loss = sup_con_loss * 0.8
            inter_task_contrastive_loss = inter_task_contrastive_loss * 0.5
        else:
            sup_con_loss = sup_con_loss * 0.5
            inter_task_contrastive_loss = inter_task_contrastive_loss * 0.5
        loss_crd = loss_aspect + loss_crd_second + loss_senti + contrastive_loss + sup_con_loss + inter_task_contrastive_loss
        print("两次相关度计算的损失 以及 可能存在的情绪encoder结果的相关度", loss_aspect.item(), sup_con_loss.item()
              , loss_senti.item(), inter_task_contrastive_loss.item())
        loss = [diversity_loss, l2_reg_loss, sparsity_loss_layers_aspect, sparsity_loss_image_aspect,
                sparsity_loss_layers_aspect_num, sparsity_loss_image_aspect_num, loss_crd]
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
            sentiment_counts=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        ### for prompt
        # import ipdb; ipdb.set_trace()

        ## for aspect-spans

        aspects_num = torch.tensor(aspects_num).to(input_ids.device)
        state, aspects_num_loss, predict_aspects_num, pseudo_loss, loss, hidden_state, logits, _ = self.prepare_state(
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

        # hidden_state, logits = self.decoder(spans, state)  ## spans: (2, 13) logits: (2, 12, 40)
        if training:
            span_loss, per_token_loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])
        else:
            span_loss, per_token_loss = self.span_loss_fct(spans[:, :], logits, span_mask[:, :])

        # span_loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

        all_loss = span_loss + self.loss_lambda * aspects_num_loss
        print("各个loss的值", span_loss.item(), aspects_num_loss.item())
        for i in range(len(loss)):
            print("继续往后的loss", loss[i].item())
            all_loss = all_loss + loss[i]
        return all_loss, predict_aspects_num, pseudo_loss

    def _get_emotion_prototype(self, encoder_outputs, sentence_mask, spans, aspect_num):
        batch_size = encoder_outputs.size(0)
        # 提取所有样本中，在 senti_prompt_mask 位置的 encoder 特征
        # 同时，提取这些位置对应的真实情感标签
        all_initial_senti_features_list = []
        all_true_senti_ids_list = []
        # 记录每个提取的特征在原始 (batch_idx, seq_pos_idx) 中的位置，方便写回
        write_back_indices_batch = []
        write_back_indices_seq = []

        for b_idx in range(batch_size):
            sample_senti_prompt_locs = sentence_mask[b_idx].nonzero(as_tuple=True)[0]  # 当前样本情绪prompt的位置
            sample_aesc_spans = spans[b_idx, 1:]  # 当前样本的真实 (s,e,senti)
            aspect_num_batch = aspect_num[b_idx]
            # print(sample_aesc_spans, aspect_num_batch, len(sample_senti_prompt_locs))
            # 确保情绪prompt槽位数量与aspect数量一致
            # if len(sample_senti_prompt_locs) == aspect_num_batch:
            #     for aspect_order_idx, loc_idx in enumerate(sample_senti_prompt_locs):
            #         all_initial_senti_features_list.append(encoder_outputs[b_idx, loc_idx, :])
            #         all_true_senti_ids_list.append(sample_aesc_spans[aspect_order_idx * 3 + 4])  # 真实情感ID (3,4,5)
            #         write_back_indices_batch.append(b_idx)
            #         write_back_indices_seq.append(loc_idx.item())  # 转换为 Python int
            for aspect_order_idx, loc_idx in enumerate(sample_senti_prompt_locs):
                if aspect_order_idx == aspect_num_batch:
                    break
                all_initial_senti_features_list.append(encoder_outputs[b_idx, loc_idx, :])
                all_true_senti_ids_list.append(sample_aesc_spans[aspect_order_idx * 3 + 4])  # 真实情感ID (3,4,5)
                write_back_indices_batch.append(b_idx)
                write_back_indices_seq.append(loc_idx.item())  # 转换为 Python int
        # print("message", all_initial_senti_features_list, all_true_senti_ids_list)
        # print("信息", write_back_indices_batch, write_back_indices_seq)

        if not all_initial_senti_features_list:
            # 如果整个批次都没有有效的 Aspect 可以进行原型学习
            # 直接返回原始的 encoder_outputs 和一个零损失
            # print("Warning: No valid aspect-sentiment features found in this batch for prototype learning.")
            return encoder_outputs.clone(), torch.tensor(0.0, device=encoder_outputs.device)  # 返回原始输出和零损失

        initial_senti_features_flat = torch.stack(all_initial_senti_features_list)  # [N_total_prompts, H]
        true_senti_ids_flat = torch.tensor(all_true_senti_ids_list, device=encoder_outputs.device,
                                           dtype=torch.long)  # [N_total_prompts]

        # --- 步骤 3: 调用原型学习模块 ---
        contrastive_loss, features_from_prototype_for_loss = self.prototype_forward(
            initial_senti_prompt_features=initial_senti_features_flat,
            true_senti_ids_for_prompts=true_senti_ids_flat
        )
        # features_from_prototype_for_loss: [N_total_prompts, prototype_dim]

        # --- 步骤 4: 将原型增强的特征融合回 encoder_outputs ---
        final_encoder_outputs = encoder_outputs.clone()

        # projected_back_features: [N_total_prompts, encoder_hidden_size]

        # 融合策略：例如，原始特征 + 投影回来的原型空间特征
        # initial_senti_features_flat 是原始的 encoder 输出
        # projected_back_features 是经过原型模块学习和投影的特征
        # 可以通过一个门控融合或直接相加
        # 假设有一个门控
        gate_proto_fusion = torch.sigmoid(self.proto_fusion_gate_linear(
            torch.cat([initial_senti_features_flat, features_from_prototype_for_loss], dim=-1)
        ))
        fused_senti_features = gate_proto_fusion * initial_senti_features_flat + \
                               (1 - gate_proto_fusion) * features_from_prototype_for_loss
        # 简化：直接相加 (残差连接)
        # fused_senti_features = initial_senti_features_flat + projected_back_features

        # 将融合后的特征写回到 final_encoder_outputs 的正确位置
        # 使用高级索引
        final_encoder_outputs[write_back_indices_batch, write_back_indices_seq, :] = fused_senti_features

        return final_encoder_outputs, contrastive_loss

    def prototype_forward(self, initial_senti_prompt_features, true_senti_ids_for_prompts):
        """
        对初始情绪Prompt特征进行原型学习和显式增强。

        :param initial_senti_prompt_features: Tensor [N, H_encoder], 初始情绪Prompt特征。
        :param true_senti_ids_for_prompts: Tensor [N], 对应的真实情感ID (3,4,5)。
        :return: tuple (contrastive_loss, enhanced_features_for_encoder)
        """
        if initial_senti_prompt_features.size(0) == 0:
            return torch.tensor(0.0, device=initial_senti_prompt_features.device, requires_grad=True), \
                   initial_senti_prompt_features.clone()

        # 1. 将初始特征投影到原型空间
        projected_features_initial = self.projection_head(initial_senti_prompt_features)  # [N, prototype_dim]

        # --- 步骤 2: 显式的原型空间特征增强 ---
        enhanced_projected_features_list = []
        prototype_labels_for_loss_list = []  # 用于对比损失的标签
        valid_indices_for_enhancement_and_loss = []  # 记录哪些特征参与了增强和损失

        for i in range(projected_features_initial.size(0)):
            current_projected_feat = projected_features_initial[i]  # [prototype_dim]
            true_sentiment_id = true_senti_ids_for_prompts[i].item()

            if true_sentiment_id in self.sentiment_id_to_idx_map:
                prototype_label_idx = self.sentiment_id_to_idx_map[true_sentiment_id]  # 0, 1, or 2

                # 获取对应的真实类别原型
                target_prototype = self.prototypes[prototype_label_idx]  # [prototype_dim]

                # **进行特征增强**
                # 方式 A: 拼接后通过 MLP
                # combined_for_enhancement = torch.cat([current_projected_feat, target_prototype],
                #                                      dim=-1)  # [prototype_dim * 2]
                # enhanced_feature_in_proto_space = self.prototype_enhancement_mlp(
                #     combined_for_enhancement)  # [prototype_dim]

                # 方式 B: 注意力 (更复杂，如果原型只有一个，则简化为与原型交互)
                query = current_projected_feat.unsqueeze(0).unsqueeze(0)  # (1, 1, prototype_dim)
                key_value = target_prototype.unsqueeze(0).unsqueeze(0)  # (1, 1, prototype_dim)
                enhanced_feature_in_proto_space = self.prototype_enhancement_attention(query, key_value, key_value)[
                    0].squeeze(0).squeeze(0)

                # 方式 C: 简单的加权平均或残差连接
                # enhanced_feature_in_proto_space = current_projected_feat + 0.5 * target_prototype # 示例
                # enhanced_feature_in_proto_space = (current_projected_feat + target_prototype) / 2

                enhanced_projected_features_list.append(enhanced_feature_in_proto_space)
                prototype_labels_for_loss_list.append(prototype_label_idx)
                valid_indices_for_enhancement_and_loss.append(i)
            else:
                # 对于不在映射中的情感ID，我们仍然保留其初始投影特征，但不参与对比损失和显式增强
                # 或者，你可以选择用一个“默认”原型或零向量进行“增强”
                enhanced_projected_features_list.append(current_projected_feat)  # 保留原始投影特征
                # 注意：这些不参与对比损失的特征，在stack后需要被正确处理

        if not valid_indices_for_enhancement_and_loss:  # 如果没有一个特征参与了增强和损失
            return torch.tensor(0.0, device=initial_senti_prompt_features.device, requires_grad=True), \
                   self.prototype_to_encoder_projection(projected_features_initial)  # 返回投影后的初始特征

        # 将所有（可能部分增强，部分原始投影）的特征堆叠起来
        all_processed_projected_features = torch.stack(enhanced_projected_features_list)  # [N, prototype_dim]

        # --- 步骤 3: 计算对比损失 (只针对那些有有效真实标签的特征) ---
        # 从 all_processed_projected_features 中提取参与对比损失的部分
        features_for_loss_stacked = all_processed_projected_features[valid_indices_for_enhancement_and_loss]
        labels_for_loss_stacked = torch.tensor(prototype_labels_for_loss_list,
                                               device=features_for_loss_stacked.device,
                                               dtype=torch.long)

        projected_features_norm_for_loss = F.normalize(features_for_loss_stacked, p=2, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=-1)

        logits_for_contrastive_loss = torch.matmul(projected_features_norm_for_loss, normalized_prototypes.t())
        logits_for_contrastive_loss = logits_for_contrastive_loss / self.temperature_prototype

        contrastive_loss = F.cross_entropy(logits_for_contrastive_loss, labels_for_loss_stacked, reduction='mean')

        # --- 步骤 4: 将所有处理过的投影特征（包括增强的和未增强的）投影回 encoder 空间 ---
        enhanced_features_for_encoder = self.prototype_to_encoder_projection(all_processed_projected_features)
        # enhanced_features_for_encoder: [N, H_encoder]

        return contrastive_loss, enhanced_features_for_encoder

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


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        # self.temperature = temperature
        # self.base_temperature = temperature

        # --- 调试入口 ---
        # print("\n--- Inside SupConLoss ---")
        if torch.isnan(features).any():
            print("!!! ERROR: Input features contain NaN.")
            return torch.tensor(0.0, device=device)  # 提前退出

        # 1. 归一化
        features = F.normalize(features, p=2, dim=1)
        if torch.isnan(features).any():
            print("!!! ERROR: NaN appeared after normalization.")
            return torch.tensor(0.0, device=device)
        # print(f"Step 1: Normalized features shape: {features.shape}")

        # 2. 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        if torch.isnan(similarity_matrix).any():
            print("!!! ERROR: NaN appeared in similarity_matrix.")
            return torch.tensor(0.0, device=device)
        # print(
        #     f"Step 2: Similarity matrix shape: {similarity_matrix.shape}, max: {similarity_matrix.max():.4f}, min: {similarity_matrix.min():.4f}")

        # 3. 创建正样本对的mask
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)

        # 4. 排除对角线
        mask_self = torch.eye(features.shape[0], dtype=torch.bool, device=device)
        mask_positive.masked_fill_(mask_self, 0)

        # 5. 准备logits (屏蔽对角线)
        logits = similarity_matrix.clone()
        logits.masked_fill_(mask_self, float('-inf'))
        if torch.isinf(logits).all():
            print("!!! WARNING: All logits are -inf. This might happen with batch size 1.")

        # 6. 稳定的log_softmax
        log_prob = F.log_softmax(logits, dim=1)
        log_prob = log_prob.masked_fill(torch.isinf(log_prob), 0)
        if torch.isnan(log_prob).any():
            print("!!! ERROR: NaN appeared after log_softmax.")
            # 让我们看看是什么导致了log_softmax出问题
            print("Logits before log_softmax:", logits)
            return torch.tensor(0.0, device=device)
        # print(f"Step 6: log_prob shape: {log_prob.shape}, max: {log_prob.max():.4f}, min: {log_prob.min():.4f}")

        # 7. 计算损失
        num_positives_per_sample = mask_positive.sum(dim=1)
        sum_log_prob_pos = (mask_positive * log_prob).sum(dim=1)

        # 安全地计算平均值，防止 0/0
        denominator = torch.clamp(num_positives_per_sample, min=1e-9)
        mean_log_prob_pos = sum_log_prob_pos / denominator
        if torch.isnan(mean_log_prob_pos).any():
            print("!!! ERROR: NaN appeared when calculating mean_log_prob_pos.")
            print("Numerator (sum_log_prob_pos):", sum_log_prob_pos)
            print("Denominator (num_positives_per_sample):", num_positives_per_sample)
            return torch.tensor(0.0, device=device)

        # 8. 最终损失计算
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        has_positives_mask = (num_positives_per_sample > 0)

        if not has_positives_mask.any():
            print("--- INFO: No positive pairs found in this batch. Returning 0 loss. ---")
            return torch.tensor(0.0, device=device)

        loss = loss[has_positives_mask].mean()

        # if torch.isnan(loss):
        #     print("!!! ERROR: Final loss is NaN.")
        #     print("Loss tensor before mean():", loss[has_positives_mask])
        # else:
        #     print(f"Step 8: Final loss computed: {loss.item():.4f}")

        # print("--- Exiting SupConLoss ---\n")
        return loss


