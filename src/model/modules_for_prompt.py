import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers.models.bart.modeling_bart import *
from src.model.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)
from src.model.modeling_bart import (PretrainedBartModel, BartDecoder,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from src.model.config import MultiModalBartConfig

from transformers import AutoConfig, AutoModel, CLIPVisionConfig, CLIPVisionModel
# CLIPVisionModel, CLIPVisionConfig
import timm
from src.model.attention import Attention_for_Senti_Prompt
from safetensors import safe_open
import ipdb
import os

os.environ['TORCH_HOME'] = r'd:\Desktop\研一内容\论文对应代码\GMP-main\pretrained'  # 强制指定下载路径

TIMM_MODELS = {
    'nf_resnet50': 2048,
    'vit_base_patch32_224': 768
    # 与Local Feature Alignment Prompt-Tuning for  Few-shot Multimodal Aspect Sentiment Analysis 比较
}

CLIP_MODELS_DIMS = {
    'openai/clip-vit-base-patch32': 768,
    # 可以根据需要添加更多 CLIP ViT 模型和它们的维度
}


def is_clip_model(model_name):
    return model_name.startswith('openai/clip-')


image_encoder_type = 'timm'
# image_model_name = 'nf_resnet50'

image_model_name = 'vit_base_patch32_224'


def get_image_encoder(image_model_name):
    if image_model_name in TIMM_MODELS.keys() and image_model_name != 'vit_base_patch32_224':  # nf_resnet50 加载逻辑 (保持不变，但排除 vit_base_patch32_224)
        image_encoder = timm.create_model("nf_resnet50", pretrained=False, num_classes=0)
        checkpoint_path = r"data/nf_Resnet50/model.safetensors"  # nf_resnet50 的 checkpoint path 保持不变
        assert os.path.exists(checkpoint_path), f"文件不存在: {checkpoint_path}"
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.fc")}
        image_encoder.load_state_dict(filtered_state_dict, strict=False)
        return image_encoder

    elif image_model_name == 'vit_base_patch32_224':  # 本地加载 vit_base_patch32_224 的逻辑
        local_checkpoint_path = r"data/vit_base_patch32_224/model.safetensors"  # <---  **修改这里：替换为你的本地权重文件路径**
        assert os.path.exists(
            local_checkpoint_path), f"本地 vit_base_patch32_224 权重文件不存在: {local_checkpoint_path}"
        image_encoder = timm.create_model(image_model_name, pretrained=False, num_classes=0
                                          )  # 使用 checkpoint_path 从本地加载
        with safe_open(local_checkpoint_path, framework="pt", device="cpu") as f:  # 使用 safe_open 加载 state_dict
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        #  不需要像 nf_resnet50 那样过滤 'head.fc'，因为 vit_base_patch32_224 通常没有 head.fc 这样的分类头
        image_encoder.load_state_dict(state_dict, strict=False)  # 加载 state_dict 到模型
        return image_encoder

    elif is_clip_model(image_model_name):  # CLIP ViT 加载逻辑 (保持不变)
        config = CLIPVisionConfig.from_pretrained(image_model_name)
        image_encoder = CLIPVisionModel.from_pretrained(
            image_model_name,
            config=config,
        )
        return image_encoder
    else:  # 默认使用 AutoModel 加载其他 HuggingFace 模型 (如果需要) (保持不变)
        return AutoModel.from_pretrained(image_model_name)


def init_image_encoder(image_model_name, frozen_image_encoder, num_image_tokens, d_text_encoder):
    image_encoder = get_image_encoder(image_model_name)
    d_image_encoder = _d_image_encoder(image_model_name, image_encoder)

    if frozen_image_encoder:
        for p in image_encoder.parameters():
            p.requires_grad = False
            image_encoder.eval()
    # 对应于论文的W矩阵，[Dv(图像编码的维度), li(超参数 设置为2)* Dt(文本嵌入BART的维度大小)]
    proj_image_features = nn.Linear(
        in_features=d_image_encoder,
        out_features=num_image_tokens * d_text_encoder,
    )
    # 这里也可以换成一个MLP的
    proj_image_features = nn.Sequential(
        nn.Linear(d_image_encoder, 2048),
        nn.ReLU(),
        nn.Linear(2048, num_image_tokens * d_text_encoder),
    )
    return proj_image_features.cuda(), d_image_encoder


def _d_image_encoder(image_encoder_type, image_model_name, image_encoder):
    ##image_model_name默认为： 'microsoft/resnet-50' 所以返回的是这个图像处理模型的维度大小
    # model_name = image_model_name
    # if model_name in TIMM_MODELS.keys():
    #     return TIMM_MODELS[model_name]
    # elif is_clip_model(model_name):
    #     return image_encoder.config.hidden_size
    # elif model_name.startswith('microsoft/resnet-'):
    #     return image_encoder.config.hidden_sizes[-1]
    # else:
    #     return image_encoder.config.hidden_size

    model_type = image_encoder_type.lower()  # 确保大小写不敏感比较
    model_name = image_model_name.lower()

    if model_type == 'timm' and model_name in TIMM_MODELS.keys():
        return TIMM_MODELS[model_name]
    elif model_type == 'clip' and is_clip_model(model_name):
        return CLIP_MODELS_DIMS.get(image_model_name, image_encoder.config.hidden_size)  # 使用字典获取已知的 CLIP 维度
    elif model_type == 'hf_resnet' and model_name.startswith('microsoft/resnet-'):
        return image_encoder.config.hidden_sizes[-1]
    elif model_type == 'hf_auto':
        return image_encoder.config.hidden_size  # 通用的 AutoModel
    else:
        raise ValueError(f"不支持的图像编码器类型或模型名称: type={image_encoder_type}, name={image_model_name}")


def encode_images(image_encoder, proj_image_features, frozen_image_encoder, pixel_values, d_image_encoder,
                  image_encoder_type):
    # image_encoder 采用冻结的 nf_resnet50模型 pixel_values = (batch_size * 3, 224, 224)
    image_encoder = image_encoder.cuda()
    pixel_values = pixel_values.cuda()
    # print('the shape of pixel_values is {}'.format(pixel_values.shape))
    batch_size = pixel_values.shape[0]
    # if frozen_image_encoder:
    #     with torch.no_grad():
    #         image_encoder.eval()
    #         visual = image_encoder(pixel_values) # → (batch_size, 2048)
    # else:
    #     visual = image_encoder(pixel_values)
    #
    # if not isinstance(visual, torch.Tensor):  # HuggingFace model
    #     visual = visual.pooler_output
    if frozen_image_encoder:
        with torch.no_grad():
            image_encoder.eval()
            if image_encoder_type.lower() == 'clip':
                visual = image_encoder.vision_model(pixel_values).pooler_output  # CLIP 模型使用 .vision_model
            else:
                visual = image_encoder(pixel_values)
    else:
        if image_encoder_type.lower() == 'clip':
            visual = image_encoder.vision_model(pixel_values).pooler_output  # CLIP 模型使用 .vision_model
        else:
            visual = image_encoder(pixel_values)

    if not isinstance(visual, torch.Tensor):  # HuggingFace 模型
        visual = visual.pooler_output

    #
    visual = visual.reshape(batch_size, d_image_encoder)
    # 维度转换。
    visual = proj_image_features(visual).cuda()
    return visual


class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim, image_encoder_type, image_model_name, frozen_image_encoder=True,
                 num_image_tokens=2):
        super(ImageEmbedding, self).__init__()
        # 如果图像编码器都冻结呢，效果会提升吗？
        # image_model_name 默认为 'microsoft/resnet-50'
        # self.frozen_image_encoder = frozen_image_encoder
        # self.final_dim = final_dim
        # self.linear = nn.Linear(final_dim, final_dim)
        # self.d_image_encoder = _d_image_encoder(image_model_name, image_encoder)
        self.frozen_image_encoder = frozen_image_encoder
        self.final_dim = final_dim
        self.linear = nn.Linear(final_dim, final_dim)

        # 存储图像编码器类型和名称
        self.image_encoder_type = image_encoder_type
        self.image_model_name = image_model_name

        # 初始化图像编码器并获取其维度
        self.image_encoder = get_image_encoder(image_model_name)  # 根据名称加载编码器
        self.d_image_encoder = _d_image_encoder(image_encoder_type, image_model_name,
                                                self.image_encoder)  # 传递类型给 _d_image_encoder

        if frozen_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        # 转换部分 把图像嵌入换成文本嵌入的部分
        self.proj_image_features = nn.Linear(
            in_features=self.d_image_encoder,
            out_features=num_image_tokens * final_dim,
        )
        # 这里也可以换成一个MLP的 替换原有的简单投影网络
        self.proj_image_features_MLP = nn.Sequential(
            nn.Linear(self.d_image_encoder, final_dim * 2),
            nn.LayerNorm(final_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.Linear(final_dim * 4, final_dim * 2),
            # nn.LayerNorm(final_dim * 2),
            # nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(final_dim * 2, num_image_tokens * final_dim)
        )

    def forward(self, image_pixel_values):
        # import ipdb; ipdb.set_trace()
        image_pixel_values = torch.stack(image_pixel_values)
        batch_size = image_pixel_values.size(0)
        # image_encoder 采用nf_resnet50 保持原论文方法不变，
        # 原始图像像素 (batch_size, 3, 224, 224) →  (batch_size * 3, 224, 224)
        # → (batch_size, num_image_tokens*final_dim)
        image_features = encode_images(image_encoder=self.image_encoder,
                                       proj_image_features=self.proj_image_features_MLP,  # 这里的转换部分也可以有修改
                                       frozen_image_encoder=self.frozen_image_encoder,
                                       pixel_values=image_pixel_values,
                                       d_image_encoder=self.d_image_encoder,
                                       image_encoder_type=self.image_encoder_type)
        # 利用Resnet50对图像进行编码 得到一个表示
        ###image_features: (batch_size, num_image_tokens*1024) (4, 2048)
        # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
        image_features = image_features.reshape(batch_size, -1,
                                                self.final_dim)  ### (4, num_image_tokens, 1024(d_model))

        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index:index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output
        # 至此的输出是凸显嵌入表示换成的可以放进文本BART的表示形式。 结果为一个列表
        # (4, num_image_tokens, 1024(d_model))-> [num_image_tokens, 1024, num_image_tokens, 1024] 完成论文公式1


class SentimentAwareMultiModalEmbedding(nn.Module):
    def __init__(self, config: MultiModalBartConfig,
                 bart_embed_tokens: nn.Embedding,  # BART原始词嵌入层
                 img_feat_id: int,
                 cls_token_id: int,
                 pad_token_id: int,
                 num_image_tokens: int,
                 text_processor_num_sentiment_categories: int,  # 从YourTextProcessor获取
                 hidden_size: int):  # BART的hidden_size
        super().__init__()
        self.bart_embed_tokens = bart_embed_tokens
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        self.embed_dim = bart_embed_tokens.embedding_dim  # 应该等于 hidden_size

        # 图像嵌入模块 (与MultiModalBartEncoder中的一致)
        self.embed_images = ImageEmbedding(self.embed_dim, self.embed_dim, image_encoder_type, image_model_name,
                                           num_image_tokens=num_image_tokens)
        # 定义情感类别和映射 (与之前 BartInputEmbedderWithSenticNet 中的类似)
        self.sentiment_categories_map = {
            "strong_negative": 2,
            "negative": 3,
            "neutral": 4,
            "positive": 5,
            "strong_positive": 6
        }
        # SenticNet情感嵌入层
        self.sentiment_category_embedder = nn.Embedding(
            text_processor_num_sentiment_categories,
            self.embed_dim,
            padding_idx=pad_token_id  # 与词嵌入维度相同，方便后续操作
        )
        nn.init.normal_(self.sentiment_category_embedder.weight, mean=0, std=self.embed_dim ** -0.5)

        # 融合层 (拼接后线性变换)
        self.sentiment_fusion_mlp = nn.Linear(self.embed_dim * 2, self.embed_dim)
        # self.fusion_activation = nn.ReLU() # 或者其他激活函数

    def forward(self, input_ids: torch.LongTensor,
                image_features: list,  # list of tensors
                precomputed_sentiment_category_ids: torch.LongTensor,
                sentence_mask: torch.BoolTensor):
        """
        input_ids: (batch_size, seq_len)
        image_features: list of image feature tensors for the batch
        recomputed_sentiment_category_ids: (batch_size, seq_len)
        sentence_mask: (batch_size, seq_len) - True/1表示是用户评论文本部分
        """
        device = input_ids.device
        # 1. 获取原始BART词嵌入
        textual_embeds = self.bart_embed_tokens(input_ids).to(device)  # (B, S, D)
        precomputed_sentiment_category_ids = precomputed_sentiment_category_ids.to(device)
        # 2. 获取情感状态嵌入
        # print("11", precomputed_sentiment_category_ids.device)
        sentiment_state_embeds = self.sentiment_category_embedder(precomputed_sentiment_category_ids).to(
            device)  # (B, S, D)

        sentence_mask_float = sentence_mask.unsqueeze(-1).float().to(device)  # (B, S, 1)
        text_parts_original_embeds = textual_embeds * sentence_mask_float

        # 3. 融合文本词嵌入和情感状态嵌入
        combined_text_sentiment_embeds = torch.cat([text_parts_original_embeds, sentiment_state_embeds], dim=-1).to(
            device)  # (B, S, D*2)
        fused_sentence_parts_embeds = torch.tanh(self.sentiment_fusion_mlp(combined_text_sentiment_embeds)).to(
            device)  # (B, S, D)
        # enhanced_textual_embeds = self.fusion_activation(enhanced_textual_embeds)
        enhanced_textual_embeds = fused_sentence_parts_embeds * sentence_mask_float + textual_embeds * (
                    1 - sentence_mask_float)

        # 4. 处理图像嵌入并替换 (与你原_embed_multi_modal逻辑类似)
        #    注意：这里我们用 enhanced_textual_embeds 作为基础
        final_combined_embeds = enhanced_textual_embeds.clone()

        # 定位图像特征在input_ids中的位置
        image_placeholder_mask = (input_ids == self.img_feat_id) | (input_ids == self.cls_token_id)  # 假设CLS也可能被图像替换

        # 获取图像嵌入 (注意：embed_images的输出是list of tensors)
        projected_image_embeds_list = self.embed_images(image_features)

        # 将图像嵌入放到对应位置
        for i in range(input_ids.shape[0]):  # 遍历batch
            current_image_placeholder_mask = image_placeholder_mask[i]  # (S,)
            projected_img_embeds = projected_image_embeds_list[i]  # (num_img_tokens_for_sample, D)

            if projected_img_embeds is not None and len(projected_img_embeds) > 0:
                # 确保替换的长度匹配
                num_placeholders = current_image_placeholder_mask.sum().item()
                if num_placeholders == len(projected_img_embeds):
                    final_combined_embeds[i, current_image_placeholder_mask] = projected_img_embeds
                elif num_placeholders > 0:  # 如果长度不匹配，这是一个潜在问题
                    # print(f"Warning: Mismatch between image placeholders ({num_placeholders}) "
                    #       f"and projected image embeddings ({len(projected_img_embeds)}) for sample {i}. "
                    #       f"Using only the first {min(num_placeholders, len(projected_img_embeds))} embeddings.")
                    # 你需要决定如何处理长度不匹配的情况，例如截断或填充
                    len_to_replace = min(num_placeholders, len(projected_img_embeds))
                    placeholder_indices = torch.where(current_image_placeholder_mask)[0]
                    if len_to_replace > 0:
                        final_combined_embeds[i, placeholder_indices[:len_to_replace]] = projected_img_embeds[
                                                                                         :len_to_replace]

        return final_combined_embeds


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, original_encoder_layer, with_cross_attention=True):
        super().__init__()
        self.with_cross_attention = with_cross_attention  # *** 2. 保存开关状态 ***
        # 直接复用原始层中的模块，保证权重一致
        self.embed_dim = original_encoder_layer.embed_dim
        self.self_attn = original_encoder_layer.self_attn
        self.self_attn_layer_norm = original_encoder_layer.self_attn_layer_norm
        self.fc1 = original_encoder_layer.fc1
        self.fc2 = original_encoder_layer.fc2
        self.final_layer_norm = original_encoder_layer.final_layer_norm
        self.activation_fn = original_encoder_layer.activation_fn
        self.dropout = original_encoder_layer.dropout

        # *** 新增的交叉注意力模块 ***
        # 我们可以用一个和self_attn配置完全一样的MultiheadAttention层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.self_attn.dropout,
            batch_first=True  # 假设我们都用 batch_first
        )
        # 对应的LayerNorm
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.gate = nn.Linear(768 * 2, 1)

    def forward(
            self,
            hidden_states,  # (seq_len, batch_size, embed_dim) -> BART默认格式
            attention_mask=None,
            cross_modal_features=None,  # (batch_size, cross_seq_len, embed_dim)
            output_attentions=False,
    ):
        # 我们的代码都假设 batch_first=False，按你的源码来
        # hidden_states: T x B x C
        # cross_modal_features: M x B x C (M是多模态序列长度)
        # attention_mask: B x T
        # cross_attention_mask: B x M

        # 1. 自注意力 (Self-Attention)
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            key_padding_mask=attention_mask,  # 注意 MultiheadAttention 的 mask 参数名
            output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 2. 交叉注意力 (Cross-Attention)
        if cross_modal_features is not None and self.with_cross_attention:
            residual = hidden_states
            # 将多模态特征转置以匹配格式：M x B x C
            cross_modal_features_t = cross_modal_features.transpose(0, 1)

            # hidden_states (文本) 作为 Query
            # cross_modal_features (图像/字幕) 作为 Key 和 Value
            cross_attn_output, cross_attn_weights = self.cross_attn(
                query=hidden_states,
                key=cross_modal_features_t,
                value=cross_modal_features_t,

            )
            hidden_states = F.dropout(cross_attn_output, p=self.dropout, training=self.training)

            gate_score = torch.sigmoid(
                self.gate(torch.cat([hidden_states, residual], dim=-1)))
            hidden_states = gate_score * residual + (1 - gate_score) * hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        # 3. 前馈网络 (Feed-Forward Network)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回 hidden_states 和两种注意力权重
        return hidden_states, (self_attn_weights, cross_attn_weights if cross_modal_features is not None and self.with_cross_attention else None)


class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, config: MultiModalBartConfig, encoder, img_feat_id,
                 cls_token_id, num_image_tokens, pad_token_id, freeze_bart):
        super().__init__()

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name,
                                           num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions
        # 源码的方式 早期融合
        # self.layers = encoder.layers
        # 修改变成混合融合模型 让encoder的每一层去与字幕信息融合
        self.layers = nn.ModuleList()
        self.fusion_start_layer = 4
        for i in range(len(encoder.layers)):
            # 判断当前层是否需要交叉注意力
            do_cross_attention = (i >= self.fusion_start_layer)

            print(f"Layer {i}: with_cross_attention = {do_cross_attention}")

            # 创建层，并传入开关状态
            layer = CrossAttentionEncoderLayer(
                original_encoder_layer=encoder.layers[i],
                with_cross_attention=do_cross_attention
            )
            self.layers.append(layer)
        if freeze_bart:
            for p in self.layers.parameters():
                p.requires_grad = False
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm
        # 相关度计算部分
        self.text_caption_cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4,
                                                                  batch_first=True, dropout=0.2)
        # self.text_caption_attn_output_projection = nn.Linear(self.config.hidden_size, 1)  # 示例：将池化后的输出投影到 1 维
        self.text_caption_attn_output_projection = nn.Sequential(
            nn.Linear(768 + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.text_caption_attn_output_projection_cross = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 768),
        )
        self.threshold = nn.Parameter(torch.tensor(0.7))
        # self.threshold = torch.tensor(0.7)
        self.temperature = nn.Parameter(torch.tensor(3.0))


    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        mask = (input_ids == self.img_feat_id) | (
                input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value

        return embedded

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                caption_mask=None,
                sentence_mask=None,
                image_mask=None,
                image_valid=None,
                score=None,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 字幕信息和字幕mask
        original_caption_mask = caption_mask
        caption_mask = caption_mask.unsqueeze(-1).to(attention_mask.device)
        # print("input_ids", x.size(), caption_mask.size())
        caption = x * caption_mask.float().to(attention_mask.device)
        image_mask = image_mask.unsqueeze(-1).to(attention_mask.device)
        image = x * image_mask.float().to(attention_mask.device)
        # 相关度计算部分
        original_sentence_mask = sentence_mask
        sentence_mask = sentence_mask.unsqueeze(-1).to(attention_mask.device)
        text = x * sentence_mask.float().to(attention_mask.device)
        loss_crd = torch.tensor(0.0, dtype=torch.float)
        batch_size = x.size(0)
        valid_indices = torch.where(image_valid)[0]

        num_valid = len(valid_indices)
        crd_logits_all = torch.zeros(batch_size, 1, device=attention_mask.device)  # 默认logit为0，对应sigmoid(0)=0.5

        # relevance_scores_all = torch.full((batch_size, 1), 0.5, device=attention_mask.device)  # 默认得分0.5 (或根据你的策略设为0.0或1.0)
        relevance_scores_all = score.unsqueeze(-1).to(attention_mask.device)  # 对于没有字幕就用预训练模型给出的参数
        # relevance_scores_all = score.unsqueeze(-1).unsqueeze(-1).to(attention_mask.device)
        # relevance_scores_all = relevance_scores_all.repeat(1, caption.size(1), caption.size(2)).to(attention_mask.device)
        if num_valid > 0:
            # 2. 提取有效样本的数据
            valid_text_emb = text[valid_indices].to(attention_mask.device)  # [num_valid, seq_len_text, hidden_size]
            valid_caption_emb = caption[valid_indices].to(
                attention_mask.device)  # [num_valid, max_len, hidden_size]
            valid_text_mask = original_sentence_mask[valid_indices].to(attention_mask.device)  # [num_valid, max_len]
            # **关键: MultiheadAttention 的 key_padding_mask 需要 True 表示 Padding 位置**
            # 假设你的 mask 是 1 表示有效, 0 表示 padding, 需要转换
            valid_caption_padding_mask = (
                    original_caption_mask[valid_indices] == 0).to(
                attention_mask.device)  # [num_valid, max_len], True for padding

            # 3. 批处理交叉注意力 (假设 batch_first=True)
            # query: text, key/value: caption
            # attn_output_valid, attn_weights = self.text_caption_cross_attention(
            #     query=valid_text_emb,
            #     key=valid_caption_emb,
            #     value=valid_caption_emb,
            #     key_padding_mask=valid_caption_padding_mask  # ****** 提供正确的 Mask ******
            # )  # Output: [num_valid, max_len, hidden_size]
            # 只使用交叉注意力的部分
            attn_output_valid, attn_weights = self.text_caption_cross_attention(
                query=valid_caption_emb,
                key=valid_text_emb,
                value=valid_text_emb,
                key_padding_mask=valid_text_mask
                # ****** 提供正确的 Mask ******
            )
            # logits_valid = self.text_caption_attn_output_projection_cross(attn_output_valid)
            # 提取最高的几个注意力权重，表示最相关的部分
            top_k_weights, _ = torch.topk(attn_weights, k=min(5, attn_weights.size(-1)), dim=-1)
            attention_focus = top_k_weights.mean(dim=-1).mean(dim=-1).unsqueeze(1)  # [num_valid, 1]

            # 4. 批处理屏蔽平均池化
            # 将 padding 位置的 attention output 置零 text 部分的
            # attn_output_valid_masked = attn_output_valid * valid_text_mask.unsqueeze(-1).float().to(
            #     attention_mask.device)
            # # 计算每个样本的有效长度
            # text_lengths_valid = valid_text_mask.sum(dim=1, keepdim=True).float().to(
            #     attention_mask.device)  # [num_valid, 1]
            # # 对有效位置求和
            # sum_attn_output_valid = attn_output_valid_masked.sum(dim=1).to(
            #     attention_mask.device)  # [num_valid, hidden_size]
            # # 计算平均值，防止除零
            # mean_attn_output_valid = sum_attn_output_valid / torch.clamp(text_lengths_valid,
            #                                                              min=1e-9)  # [num_valid, hidden_size]

            # 4. 批处理屏蔽平均池化
            # 将 padding 位置的 attention output 置零
            attn_output_valid_masked = attn_output_valid * valid_caption_padding_mask.unsqueeze(-1).float().to(
                attention_mask.device)
            # 计算每个样本的有效长度
            text_lengths_valid = valid_caption_padding_mask.sum(dim=1, keepdim=True).float().to(
                attention_mask.device)  # [num_valid, 1]
            # 对有效位置求和
            sum_attn_output_valid = attn_output_valid_masked.sum(dim=1).to(
                attention_mask.device)  # [num_valid, hidden_size]
            # 计算平均值，防止除零
            mean_attn_output_valid = sum_attn_output_valid / torch.clamp(text_lengths_valid,
                                                                         min=1e-9)  # [num_valid, hidden_size]

            # 计算文本和图像表示的余弦相似度
            text_pooled = valid_text_emb.sum(dim=1) / torch.clamp(valid_text_mask.sum(dim=1, keepdim=True), min=1e-9)
            caption_pooled = valid_caption_emb.sum(dim=1) / torch.clamp(
                (~valid_caption_padding_mask).sum(dim=1, keepdim=True), min=1e-9)
            cosine_sim = F.cosine_similarity(text_pooled, caption_pooled, dim=1, eps=1e-8).unsqueeze(
                1)  # [num_valid, 1]

            # 5. 多特征融合
            # 将注意力焦点和余弦相似度与平均表示拼接
            combined_features = torch.cat([
                mean_attn_output_valid,  # 交叉注意力特征
                cosine_sim,  # 余弦相似度特征
            ], dim=1)
            # 5. 批处理线性投影，得到 Logits
            logits_valid = self.text_caption_attn_output_projection(combined_features)  # [num_valid, 1]

            # 6. 计算有效样本的相关性得分 (用于门控)
            scores_valid = torch.sigmoid(logits_valid).to(attention_mask.device)  # [num_valid, 1]

            # 7. 计算 CRD 损失 (仅针对有效样本) 用于得到一个门控
            target_labels_valid = score[valid_indices].unsqueeze(1).to(attention_mask.device)  # [num_valid, 1]
            # 7. 计算 CRD 损失 (仅针对有效样本) 只包含交叉注意力
            # target_labels_valid = score[valid_indices].unsqueeze(1).unsqueeze(-1).to(attention_mask.device)  # [num_valid, 1]
            # # print("weidu", logits_valid.size(), target_labels_valid.size())
            # target_labels_valid = target_labels_valid.repeat(1, logits_valid.size(1), logits_valid.size(2)).to(attention_mask.device)
            # print("weidu", logits_valid.size(), target_labels_valid.size())
            # 再转为硬标签
            # hard_labels = (target_labels_valid > self.threshold).float()
            #
            # criterion_crd = nn.BCELoss()
            # loss_crd_hard = criterion_crd(scores_valid, hard_labels)
            # 就正常的使用软标签
            loss_crd_soft = F.binary_cross_entropy(scores_valid, target_labels_valid)  # 无需阈值处理
            # loss_crd = 0.6 * loss_crd_soft + 0.4 * loss_crd_hard
            loss_crd = loss_crd_soft
            # 8. 更新完整批次的 logits 和 scores
            # 使用 scatter_ 或 index_put_ 更新特定索引的值
            # crd_logits_all[valid_indices] = logits_valid
            relevance_scores_all[valid_indices] = scores_valid

        threshold = torch.sigmoid(self.threshold)
        temperature = torch.abs(self.temperature) + 1.0  # 确保温度参数为正

        # 平滑阈值处理
        adjusted_scores = torch.sigmoid((relevance_scores_all - threshold) * temperature)

        # 硬阈值+软权重结合
        hard_mask = (relevance_scores_all > adjusted_scores).bool()
        gating_scores = relevance_scores_all * hard_mask
        # gating_scores = relevance_scores_all
        # 门控形式需要多一维
        gating_scores = gating_scores.unsqueeze(1).to(attention_mask.device)
        caption = caption * gating_scores

        # 把iuput_id里面的对应数据也改了
        image = image * gating_scores
        mask_image_mask = image_mask.repeat(1, 1, x.size(-1)).to(attention_mask.device)
        x = x.masked_scatter(
            mask_image_mask,
            image[mask_image_mask]
        )
        caption_mask_1 = caption_mask.repeat(1, 1, x.size(-1)).to(attention_mask.device)
        x = x.masked_scatter(
            caption_mask_1,
            caption[caption_mask_1]
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                # x, attn = encoder_layer(x,
                #                         attention_mask,
                #                         output_attentions=output_attentions)
                x, attn = encoder_layer(
                    x,
                    attention_mask=attention_mask,
                    cross_modal_features=caption,  # 需要从外部传入
                    output_attentions=output_attentions
                )
            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(
            last_hidden_state=x.float(),
            hidden_states=tuple([hs.float() for hs in encoder_states]) if encoder_states else None,
            attentions=tuple([attn.float() for attn in all_attentions]) if all_attentions else None
        ), loss_crd, gating_scores, hard_mask


class MultiModalBartEncoder_for_Generating_aspect_prompt(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self,
                 use_generated_prompt,
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_aspect_prompt):
        super().__init__()

        self.use_generated_prompt = use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_aspect_prompt = use_different_aspect_prompt

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name,
                                           num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        self.aspect_linear = nn.Linear(768, 768)
        self.aspect_relu = nn.LeakyReLU()

    def _embed_multi_modal(self, generated_aspect_prompt, aspects_num, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        mask = (input_ids == self.img_feat_id) | (
                input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value

        prompt_mask = (input_ids == self.aspect_prompt_token_id)

        if self.use_generated_prompt:
            if self.use_different_aspect_prompt:
                # self.aspect_linear = self.aspect_linear.to(generated_aspect_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_aspect_prompt.device)

                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    for j in range(aspect_num):
                        # aspect_linear = nn.Linear(768, 768).to(
                        #     generated_aspect_prompt.device)  ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        # aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        prompt_embedding = self.aspect_linear(generated_aspect_prompt[index])
                        prompt_embedding = self.aspect_relu(prompt_embedding)
                        ###可以加入激活函数
                        # prompt_embedding = nn.LeakyReLU(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)
                    embedded[index, prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    prompt_embedding_ = generated_aspect_prompt[index].repeat(aspect_num, 1)

                    embedded[index, prompt_mask[index]] = prompt_embedding_
        return embedded

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_prompt, aspects_num,
                                                input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class MultiModalBartEncoder_for_Generating_sentiment_prompt(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.
    ！！！这个部分是对整个Prompt 进行编码的过程， 后续解码后就得到了所有PRompt的可能结果
    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, use_generated_prompt,
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_senti_prompt, prompt_pool_num,
                 diversity_loss_weight, l2_reg_weight, is_few_shot):
        super().__init__()
        # 前三个是特殊标记
        self.use_generated_prompt = use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_senti_prompt = use_different_senti_prompt  # 是否使用SPD的模块生成Prompt
        # 前两个是特殊标记  encoder  是 BART模型的编码器部分 所以这些都是BART 的 编码器包含的信息
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        # 图像编码表示 -> 文本表示的结果
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name,
                                           num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm
        self.is_few_shot = is_few_shot
        self.aspect_linear = nn.Linear(768, 768)
        self.aspect_relu = nn.LeakyReLU()
        self.aspect_MLP = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 768),
            # nn.LayerNorm(768),
            # nn.GELU(),
            # nn.Dropout(0.1)
        )
        # 添加部分：
        # 1、情绪Prompt池部分。这个维度保持与768一致
        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_num, 768))  # 假设最大5个aspect
        # 2、用于修正Prompt的转换器MLP
        # self.text_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768),
        #     nn.Dropout(0.2),
        #     nn.Linear(768, 768)
        # )
        # # 用于将Prompt也转换的MLP
        # self.prompt_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768),
        #     nn.Dropout(0.2),
        #     nn.Linear(768, 768)
        # )
        # 注意力的头数也是一个可以修改的指标
        # 6.1 4 8 6 -> 6 6 6 0.2 -> 0.3 -> 2 2
        if not is_few_shot:
            self.attention = nn.MultiheadAttention(768, 4, batch_first=True, dropout=0.1)  # 4头注意力
            self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层
            # 方法3的权重计算方法的权重矩阵。
            self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768

            # 计算损失函数的权重
            self.diversity_loss_weight = diversity_loss_weight
            self.l2_reg_weight = l2_reg_weight
            self.final_LayerNorm = nn.LayerNorm(768)
            # 接下来是关于图像嵌入与文本嵌入的部分：
            self.image_cross_attention = nn.MultiheadAttention(768, 4, batch_first=True, dropout=0.1)  # 4头注意力
            self.gate_proj_image = nn.Linear(768 * 2, 1)
            self.prompt_attention = nn.MultiheadAttention(768, 4, batch_first=True, dropout=0.1)
            self.prompt_proj_image = nn.Linear(768 * 2, 1)
        else:
            # 0.2->0.4 不好 ->0.1
            self.attention = nn.MultiheadAttention(768, 2, batch_first=True, dropout=0.2)  # 4头注意力
            self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层
            # 方法3的权重计算方法的权重矩阵。
            self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768

            # 计算损失函数的权重
            self.diversity_loss_weight = diversity_loss_weight
            self.l2_reg_weight = l2_reg_weight
            self.final_LayerNorm = nn.LayerNorm(768)
            # 接下来是关于图像嵌入与文本嵌入的部分：
            self.image_cross_attention = nn.MultiheadAttention(768, 6, batch_first=True, dropout=0.2)  # 4头注意力
            self.gate_proj_image = nn.Linear(768 * 2, 1)
            self.prompt_attention = nn.MultiheadAttention(768, 2, batch_first=True, dropout=0.2)
            self.prompt_proj_image = nn.Linear(768 * 2, 1)

    def _embed_multi_modal(self, generated_senti_prompt, aspects_num, input_ids, image_features,
                           sentence_mask, image_mask, encoder_outputs, image_caption_valid, image_caption_mask):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        mask = (input_ids == self.img_feat_id) | (
                input_ids == self.cls_token_id)  # 定位图像特征位置
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)  # 图像编码表示 -> 文本表示的结果
        embedded = self.embed_tokens(input_ids)  # 分词结果变成了Token值
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value
        # 为Prompt的Ps 部分先编码
        if self.use_generated_prompt:
            senti_prompt_mask = (input_ids == self.senti_prompt_token_id)  # 指示整个输入文本的需要放置情绪Prompt的位置
            # import ipdb; ipdb.set_trace()
            if self.use_different_senti_prompt:  # 是否需要为每一个方面都设计一个不同的情绪Prompt
                # self.aspect_linear = self.aspect_linear.to(generated_senti_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_senti_prompt.device)
                # 遍历一个batch的样本个数
                for index in range(len(aspects_num)):  # 再MASC任务，Aspect的数量已知
                    aspect_num = aspects_num[index]
                    sentence_mask_tmp = sentence_mask[index]
                    image_mask_tmp = image_mask[index]
                    encoder_outputs_tmp = encoder_outputs[index]
                    image_caption_valid_tmp = image_caption_valid[index]
                    image_caption_mask_tmp = image_caption_mask[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    encoder_outputs_tmp_attn = encoder_outputs_tmp.unsqueeze(0).to(input_ids.device)
                    # 遍历一个样本中的方面个数
                    for j in range(aspect_num):
                        # 为每一个生成的情绪Prompt做一个线性变换 借助图像表示编码解码后的结果，得到情绪相关的embedding的结果
                        # BART base 512 Large 1024
                        # aspect_linear = nn.Linear(768, 768).to(generated_senti_prompt.device)
                        # aspect_relu = nn.LeakyReLU().to(generated_senti_prompt.device)
                        # prompt_embedding = self.aspect_linear(generated_senti_prompt[index])
                        # prompt_embedding = self.aspect_relu(prompt_embedding)
                        prompt_embedding = self.aspect_MLP(generated_senti_prompt[index])
                        # 把总情绪Prompt的方法拿过来。
                        mask_expanded = sentence_mask_tmp.bool()
                        # 使用广播机制提取文本特征 [b,s,768]
                        text_embeddings = encoder_outputs_tmp[mask_expanded].unsqueeze(0).to(input_ids.device)
                        text_emb = text_embeddings.permute(1, 0, 2).to(input_ids.device)  # [s,b,768]

                        # --- 新增：图像嵌入交叉注意力计算 ---
                        # 现在利用相关度对 图像信息进行了修正， 其实可以把这个图像融合也放回来试试效果
                        image_mask_expanded_for_image = image_mask_tmp.bool()  # 为图像 mask 扩展维度
                        image_embeddings = encoder_outputs_tmp[image_mask_expanded_for_image].unsqueeze(0)  # 提取图像嵌入
                        image_emb = image_embeddings.permute(1, 0, 2).to(
                            input_ids.device)  # [s, b, 768]  图像嵌入，准备作为 Key/Value

                        image_mask_expanded_for_image_caption = image_mask_expanded_for_image.unsqueeze(-1).to(
                            input_ids.device)
                        image_embeddings_caption = encoder_outputs_tmp * image_mask_expanded_for_image_caption.float()
                        image_embeddings_caption = image_embeddings_caption.to(input_ids.device).unsqueeze(0)
                        # 为了保证交叉注意力的计算，这里用乘法保持长度一致。
                        caption_mask_expand = image_caption_mask_tmp.unsqueeze(-1).to(input_ids.device)  # [b, s, 1]
                        image_caption_valid_expand = image_caption_valid_tmp.unsqueeze(-1).unsqueeze(-1).to(
                            input_ids.device)  # [b, 1, 1]
                        caption = encoder_outputs_tmp * caption_mask_expand.float()
                        caption = caption.unsqueeze(0)
                        sample_caption_embedding = caption * image_caption_valid_expand + image_embeddings_caption * (
                                1 - image_caption_valid_expand)
                        sample_caption_mask = caption_mask_expand * image_caption_valid_expand + image_mask_expanded_for_image_caption * (
                                1 - image_caption_valid_expand)
                        sample_caption_embedding = sample_caption_embedding.to(input_ids.device)
                        sample_caption_mask = sample_caption_mask.to(input_ids.device)
                        sample_caption_mask = ~sample_caption_mask.squeeze(-1).bool()
                        sample_caption_mask = sample_caption_mask.unsqueeze(0)
                        # 扩展prompt池到batch维度 [prompt_num, b, 768]
                        # prompt_pool = self.prompt_pool.unsqueeze(1).expand(-1, text_emb.size(1), -1).to(
                        #     input_ids.device)
                        # print("text 维度", text_emb.size())
                        senti_features = prompt_embedding.unsqueeze(0)
                        # 注意力计算 [s,b,768]
                        # print("维度", senti_features.size(), text_embeddings.size())\
                        context_only_mask_hf = image_mask_tmp + image_caption_mask_tmp + sentence_mask_tmp
                        context_only_mask_hf = context_only_mask_hf.unsqueeze(0).to(input_ids.device)
                        cross_attention_key_padding_mask = (context_only_mask_hf == 0)
                        # attn_output, _ = self.attention(
                        #     query=senti_features,
                        #     key=text_embeddings,
                        #     value=text_embeddings,
                        #     key_padding_mask=None
                        # )
                        # 直接一步到位

                        # print("ww", senti_features.size(), encoder_outputs_tmp_attn.size(), cross_attention_key_padding_mask.size())
                        attn_output, _ = self.attention(
                            query=senti_features,
                            key=encoder_outputs_tmp_attn,  # Shape: (SeqLen, Batch, Dim)
                            value=encoder_outputs_tmp_attn,
                            key_padding_mask=cross_attention_key_padding_mask  # Shape: (Batch, SeqLen)
                        )

                        gate = torch.sigmoid(self.gate_proj(torch.cat([senti_features, attn_output], dim=-1)))
                        # 方式1： 直接将门控机制运用在原始和Prompt上
                        # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
                        # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。

                        final_features = gate * senti_features + (1 - gate) * (attn_output)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控

                        # final_features = senti_features + attn_output
                        # 调整一下需不需要保证原始信息居多的情况
                        # final_features = gate * senti_features + (1 - gate) * (attn_output)
                        # final_features = final_features.to(input_ids.device)
                        # 再把信息放回text_embedding的部分。
                        # text_emb_for_image_attn = text_embeddings.permute(1, 0, 2).to(
                        #     input_ids.device)  # 使用 Prompt 增强后的文本表示 final_features 作为 Query (需要调整维度顺序)

                        # ----------- 再增加与图像嵌入的修正部分
                        # 注意力计算 [s,b,768]
                        # image_cross_attn_output, _ = self.image_cross_attention(  # 图像交叉注意力计算
                        #     query=final_features,  # Query: Prompt 增强后的文本表示
                        #     key=image_emb,  # Key: 图像嵌入
                        #     value=image_emb,  # Value: 图像嵌入
                        #     key_padding_mask=None  # 图像部分是否需要 padding mask？ 根据实际情况添加
                        # )  # image_cross_attn_output: [s, b, 768]  图像交叉注意力输出
                        # print("ww", sample_caption_embedding.size(), sample_caption_mask.size())
                        # 其他模态的融合
                        # ------------
                        # image_cross_attn_output, _ = self.image_cross_attention(  # 图像交叉注意力计算
                        #     query=final_features,  # Query: Prompt 增强后的文本表示
                        #     key=sample_caption_embedding,  # Key: 图像嵌入
                        #     value=sample_caption_embedding,  # Value: 图像嵌入
                        #     key_padding_mask=sample_caption_mask  # 图像部分是否需要 padding mask？ 根据实际情况添加
                        # )  # image_cross_attn_output: [s, b, 768]  图像交叉注意力输出
                        #
                        # image_cross_attn_output_result = image_cross_attn_output.permute(1, 0, 2)
                        #
                        # gate_image = torch.sigmoid(
                        #     self.gate_proj_image(torch.cat([final_features, image_cross_attn_output_result],
                        #                                    dim=-1)))
                        # # 把融合的特征也放入其中。
                        # final_features = gate_image * final_features + (1 - gate_image) * (
                        #         final_features + image_cross_attn_output_result)

                        # final_features = final_features + image_cross_attn_output_result
                        # -------------
                        # 调整一下需不需要保证原始信息居多的情况
                        # final_features = gate_image * final_features + (1 - gate_image) * (
                        #         image_cross_attn_output_result)

                        # # MLP处理 ------------------------------------------------------
                        # Prompt = self.prompt_pool.unsqueeze(0)  # [1, num, 768]
                        # prompt_attn, _ = self.prompt_attention(
                        #     query=final_features,
                        #     key=Prompt,
                        #     value=Prompt,
                        #     key_padding_mask=None
                        # )
                        # gate_prompt = torch.sigmoid(self.prompt_proj_image(torch.cat([final_features, prompt_attn], dim=-1)))
                        # final_features = gate_prompt * final_features + (1 - gate_prompt) * (prompt_attn + final_features)

                        # mlp_output = self.prompt_mlp(attn_output.permute(1, 0, 2)).to(
                        #     input_ids.device)  # [b,s,768]
                        # text_mlp = self.text_mlp(text_embeddings).to(input_ids.device)  # [b,s,768]
                        # # print("计算权重前两者维度", mlp_output.shape, text_mlp.shape)
                        # # 权重计算与融合 -------------------------------------------------
                        # weights = self.compute_correlation_weights_learnable(text_mlp, mlp_output)  # [b, n, n]
                        # # 加权融合原始文本特征
                        # # print("两种维度大小", weights.shape, text_embeddings.shape)
                        # # prompt_text = attn_output.permute(1, 0, 2)
                        #
                        # enhanced_text = torch.matmul(weights, mlp_output)  # [b,s,768] 好像之前乘这个结果也很高呢。
                        # # enhanced_text = torch.matmul(weights, prompt_text)
                        # enhanced_text = enhanced_text.to(input_ids.device)
                        # 融合Prompt信息和原文的文本嵌入表示

                        # 在forward方法中添加：
                        # print("w维度", senti_features.size(), attn_output.size())
                        # 也增加残差链接和归一化
                        # final_features = final_features + senti_features
                        # final_features = self.final_LayerNorm(final_features)

                        final_features = final_features.squeeze(0)  # 再把batch维度去除
                        ###可以加入激活函数å
                        # prompt_embedding = nn.LeakyReLU(prompt_embedding)
                        prompt_embedding_list.append(final_features)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)

                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]

                    prompt_embedding_ = generated_senti_prompt[index].repeat(aspect_num, 1)

                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
        return embedded  # 在一个batch中的 每一个样本中的每一个Aspect上用一个Prompt 把

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                sentence_mask=None,
                image_mask=None,
                encoder_outputs=None,
                image_caption_valid=None,
                image_caption_mask=None,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_prompt, aspects_num,
                                                input_ids, image_features, sentence_mask, image_mask,
                                                encoder_outputs, image_caption_valid,
                                                image_caption_mask) * self.embed_scale
        # 这里的处理逻辑一样 前面的处理只是让SPD情绪处理和图像表示的的内容经过Embedding得到隐藏层状态， 后续还需要经过BART的Encoder
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        # 后续的输出就包含了Encoder的每一层的状态以及最终层输出，那很完美，就是我们需要的信息。
        return BaseModelOutput(
            last_hidden_state=x.float(),
            hidden_states=tuple([hs.float() for hs in encoder_states]) if encoder_states else None,
            attentions=tuple([attn.float() for attn in all_attentions]) if all_attentions else None
        )

    def compute_correlation_weights(self, tensor1, tensor2):
        # 将Prompt和文本特征进行一个矩阵乘法 只是参考论文的实现权重的放大

        interaction_scores = torch.einsum('bsd,bpd->bsp',
                                          tensor1,
                                          tensor2)
        # Sigmoid激活
        sigmoid_scores = torch.sigmoid(interaction_scores)  # [b, n, n]

        # 动态归一化 [b, n, 1]
        s_min = sigmoid_scores.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        s_max = sigmoid_scores.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        epsilon = 1e-8  # 防止除零

        # 归一化公式 [n, n]
        weights = (sigmoid_scores - s_min) / (s_max - s_min + epsilon)

        return weights

    # 第二种方法是常规的余弦相似度计算

    def compute_correlation_weights_cosine(self, tensor1, tensor2):
        """使用余弦相似度计算权重"""
        # 归一化张量以计算余弦相似度
        tensor1_normalized = F.normalize(tensor1, p=2, dim=-1)  # [b, s, d]
        tensor2_normalized = F.normalize(tensor2, p=2, dim=-1)  # [b, p, d]

        # 计算余弦相似度矩阵 [b, s, p] (或 bsp -> bsn 如果p和n一致)
        similarity_scores = torch.matmul(tensor1_normalized, tensor2_normalized.transpose(1, 2))

        # 可以选择是否使用 Sigmoid 或其他激活函数，或者直接使用相似度作为权重
        weights = torch.sigmoid(similarity_scores)  # 例如使用 Sigmoid 激活

        return weights

    # 第三种方法则是使用可学习的权重矩阵，不直接计算相似度，而是让模型自已学习
    def compute_correlation_weights_learnable(self, tensor1, tensor2):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          self.learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活
        # weights = interaction_scores
        return weights

    # 计算损失的两个部分，
    def diversity_loss_cosine_distance(self):
        """计算 Prompt 池的余弦距离多样性损失"""
        prompt_pool = self.prompt_pool  # 直接使用 self.prompt_pool
        num_prompts = prompt_pool.size(0)
        if num_prompts <= 1:
            return torch.tensor(0.0, device=prompt_pool.device)

        prompt_pool_normalized = F.normalize(prompt_pool, p=2, dim=1)
        similarity_matrix = torch.matmul(prompt_pool_normalized, prompt_pool_normalized.transpose(0, 1))
        mask = 1 - torch.eye(num_prompts, device=prompt_pool.device)
        masked_similarity_matrix = similarity_matrix * mask
        diversity_loss = masked_similarity_matrix.sum() / (num_prompts * (num_prompts - 1) + 1e-8)
        return diversity_loss * self.diversity_loss_weight  # 应用权重

    def l2_regularization_loss(self):
        """计算 Prompt 池的 L2 正则化损失"""
        l2_reg_loss = torch.sum(self.prompt_pool ** 2)  # 计算 Prompt 池参数的平方和
        return l2_reg_loss * self.l2_reg_weight  # 应用权重


class MultiModalBartEncoder_for_Generating_Dual_prompts(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self,
                 use_generated_aspect_prompt, use_generated_senti_prompt,
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_aspect_prompt, use_different_senti_prompt,
                 NEU_id, POS_id, NEG_id):
        super().__init__()

        self.use_generated_aspect_prompt = use_generated_aspect_prompt
        self.use_generated_senti_prompt = use_generated_senti_prompt

        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_aspect_prompt = use_different_aspect_prompt
        self.use_different_senti_prompt = use_different_senti_prompt

        # if self.use_different_senti_prompt:
        #     self.attention_for_senti_prompt = Attention_for_Senti_Prompt(n_head=8, model_dim=768, drop_rate=0.2)
        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        self.neu_id = NEU_id
        self.pos_id = POS_id
        self.neg_id = NEG_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name,
                                           num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()
        # self.aspect_linear = nn.Sequential(nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768))

    def _embed_multi_modal(self, generated_aspect_prompt, generated_senti_prompt, aspects_num, input_ids,
                           image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # import ipdb; ipdb.set_trace()
        device = generated_aspect_prompt.device
        batch_size = input_ids.size(0)
        mask = (input_ids == self.img_feat_id) | (
                input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value

        ipdb.set_trace()
        if self.use_generated_aspect_prompt:
            ##aspect_prompt
            aspect_prompt_mask = (input_ids == self.aspect_prompt_token_id)
            if self.use_different_aspect_prompt:
                # self.aspect_linear = self.aspect_linear.to(device)
                # self.aspect_relu = self.aspect_relu.to(device)
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    aspect_prompt_embedding_list = []
                    for j in range(aspect_num):
                        aspect_linear = nn.Linear(768, 768).to(
                            generated_aspect_prompt.device)  ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        aspect_prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        aspect_prompt_embedding = aspect_relu(aspect_prompt_embedding)
                        aspect_prompt_embedding_list.append(aspect_prompt_embedding)
                    aspect_prompt_embedding_ = torch.cat(aspect_prompt_embedding_list, dim=0)
                    embedded[index, aspect_prompt_mask[index]] = aspect_prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    aspect_prompt_embedding_ = generated_aspect_prompt[index].repeat(aspect_num, 1)
                    embedded[index, aspect_prompt_mask[index]] = aspect_prompt_embedding_

        ##sentiment_prompt

        if self.use_generated_senti_prompt:
            '''
            # if self.use_different_senti_prompt:
            以下使用的是attention机制，senti_prompt_token和sentiments_embdedding
            # sentiments_ids = torch.tensor([self.neu_id, self.pos_id, self.neg_id]).to(device)
            # sentiments_embdedding = self.embed_tokens(sentiments_ids)
            # senti_prompt_mask = (input_ids == self.senti_prompt_token_id)
            # for index in range(len(aspects_num)):
                
            #     aspect_num = aspects_num[index]
            #     expanded_sentiments_embdedding = sentiments_embdedding.expand(aspect_num, sentiments_embdedding.size(0), sentiments_embdedding.size(1))
            #     original_senti_prompt = embedded[index, senti_prompt_mask[index]].unsqueeze(1)
            #     new_senti_prompt = self.attention_for_senti_prompt(original_senti_prompt, expanded_sentiments_embdedding, expanded_sentiments_embdedding).squeeze()
            #     # import ipdb; ipdb.set_trace()
            #     embedded[index, senti_prompt_mask[index]] = new_senti_prompt
            '''
            ##换成senti_prompt也是生成形式看看
            senti_prompt_mask = (input_ids == self.senti_prompt_token_id)
            # import ipdb; ipdb.set_trace()
            if self.use_different_senti_prompt:
                # self.aspect_linear = self.aspect_linear.to(generated_senti_prompt.device)
                # self.aspect_relu = self.aspect_relu.to(generated_senti_prompt.device)

                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    for j in range(aspect_num):
                        senti_linear = nn.Linear(768, 768).to(generated_senti_prompt.device)
                        senti_relu = nn.LeakyReLU().to(generated_senti_prompt.device)
                        prompt_embedding = senti_linear(generated_senti_prompt[index])
                        prompt_embedding = senti_relu(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
                    prompt_embedding_ = torch.cat(prompt_embedding_list, dim=0)
                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_
            else:
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    prompt_embedding_ = generated_senti_prompt[index].repeat(aspect_num, 1)

                    embedded[index, senti_prompt_mask[index]] = prompt_embedding_

        return embedded

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                generated_aspect_prompt=None,
                generated_senti_prompt=None,
                aspects_num=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(generated_aspect_prompt, generated_senti_prompt, aspects_num,
                                                input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class MultiModalBartDecoder_span(nn.Module
                                 ):  # AOE task and all downstream tasks
    def __init__(self,
                 config: MultiModalBartConfig,
                 tokenizer,
                 decoder,
                 pad_token_id,
                 label_ids,
                 causal_mask,
                 need_tag=True,
                 only_sc=False,
                 avg_feature=False,
                 use_encoder_mlp=True):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.causal_mask = causal_mask
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        # label_ids = sorted(label_ids, reverse=False)
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids) + 1
        self.need_tag = need_tag
        self.only_sc = only_sc
        mapping = torch.LongTensor([0, 2] + label_ids)
        ###mapping: [0, 2, 50276, 50277, 50278, 50281]
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.dropout_layer = nn.Dropout(0.1)

        self.end_text_id = tokenizer.end_text_id
        self.avg_feature = avg_feature
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.Dropout(0.3),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state, only_sc=False):
        # import ipdb; ipdb.set_trace()
        '''
        tokens: [[0, 2, 2, 16, 16, 4, 18, 18, 4, 1, 1, 1, 1],
                 [0, 2, 2, 15, 16, 3, 25, 26, 5, 28, 28, 4, 1]]
        '''
        # import ipdb; ipdb.set_trace()
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output  ##(batch, 72=38(len(image_token+begin_image+end_image(36+1+1)))+34(max_tex_len(包含begin_text_id(0) and end_text_id(2)) in batch), 768)
        encoder_pad_mask = state.encoder_mask  ##(batch, 72)
        first = state.first
        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(
            self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(
            src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        # print("src_token的维度", src_tokens.shape)  # (2, 34=(max_len - end_index))
        # 这个的维度长度大小对应于 数据集中的文本模态特征
        end_index = state.end_index  # 这里做出修改，源代码的所有64 都只是代表了image_token=2的情况，所以它并不是一个通用的处理
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)  ###Sequence
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        # print('word_mapped_tokens', word_mapped_tokens)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens,
                             word_mapped_tokens)

        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        '''
        {'AESC': 50281, 'POS': 50276, 'NEU': 50277, 'NEG': 50278}
        tensor([[0, 50276, 50276, 4644, 4644, 50278, 798, 798, 50278, 2, 1, 1, 1],
                [0, 50276, 50276, 9517, 957, 50277, 2561, 7772, 50281, 2762, 2762, 50278, 2]])
        将tokens中的index以及标签都转化为vocabulary中的token_id
        '''

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(
                self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.
                                causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=self.
                                causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)

        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size (2, 12(去掉了 end_token_id), 768)
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        # myf: 从此处开始的代码就是源码中为了生成标准三元组构建的内容了，为标准的指针网络，但是对于MASC这个任务来说，复杂的生成任务反而不太好
        # 所以我们单独使用一个分类器任务，与Aspect_num的构建使用同样的方式， 故我们同样将dict这个decoder结果返回到各个MAESC代码中使用

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1),
             self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)
        ##建立空的logits
        # print('logits', logits.shape) (bsz, max_len,  self.src_start_index + src_tokens.size(-1)) -> (2, 12, 40=6+34)
        # 首先计算的是

        if self.need_tag:
            '''
            self.decoder.embed_tokens.weight: (50289, 768)
            self.label_start_id: 50276
            '''
            tag_scores = F.linear(
                hidden_state,
                self.dropout_layer(
                    self.decoder.embed_tokens.
                    weight[self.label_start_id:self.label_start_id +
                                               3]))  # bsz x max_len x num_class
            logits[:, :, 3:self.src_start_index] = tag_scores  ###给情感的position赋值[:, :, (3, 4, 5)]
        if not only_sc:
            eos_scores = F.linear(
                hidden_state,
                self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))
            '''
            ['</s>(eos_token)', '<mask>', '<pad>', '<s>(bos_token)', '<unk>']
            [2, 50264, 1, 0, 3]
            '''

            # bsz x max_bpe_len(image_len + text_len) x hidden_size: (2, 72, 768)
            src_outputs = state.encoder_output
            if hasattr(self, 'encoder_mlp') and not only_sc:
                src_outputs = self.encoder_mlp(src_outputs)

            if first is not None:
                mask = first.eq(0)
                src_outputs = src_outputs.gather(
                    index=first.unsqueeze(2).repeat(1, 1,
                                                    src_outputs.size(-1)),
                    dim=1)
            else:
                # mask = state.encoder_mask[:, 64:].eq(0)  # 故此处的64 -> end_index
                mask = state.encoder_mask[:, end_index:].eq(0)
                # src_outputs = self.decoder.embed_tokens(src_tokens)
            mask = mask.unsqueeze(1)  ## bsz x 1 x max_word_len: (2, 1, 34=(max_len - end_index))
            input_embed = self.decoder.embed_tokens(
                src_tokens)  # bsz x max_word_len x hidden_size: (2, 34, 768); src_tokens: (2, 34=(max_len - end_index))
            input_embed = self.dropout_layer(input_embed)
            if self.avg_feature:  # 先把feature合并一下
                # src_outputs = (src_outputs[:, 64:] + input_embed) / 2  # 故此处的64 -> end_index
                src_outputs = (src_outputs[:, end_index:] + input_embed) / 2
            word_scores = torch.einsum(
                'blh,bnh->bln', hidden_state,
                src_outputs[:, end_index:])  # bsz x max_len x max_word_len: (2, 12, 34=(max_len - end_index))
            # 故此处的64 -> end_index  src_outputs[:, 64:]
            if not self.avg_feature:
                gen_scores = torch.einsum(
                    'blh,bnh->bln', hidden_state,
                    input_embed)  # bsz x max_len x max_word_len: (2, 12, 34)
                word_scores = (gen_scores + word_scores) / 2
            mask = mask.__or__(
                src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))  ###(2, 1, 34)
            word_scores = word_scores.masked_fill(mask, -1e32)  ###(bts, max_len, max_word_len)
            logits[:, :, self.src_start_index:] = word_scores
            ###logits.shape (bts, max_len, max_word_len+6): (2, 12, 40)
            logits[:, :, 1:2] = eos_scores
        # print(torch.argmax(logits[0], dim=-1))
        return hidden_state, logits

    def decode(self, tokens, state, only_sc=False):
        # print("输出", self(tokens, state, only_sc)[0], self(tokens, state, only_sc)[1])
        return self(tokens, state, only_sc)[1][:, -1]


class Span_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.loss_fct = nn.CrossEntropyLoss()
        self.fc = nn.LogSoftmax(dim=-1)
        # 增加情感权重
        self.emotion_ids = [3, 4, 5]  # 默认为空（不特殊处理）
        if not args.is_classifier:
            if args.dataset[0][0] == 'twitter15':
                self.class_weights = [1.6, 1.0, 3.0]  # 积极 中性 消极
            else:
                self.class_weights = [1.0, 1.0, 1.3]
            # if args.dataset[0][0] == 'twitter15':
            #     self.class_weights = [1.7, 0.8, 2.0]  # 积极 中性 消极
            # else:
            #     self.class_weights = [1.5, 1.0, 1.7]
        # 如果采用生成任务的话，整个损失都需要变成0, 放弃模型使用生成的部分
        else:
            if args.dataset[0][0] == 'twitter15':
                self.class_weights = [0.0, 0.0, 0.0]  # 积极 中性 消极
            else:
                self.class_weights = [0.0, 0.0, 0.0]
        self.default_other_token_weight = 1
        self.default_other_token_weight_emotion = 0
        self.pad_token_id = 1
        self.focal_loss = FocalLoss()
        # 确保 emotion_ids 和 emotion_class_weights 长度一致 36 78 22 twitter15 45 77 20
        if len(self.emotion_ids) != len(self.class_weights):
            raise ValueError("emotion_ids and emotion_class_weights must have the same length.")

        # 创建一个从情感ID到其特定权重的映射，方便查找
        self.emotion_id_to_weight_map = {
            eid: weight_val
            for eid, weight_val in zip(self.emotion_ids, self.class_weights)
        }

    def forward(self, tgt_tokens, pred, mask):
        '''
        tgt_tokens: (2 (batch-size), 12 (max_len+1))
        pred: (2, 12, 40 (max_word_len))
        '''
        # 1. 获取当前批次/样本的实际 vocab_size
        current_vocab_size = pred.size(-1)  # 例如，您提到的 40

        # 2. 动态构建当前批次的类别权重张量
        # 初始化所有 token 的权重为 default_other_token_weight
        current_class_weights = torch.full(
            (current_vocab_size,),
            self.default_other_token_weight,
            dtype=pred.dtype,  # 与 pred 的 dtype 一致
            device=pred.device  # 与 pred 的 device 一致
        )
        current_class_weights_emotion = torch.full(
            (current_vocab_size,),
            self.default_other_token_weight_emotion,
            dtype=pred.dtype,  # 与 pred 的 dtype 一致
            device=pred.device  # 与 pred 的 device 一致
        )

        # 为特定的情感 ID 设置其权重
        for eid, specific_weight in self.emotion_id_to_weight_map.items():
            if 0 <= eid < current_vocab_size:  # 确保情感ID在当前词汇表范围内
                current_class_weights[eid] = specific_weight
                current_class_weights_emotion[eid] = specific_weight
            # else:
            # 如果某个预定义的情感ID超出了当前动态词汇表的范围，
            # 它的权重将保持为 default_other_token_weight (或者您可以选择报错或跳过)
            # print(f"Warning: Emotion ID {eid} is out of current_vocab_size ({current_vocab_size}).")

        # 3. 准备 target_labels (将 padding 替换为 ignore_index)
        target_labels = tgt_tokens.clone()

        tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)

        # 4. 计算加权交叉熵损失
        output = F.cross_entropy(
            input=pred.transpose(1, 2),  # (batch, current_vocab_size, seq_len)
            target=tgt_tokens,  # (batch, seq_len)
            weight=current_class_weights,  # **使用动态构建的权重**
            ignore_index=-100,
            reduction='mean',
        )
        per_token_loss = F.cross_entropy(
            input=pred.transpose(1, 2),
            target=tgt_tokens,
            weight=current_class_weights_emotion,
            ignore_index=-100,
            reduction='none'
        )
        # 用Focal Loss
        # output = self.focal_loss(
        #     pred.transpose(1, 2),  # (batch, current_vocab_size, seq_len)
        #     tgt_tokens,  # (batch, seq_len)
        #     current_class_weights,  # **使用动态构建的权重**
        # )
        return output, per_token_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        # alpha 可以是一个 float (所有类别共享) 或一个 Tensor (每个类别一个)
        # 如果 alpha 是 Tensor，其长度应为 vocab_size，并且在情感标签 ID 处设置特定权重
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets, alpha):
        # logits: (B, C, L) or (B*L, C) - C 是类别数 (output_vocab_size)
        # targets: (B, L) or (B*L)
        self.alpha = alpha

        # 打印输入形状和 targets 的值范围，以便调试
        print(f"FocalLoss - logits shape: {logits.shape}")  # 应该是 (N, C)
        print(f"FocalLoss - targets shape: {targets.shape}")  # 应该是 (N)
        if targets.numel() > 0:  # 只有当 targets 非空时才计算 min/max
            print(f"FocalLoss - targets min: {targets.min()}, targets max: {targets.max()}")
            print(f"FocalLoss - num_classes (C from logits): {logits.size(1)}")

        # 检查 targets 是否越界 (在替换 ignore_index 之前)
        num_classes = logits.size(1)
        if targets.numel() > 0 and ((targets[targets != self.ignore_index].min() < 0) or \
                                    (targets[targets != self.ignore_index].max() >= num_classes)):
            print("!!! ERROR in FocalLoss: Targets (excluding ignore_index) are out of bounds !!!")
            problematic_targets_original = targets[
                (targets != self.ignore_index) & ((targets < 0) | (targets >= num_classes))]
            print(f"Problematic original target values: {problematic_targets_original}")

        # 计算 log_softmax 以获得 log_probs
        log_probs = F.log_softmax(logits, dim=1)  # 在类别维度上

        # 收集真实标签对应的 log_probs
        # targets 需要是 LongTensor
        # targets.unsqueeze(1) -> (B*L, 1) if targets is (B*L)
        # log_probs_for_targets = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # 为了处理 ignore_index，我们先不 squeeze，并创建 mask
        gathered_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1))  # (B*L, 1)

        # 创建有效标签的 mask
        valid_mask = (targets != self.ignore_index)  # (B*L)

        # 应用 mask，只计算有效标签的损失
        gathered_log_probs = gathered_log_probs[valid_mask]  # (Num_Valid_Tokens, 1)
        targets_valid = targets[valid_mask]  # (Num_Valid_Tokens)

        if gathered_log_probs.numel() == 0:  # 如果没有有效标签
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs_for_targets = torch.exp(gathered_log_probs.squeeze(1))  # (Num_Valid_Tokens)

        # 计算 Focal Loss 项
        focal_term = (1 - probs_for_targets) ** self.gamma

        # 应用 alpha 权重 (如果提供)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_factor = self.alpha
            elif isinstance(self.alpha, torch.Tensor):
                # 从 alpha tensor 中选择对应类别的权重
                # targets_valid 是有效标签的真实 ID
                alpha_factor = self.alpha.to(targets_valid.device).gather(0, targets_valid)
            else:
                raise TypeError("alpha must be float, int or torch.Tensor.")
            loss = -alpha_factor * focal_term * gathered_log_probs.squeeze(1)
        else:
            loss = -focal_term * gathered_log_probs.squeeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class MultiModalBartDecoder_MLM(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.decoder.embed_tokens.num_embeddings)))

    def forward(self, labels, input_ids, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask[:decoder_input_ids.size(1), :
                                                                        decoder_input_ids.size(1)],
        )

        lm_logits = F.linear(decoder_outputs[0][:, 1:],
                             self.decoder.embed_tokens.weight,
                             bias=self.final_logits_bias)

        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                lm_logits.view(-1, self.decoder.embed_tokens.weight.size(0)),
                labels.reshape(-1))

            return lm_loss


class MultiModalBartDecoder_ANP_generate(nn.Module):  # AOG task
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.register_buffer(
            "final_logits_bias",
            torch.zeros((1, self.decoder.embed_tokens.num_embeddings)))

    def forward(self, labels, input_ids, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask[:decoder_input_ids.size(1), :
                                                                        decoder_input_ids.size(1)],
        )

        lm_logits = F.linear(decoder_outputs[0][:, 1:],
                             self.decoder.embed_tokens.weight,
                             bias=self.final_logits_bias)

        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()
            # labels[labels == self.cls_token_id] = -100
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                lm_logits.view(-1, self.decoder.embed_tokens.weight.size(0)),
                labels.reshape(-1))

            return lm_loss


class MultiModalBartDecoder_sentiment(nn.Module):  # MSP task
    def __init__(self,
                 config: MultiModalBartConfig,
                 decoder,
                 senti_ids,
                 senti_nums=3):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.senti_ids = senti_ids
        self.dropout_layer = nn.Dropout(0.1)
        self.senti_head = BartClassificationHead(config.d_model,
                                                 config.d_model, senti_nums,
                                                 config.classif_dropout)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, senti_labels, encoder_outputs, attention_mask,
                senti_decoder_input_ids):
        decoder_outputs = self.decoder(
            input_ids=senti_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=None,
            decoder_causal_mask=None,
        )

        # predict_senti = F.linear(
        #     decoder_outputs[0][:, 1],
        #     self.dropout_layer(self.decoder.embed_tokens.
        #                        weight[self.senti_ids[0]:self.senti_ids[2] +
        #                               1]))  # bsz
        # predict_senti = torch.flip(predict_senti, dims=[-1])
        predict_senti = self.senti_head(decoder_outputs[0][:, 1])
        loss_fct = nn.CrossEntropyLoss()
        senti_loss = loss_fct(predict_senti, senti_labels)
        return senti_loss, predict_senti


class MultiModalBartDecoder_MRM(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder, causal_mask,
                 args):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.causal_mask = causal_mask
        self.args = args
        self.mrm_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classif_dropout,
        )
        self._init_weights(self.mrm_head.dense)
        self._init_weights(self.mrm_head.out_proj)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, mrm_labels, mrm_masks, encoder_outputs, attention_mask,
                mrm_decoder_input_ids, mrm_decoder_attention_mask):

        decoder_padding_mask = mrm_decoder_attention_mask.eq(0)
        decoder_outputs = self.decoder(
            input_ids=mrm_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=self.causal_mask[:mrm_decoder_input_ids.size(
                1), :mrm_decoder_input_ids.size(1)].to(
                mrm_decoder_input_ids.device),
        )
        region_representation = decoder_outputs[0][mrm_masks.bool()]
        if len(region_representation) > 0:
            predict_cls = self.mrm_head(region_representation)
            loss_fct = nn.CrossEntropyLoss()
            mrm_labels = torch.cat(mrm_labels,
                                   dim=0).to(encoder_outputs.device)

            if self.args.mrm_loss_type == 'KL':
                predict_cls = F.log_softmax(predict_cls, dim=-1)
                mrm_loss = F.kl_div(predict_cls.double(),
                                    mrm_labels.double().squeeze(1),
                                    reduction='batchmean')
            else:
                raise RuntimeError("wrong mrm type")
        else:
            mrm_loss = 0

        return mrm_loss


'''
generate_aspect_prompt based on the multimodal context
'''


class MultiModalBartDecoder_generate_aspect_prompt(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.aspect_prompt_linear = nn.Linear(768, 768)

    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask):
        # import ipdb; ipdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask.eq(0),
            decoder_padding_mask=decoder_attention_mask.eq(0),
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        aspect_prompt_logits = self.aspect_prompt_linear(prompt_logits)

        return aspect_prompt_logits


'''
generate_sentiment_prompt based on the multimodal context
'''


class MultiModalBartDecoder_generate_sentiment_prompt(nn.Module):
    """
    做出如下修改。 这里是SPD的部分  然后为了更新这个Prompt池相关的内容， 直接在此处计算完成损失函数，并直接返回，在总损失中将其整合。
    """

    def __init__(self, config: MultiModalBartConfig, decoder, prompt_pool_num, diversity_loss_weight, l2_reg_weight,
                 is_few_shot, freeze_bart):
        super().__init__()
        self.config = config
        self.decoder = decoder
        if freeze_bart:
            for p in self.decoder.parameters():
                p.requires_grad = False
        # self.senti_prompt_linear = nn.Linear(768, 768)
        self.senti_prompt_linear = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768)
        )
        # 添加部分：
        # 1、情绪Prompt池部分。这个维度保持与768一致
        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_num, 768))
        # 2、用于修正Prompt的转换器MLP
        if not is_few_shot:
            self.text_mlp = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Dropout(0.2),
                nn.Linear(768, 768)
            )
            # 用于将Prompt也转换的MLP
            self.prompt_mlp = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Dropout(0.2),
                nn.Linear(768, 768)
            )
        else:
            self.text_mlp = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Dropout(0.2),
                nn.Linear(768, 768)
            )
            # 用于将Prompt也转换的MLP
            self.prompt_mlp = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Dropout(0.2),
                nn.Linear(768, 768)
            )
        self.is_few_shot = is_few_shot
        # 注意力的头数也是一个可以修改的指标
        # 5.31 集体下调 4，8，4，4，-> 2 4 2 2 (15没影响，17到是受影响了) 4 4 2 2
        self.ffn = nn.Sequential(
            nn.Linear(768, 768 * 4),
            nn.GELU(), # 或者 nn.GELU()，GELU在Transformer中更常用
            nn.Dropout(0.2),
            nn.Linear(768 * 4, 768)
        )
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(768)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        if not is_few_shot:
            self.attention = nn.MultiheadAttention(768, 4, dropout=0.2)  # 4头注意力
            self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层
            # 方法3的权重计算方法的权重矩阵。
            self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768

            # 计算损失函数的权重
            self.diversity_loss_weight = diversity_loss_weight
            self.l2_reg_weight = l2_reg_weight
            self.final_LayerNorm = nn.LayerNorm(768)
            # 接下来是关于图像嵌入与文本嵌入的部分：
            self.image_cross_attention = nn.MultiheadAttention(768, 6, dropout=0.2)  # 4头注意力
            self.gate_proj_image = nn.Linear(768 * 2, 1)
            # 名词增强部分
            self.nouns_cross_attention = nn.MultiheadAttention(768, 4, batch_first=True, dropout=0.2)
            self.gate_proj_nouns = nn.Linear(768 * 2, 1)

            self.nouns_cross_attention_image = nn.MultiheadAttention(768, 4, batch_first=True, dropout=0.2)
            self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)
        else:
            # 4 4 2 2 dropout 0.2
            self.attention = nn.MultiheadAttention(768, 8, dropout=0.2)  # 4头注意力
            self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层
            # 方法3的权重计算方法的权重矩阵。
            self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768
            # 计算损失函数的权重
            self.diversity_loss_weight = diversity_loss_weight
            self.l2_reg_weight = l2_reg_weight
            self.final_LayerNorm = nn.LayerNorm(768)
            # 接下来是关于图像嵌入与文本嵌入的部分：
            self.image_cross_attention = nn.MultiheadAttention(768, 8, dropout=0.2)  # 4头注意力
            self.gate_proj_image = nn.Linear(768 * 2, 1)
            # 名词增强部分
            self.nouns_cross_attention = nn.MultiheadAttention(768, 2, batch_first=True, dropout=0.1)
            self.gate_proj_nouns = nn.Linear(768 * 2, 1)

            self.nouns_cross_attention_image = nn.MultiheadAttention(768, 2, batch_first=True, dropout=0.1)
            self.gate_proj_nouns_image = nn.Linear(768 * 2, 1)

    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                sentence_mask, image_mask, noun_embeds, noun_mask, image_caption_valid, image_caption_mask,
                score):
        # 增加部分3、利用原文的index指示器，我们只使用文本模态嵌入信息计算Prompt
        # import ipdb; ipdb.set_trace()
        # ---------------- 新增部分：
        mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]
        mask_expanded_encoder = mask_expanded.repeat(1, 1, encoder_outputs.size(2)).to(attention_mask.device)
        # 使用广播机制提取文本特征 [b,s,768]
        text_embeddings = encoder_outputs * mask_expanded.float()

        # --- 新增：图像嵌入交叉注意力计算 ---
        # 现在利用相关度对 图像信息进行了修正， 其实可以把这个图像融合也放回来试试效果
        image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        image_embeddings = encoder_outputs * image_mask_expanded_for_image.float()  # 提取图像嵌入
        # 与图像字幕一起，如果有字幕用字幕， 没字幕用图像特征。 后续的图像特征 全部以sample_caption_embedding代替使用，

        caption_mask_expand = image_caption_mask.unsqueeze(-1).to(attention_mask.device)  # [b, s, 1]
        image_caption_valid_expand = image_caption_valid.unsqueeze(-1).unsqueeze(-1).to(
            attention_mask.device)  # [b, 1, 1]
        caption = encoder_outputs * caption_mask_expand.float()
        sample_caption_embedding = caption * image_caption_valid_expand + image_embeddings * (
                    1 - image_caption_valid_expand)
        sample_caption_mask = caption_mask_expand * image_caption_valid_expand + image_mask_expanded_for_image * (
                    1 - image_caption_valid_expand)
        sample_caption_embedding = sample_caption_embedding.to(attention_mask.device)
        sample_caption_mask = sample_caption_mask.to(attention_mask.device)
        sample_caption_mask = ~sample_caption_mask.squeeze(-1).bool()

        # 6.1 不做这个名词集合增强
        # ---------------------------------
        # 名词集合增强部分
        # 再做任何操作前，先和名词做一个交叉注意力计算，让文本嵌入与更加关注特定的Aspect，也就是我们的目标
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
        # # text_emb = text_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]
        #
        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     text_embeddings[mask_expanded_encoder]
        # )
        # 再对图像或者说字幕特征 做一个名词集合的修正
        # nouns_attn_output_image, attn_weights = self.nouns_cross_attention_image(
        #     query=image_embeddings,
        #     key=noun_embeds,
        #     value=noun_embeds,
        #     key_padding_mask=noun_mask,  # True 表示需要被掩码的位置
        #     need_weights=True  # 返回注意力权重以便调试
        # )

        # nouns_attn_output_image, attn_weights = self.nouns_cross_attention_image(
        #     query=sample_caption_embedding,
        #     key=noun_embeds,
        #     value=noun_embeds,
        #     key_padding_mask=noun_mask,  # True 表示需要被掩码的位置
        #     need_weights=True  # 返回注意力权重以便调试
        # )
        # gate_nouns_image = torch.sigmoid(
        #     self.gate_proj_nouns_image(torch.cat([image_embeddings, nouns_attn_output_image], dim=-1)))
        # image_mask_expanded_for_image = image_mask_expanded_for_image.repeat(1, 1, encoder_outputs.size(2)).to(
        #     attention_mask.device)
        # caption_mask_expand = caption_mask_expand.repeat(1, 1, encoder_outputs.size(2)).to(attention_mask.device)
        #
        # updated_outputs = encoder_outputs.clone()
        # # 计算完毕在重新分配回去的部分
        # for i in range(image_embeddings.size(0)):
        #     if image_caption_valid[i] == 1:  # 字幕信息可用
        #         feature_mask = caption_mask_expand[i]
        #         feature = sample_caption_embedding[i]
        #         # print("wdd", feature.size(), nouns_attn_output_image.size())
        #         feature = gate_nouns_image[i] * feature + (1 - gate_nouns_image[i]) * nouns_attn_output_image[i]
        #         # print("ww", feature.size(), feature_mask.size())
        #         updated_outputs[i] = updated_outputs[i].masked_scatter(
        #             feature_mask.bool(),
        #             feature[feature_mask]
        #         )
        #     else:
        #         feature_mask = image_mask_expanded_for_image[i]
        #         feature = sample_caption_embedding[i]
        #         feature = gate_nouns_image[i] * feature + (1 - gate_nouns_image[i]) * nouns_attn_output_image[i]
        #         updated_outputs[i] = updated_outputs[i].masked_scatter(
        #             feature_mask.bool(),
        #             feature[feature_mask]
        #         )
        #
        # # image_embeddings = gate_nouns_image * image_embeddings + (1 - gate_nouns_image) * nouns_attn_output_image
        # #
        # # #  把计算结果先放回去，
        # # encoder_outputs = encoder_outputs.masked_scatter(
        # #     image_mask_expanded_for_image,
        # #     image_embeddings[image_mask_expanded_for_image]
        # # )
        # encoder_outputs = updated_outputs
        # -------------------------------------------

        image_embeddings_new = encoder_outputs * image_mask_expanded_for_image.float()  # 提取图像嵌入
        image_emb = image_embeddings_new.permute(1, 0, 2).to(attention_mask.device)  # [s, b, 768]  图像嵌入，准备作为 Key/Value
        text_embeddings_new = encoder_outputs * mask_expanded.float()
        text_emb = text_embeddings_new.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]
        caption = encoder_outputs * caption_mask_expand.float()
        sample_caption_embedding = caption * image_caption_valid_expand + image_embeddings * (
                1 - image_caption_valid_expand).to(attention_mask.device)
        sample_caption_embedding_att = sample_caption_embedding.permute(1, 0, 2).to(attention_mask.device)

        # -----------------------------------
        # 添加的融合信息部分：
        # 扩展prompt池到batch维度 [prompt_num, b, 768]

        prompt_pool = self.prompt_pool.unsqueeze(1).expand(-1, text_emb.size(1), -1).to(attention_mask.device)
        # print("text 维度", text_emb.size())
        # 注意力计算 [s,b,768]
        attn_output, _ = self.attention(
            query=text_emb,
            key=prompt_pool,
            value=prompt_pool,
            key_padding_mask=None
        )
        # Transformer型的处理
        final_features = self.layer_norm1(text_emb + self.dropout1(attn_output))
        ffn_output = self.ffn(final_features)
        final_features = self.layer_norm2(final_features + self.dropout2(ffn_output))
        final_features = final_features.permute(1, 0, 2)
        # MLP处理 ------------------------------------------------------
        # mlp_output = self.prompt_mlp(attn_output.permute(1, 0, 2)).to(attention_mask.device)  # [b,s,768]
        # text_mlp = self.text_mlp(text_embeddings).to(attention_mask.device)  # [b,s,768]
        # # print("计算权重前两者维度", mlp_output.shape, text_mlp.shape)
        # # 权重计算与融合 -------------------------------------------------
        # # weights = self.compute_correlation_weights_learnable(text_mlp, mlp_output)  # [b, n, n]
        #
        # weights = self.compute_correlation_weights_learnable(text_mlp, mlp_output)  # [b, n, n]
        # # 加权融合原始文本特征
        # # print("两种维度大小", weights.shape, text_embeddings.shape)
        # # prompt_text = attn_output.permute(1, 0, 2)
        #
        # enhanced_text = torch.matmul(weights, text_embeddings)  # [b,s,768] 好像之前成这个结果也很高呢。
        # # enhanced_text = torch.matmul(weights, mlp_output)
        # enhanced_text = enhanced_text.to(attention_mask.device)
        # # 融合Prompt信息和原文的文本嵌入表示
        # # 在forward方法中添加：
        # gate = torch.sigmoid(self.gate_proj(torch.cat([text_embeddings, enhanced_text], dim=-1)))
        #
        # # 方式1： 直接将门控机制运用在原始和Prompt上
        # # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # # 调整一下需不需要保证原始信息居多的情况
        #
        # final_features = gate * text_embeddings + (1 - gate) * (
        #             text_embeddings + enhanced_text)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控

        # 方式3: 直接相加
        # final_features = text_embeddings + enhanced_text

        final_features = final_features.to(attention_mask.device)
        # 再把信息放回text_embedding的部分。
        text_emb_for_image_attn = text_embeddings.permute(1, 0, 2).to(
            attention_mask.device)  # 使用 Prompt 增强后的文本表示 final_features 作为 Query (需要调整维度顺序)

        # ----------- 再增加与图像嵌入的修正部分
        # 注意力计算 [s,b,768]
        # image_cross_attn_output, _ = self.image_cross_attention(  # 图像交叉注意力计算
        #     query=final_features.permute(1, 0, 2).to(attention_mask.device),  # Query: Prompt 增强后的文本表示
        #     key=image_emb,  # Key: 图像嵌入
        #     value=image_emb,  # Value: 图像嵌入
        #     key_padding_mask=None  # 图像部分是否需要 padding mask？ 根据实际情况添加
        # )  # image_cross_attn_output: [s, b, 768]  图像交叉注意力输出
        # 用综合的信息来计算交叉注意力。
        # image_cross_attn_output, _ = self.image_cross_attention(  # 图像交叉注意力计算
        #     query=final_features.permute(1, 0, 2).to(attention_mask.device),  # Query: Prompt 增强后的文本表示
        #     key=sample_caption_embedding_att,  # Key: 图像嵌入
        #     value=sample_caption_embedding_att,  # Value: 图像嵌入
        #     key_padding_mask=sample_caption_mask  # 图像部分是否需要 padding mask？ 根据实际情况添加
        # )
        # image_cross_attn_output_result = image_cross_attn_output.permute(1, 0, 2)
        #
        # #
        # gate_image = torch.sigmoid(self.gate_proj_image(torch.cat([final_features, image_cross_attn_output_result],
        #                                                           dim=-1)))
        # # 把融合的特征也放入其中。
        # # 调整一下需不需要保证原始信息居多的情况
        # final_features = gate_image * final_features + (1 - gate_image) * (
        #             final_features + image_cross_attn_output_result)
        #
        # final_features = gate_image * final_features + (1 - gate_image) * (image_cross_attn_output_result)
        # # 将相似度修改到这里的融合阶段来用相关度调整增强部分的贡献 (更温和)
        # # 如果相关度低，那么增强部分的贡献就小
        # # final_features = final_features + score * (final_features - image_cross_attn_output_result)
        # # 也增加残差链接和归一化
        # # final_features = final_features + text_embeddings
        # # final_features = self.final_LayerNorm(final_features)
        #
        # # print("用于放回encoder结果的维度", mask_expanded.repeat(1, 1, final_features.size(2)).shape)
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, final_features.size(2)).to(attention_mask.device)
        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     final_features[mask_expanded_encoder]
        # )
        # -----------------
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,  # 情感提示模板的token索引（如特殊占位符序列）
            # encoder_hidden_states=encoder_outputs,  # 多模态编码器的综合输出
            encoder_hidden_states=final_features,
            encoder_padding_mask=attention_mask.eq(0),  # 将padding位置设为True
            decoder_padding_mask=decoder_attention_mask.eq(0),  # 过滤padding位置
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        senti_prompt_logits = self.senti_prompt_linear(prompt_logits)

        # --- 新增：计算和返回损失函数 并将损失直接返回---
        # diversity_loss = self.diversity_loss_cosine_distance()
        # l2_reg_loss = self.l2_regularization_loss()
        # # orthogonal_loss = self.orthogonal_regularization()
        # l2_reg_loss = l2_reg_loss
        l2_reg_loss = diversity_loss = torch.tensor(0.0, dtype=torch.float)
        return senti_prompt_logits, diversity_loss, l2_reg_loss

    def compute_correlation_weights(self, tensor1, tensor2):
        # 将Prompt和文本特征进行一个矩阵乘法 只是参考论文的实现权重的放大

        interaction_scores = torch.einsum('bsd,bpd->bsp',
                                          tensor1,
                                          tensor2)
        # Sigmoid激活
        sigmoid_scores = torch.sigmoid(interaction_scores)  # [b, n, n]

        # 动态归一化 [b, n, 1]
        s_min = sigmoid_scores.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        s_max = sigmoid_scores.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        epsilon = 1e-8  # 防止除零

        # 归一化公式 [n, n]
        weights = (sigmoid_scores - s_min) / (s_max - s_min + epsilon)

        return weights

    # 第二种方法是常规的余弦相似度计算

    def compute_correlation_weights_cosine(self, tensor1, tensor2):
        """使用余弦相似度计算权重"""
        # 归一化张量以计算余弦相似度
        tensor1_normalized = F.normalize(tensor1, p=2, dim=-1)  # [b, s, d]
        tensor2_normalized = F.normalize(tensor2, p=2, dim=-1)  # [b, p, d]

        # 计算余弦相似度矩阵 [b, s, p] (或 bsp -> bsn 如果p和n一致)
        similarity_scores = torch.matmul(tensor1_normalized, tensor2_normalized.transpose(1, 2))

        # 可以选择是否使用 Sigmoid 或其他激活函数，或者直接使用相似度作为权重
        weights = torch.sigmoid(similarity_scores)  # 例如使用 Sigmoid 激活

        return weights

    # 第三种方法则是使用可学习的权重矩阵，不直接计算相似度，而是让模型自已学习
    def compute_correlation_weights_learnable(self, tensor1, tensor2):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          self.learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活
        # 可以使用低秩分解的方法辅助实现
        # projected_tensor1 = torch.matmul(tensor1, self.learnable_weight_matrix_A)  # [b,s,r]
        # projected_tensor2 = torch.matmul(tensor2, self.learnable_weight_matrix_A)  # [b,p,r]
        #
        # # 交互计算
        # interaction_scores = torch.einsum('bsr,bpr->bsp',
        #                                   projected_tensor1,
        #                                   projected_tensor2)  # [b,s,p]
        # weights = torch.sigmoid(interaction_scores)
        # weights = interaction_scores
        return weights

    # 计算损失的两个部分，
    def diversity_loss_cosine_distance(self):
        """计算 Prompt 池的余弦距离多样性损失"""
        prompt_pool = self.prompt_pool  # 直接使用 self.prompt_pool
        num_prompts = prompt_pool.size(0)
        if num_prompts <= 1:
            return torch.tensor(0.0, device=prompt_pool.device)

        prompt_pool_normalized = F.normalize(prompt_pool, p=2, dim=1)
        similarity_matrix = torch.matmul(prompt_pool_normalized, prompt_pool_normalized.transpose(0, 1))
        mask = 1 - torch.eye(num_prompts, device=prompt_pool.device)
        masked_similarity_matrix = similarity_matrix * mask
        diversity_loss = masked_similarity_matrix.sum() / (num_prompts * (num_prompts - 1) + 1e-8)
        return diversity_loss * self.diversity_loss_weight  # 应用权重

    def l2_regularization_loss(self):
        """计算 Prompt 池的 L2 正则化损失"""
        l2_reg_loss_prompt = torch.sum(self.prompt_pool ** 2)  # 计算 Prompt 池参数的平方和
        # l2_reg_loss = (
        #         torch.norm(self.learnable_weight_matrix_A, p=2) +
        #         torch.norm(self.learnable_weight_matrix_B, p=2)
        # )
        return (l2_reg_loss_prompt) * self.l2_reg_weight  # 应用权重

    def orthogonal_regularization(self):
        """正交约束项"""
        A, B = self.learnable_weight_matrix_A, self.learnable_weight_matrix_B
        orth_loss = torch.norm(torch.mm(A.T, A) - torch.eye(self.rank, device=A.device)) + \
                    torch.norm(torch.mm(B, B.T) - torch.eye(self.rank, device=B.device))
        return self.ortho_lambda * orth_loss


class MultiModalBartDecoder_aspects_num(nn.Module):  # MSP task
    def __init__(self,
                 config: MultiModalBartConfig,
                 decoder,
                 max_aspects_nums=5):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.dropout_layer = nn.Dropout(0.1)
        self.aspects_num_head = BartClassificationHead(config.d_model,
                                                       config.d_model, max_aspects_nums,
                                                       config.classif_dropout)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, aspects_num_labels, encoder_outputs, attention_mask,
                aspects_num_decoder_input_ids):
        decoder_outputs = self.decoder(
            input_ids=aspects_num_decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=None,
            decoder_causal_mask=None,
        )

        # predict_aspects_num = F.linear(
        #     decoder_outputs[0][:, 1],
        #     self.dropout_layer(self.decoder.embed_tokens.
        #                        weight[self.aspects_num_ids[0]:self.aspects_num_ids[2] +
        #                               1]))  # bsz
        # predict_aspects_num = torch.flip(predict_aspects_num, dims=[-1])
        predict_aspects_num_logits = self.aspects_num_head(decoder_outputs[0][:, 1])
        loss_fct = nn.CrossEntropyLoss()
        aspects_num_loss = loss_fct(predict_aspects_num_logits, aspects_num_labels)
        return aspects_num_loss, predict_aspects_num_logits
