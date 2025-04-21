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
        with safe_open(local_checkpoint_path, framework="pt", device="cpu") as f: #  使用 safe_open 加载 state_dict
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        #  不需要像 nf_resnet50 那样过滤 'head.fc'，因为 vit_base_patch32_224 通常没有 head.fc 这样的分类头
        image_encoder.load_state_dict(state_dict, strict=False) #  加载 state_dict 到模型
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
                self.image_encoder.eval()
        # 转换部分 把图像嵌入换成文本嵌入的部分
        self.proj_image_features = nn.Linear(
            in_features=self.d_image_encoder,
            out_features=num_image_tokens * final_dim,
        )
        # 这里也可以换成一个MLP的  但是实验发现还是简单一点好。
        self.proj_image_features_MLP = nn.Sequential(
            nn.Linear(self.d_image_encoder, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_image_tokens * final_dim),
        )

    def forward(self, image_pixel_values):
        # import ipdb; ipdb.set_trace()
        image_pixel_values = torch.stack(image_pixel_values)
        batch_size = image_pixel_values.size(0)
        # image_encoder 采用nf_resnet50 保持原论文方法不变，
        # 原始图像像素 (batch_size, 3, 224, 224) →  (batch_size * 3, 224, 224)
        # → (batch_size, num_image_tokens*final_dim)
        image_features = encode_images(image_encoder=self.image_encoder,
                                       proj_image_features=self.proj_image_features,  # 这里的转换部分也可以有修改
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


class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, config: MultiModalBartConfig, encoder, img_feat_id,
                 cls_token_id, num_image_tokens):
        super().__init__()

        self.img_feat_id = img_feat_id  # 图像特征标识符
        self.cls_token_id = cls_token_id  # CLS标记
        embed_tokens = encoder.embed_tokens  # 复用BART的词嵌入层
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions
        # 其余encoder的 部分包括BART原先的内容，并不做修改。
        self.embed_tokens = embed_tokens
        # 图像特征嵌入模块 至此得到了图像嵌入表示，能在BART形式的文本嵌入表示 这里只是经过线性层换成了BART需要的维度，但是他并没有经过BART的encoder处理
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        # 将文本和图像都变成了嵌入，其中的图像表示会经过变换 变成lt*dt的形式 至此完成公式3 部分，但是CLIPCAP处理图像得到的文本好像没找到在哪里
        mask = (input_ids == self.img_feat_id) | (
                input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()
        # 让每一个batch中样本的图像放到对应的位置上
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
        # 至此的信息是让图像表示经过转换后得到BART的维度之后， 在经过BART其中的Encoder
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos  # 残差的连接是一个位置编码
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
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()

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
                        aspect_linear = nn.Linear(768, 768).to(
                            generated_aspect_prompt.device)  ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        prompt_embedding = aspect_relu(prompt_embedding)
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
                 cls_token_id, num_image_tokens, use_different_senti_prompt):
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
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name, num_image_tokens=num_image_tokens)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()

    def _embed_multi_modal(self, generated_senti_prompt, aspects_num, input_ids, image_features):
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
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    prompt_embedding_list = []
                    # 遍历一个样本中的方面个数
                    for j in range(aspect_num):
                        # 为每一个生成的情绪Prompt做一个线性变换 借助图像表示编码解码后的结果，得到情绪相关的embedding的结果
                        # BART base 512 Large 1024
                        aspect_linear = nn.Linear(768, 768).to(generated_senti_prompt.device)
                        aspect_relu = nn.LeakyReLU().to(generated_senti_prompt.device)
                        prompt_embedding = aspect_linear(generated_senti_prompt[index])
                        prompt_embedding = aspect_relu(prompt_embedding)
                        ###可以加入激活函数å
                        # prompt_embedding = nn.LeakyReLU(prompt_embedding)
                        prompt_embedding_list.append(prompt_embedding)
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
        self.embed_images = ImageEmbedding(embed_dim, embed_dim, image_encoder_type, image_model_name, num_image_tokens=num_image_tokens)
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
        return logits

    def decode(self, tokens, state, only_sc=False):
        return self(tokens, state, only_sc)[:, -1]


class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss_fct = nn.CrossEntropyLoss()
        self.fc = nn.LogSoftmax(dim=-1)

    def forward(self, tgt_tokens, pred, mask):
        '''
        tgt_tokens: (2 (batch-size), 12 (max_len+1))
        pred: (2, 12, 40 (max_word_len))
        '''

        tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        output = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))  ##每一个词都有12种类别， input= (40, 12)
        return output


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

    def __init__(self, config: MultiModalBartConfig, decoder, prompt_pool_num, diversity_loss_weight, l2_reg_weight):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.senti_prompt_linear = nn.Linear(768, 768)
        # 添加部分：
        # 1、情绪Prompt池部分。这个维度保持与768一致
        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_num, 768))  # 假设最大5个aspect
        # 2、用于修正Prompt的转换器MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        # 用于将Prompt也转换的MLP
        self.prompt_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        # 注意力的头数也是一个可以修改的指标
        self.attention = nn.MultiheadAttention(768, 4)  # 4头注意力
        self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层
        # 方法3的权重计算方法的权重矩阵。
        self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768

        # 计算损失函数的权重
        self.diversity_loss_weight = diversity_loss_weight
        self.l2_reg_weight = l2_reg_weight

        # 接下来是关于图像嵌入与文本嵌入的部分：
        # self.image_cross_attention = nn.MultiheadAttention(768, 4)  # 4头注意力
        # self.gate_proj_image = nn.Linear(768 * 2, 1)

    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                sentence_mask, image_mask):
        # 增加部分3、利用原文的index指示器，我们只使用文本模态嵌入信息计算Prompt
        # import ipdb; ipdb.set_trace()
        # ---------------- 新增部分：
        mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]

        # 使用广播机制提取文本特征 [b,s,768]
        text_embeddings = encoder_outputs * mask_expanded.float()
        text_emb = text_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]

        # --- 新增：图像嵌入交叉注意力计算 ---
        image_mask_expanded_for_image = image_mask.unsqueeze(-1).to(attention_mask.device)  # 为图像 mask 扩展维度
        image_embeddings = encoder_outputs * image_mask_expanded_for_image.float()  # 提取图像嵌入
        image_emb = image_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s, b, 768]  图像嵌入，准备作为 Key/Value

        # 扩展prompt池到batch维度 [prompt_num, b, 768]
        prompt_pool = self.prompt_pool.unsqueeze(1).expand(-1, text_emb.size(1), -1).to(attention_mask.device)

        # 注意力计算 [s,b,768]
        attn_output, _ = self.attention(
            query=text_emb,
            key=prompt_pool,
            value=prompt_pool,
            key_padding_mask=None
        )

        # MLP处理 ------------------------------------------------------
        mlp_output = self.prompt_mlp(attn_output.permute(1, 0, 2)).to(attention_mask.device)  # [b,s,768]
        text_mlp = self.text_mlp(text_embeddings).to(attention_mask.device)  # [b,s,768]
        # print("计算权重前两者维度", mlp_output.shape, text_mlp.shape)
        # 权重计算与融合 -------------------------------------------------
        weights = self.compute_correlation_weights_learnable(text_mlp, mlp_output)  # [b, n, n]
        # 加权融合原始文本特征
        # print("两种维度大小", weights.shape, text_embeddings.shape)
        # prompt_text = attn_output.permute(1, 0, 2)

        enhanced_text = torch.matmul(weights, text_embeddings)  # [b,s,768] 好像之前成这个结果也很高呢。
        # enhanced_text = torch.matmul(weights, prompt_text)
        enhanced_text = enhanced_text.to(attention_mask.device)
        # 融合Prompt信息和原文的文本嵌入表示
        # 在forward方法中添加：
        gate = torch.sigmoid(self.gate_proj(torch.cat([text_embeddings, enhanced_text], dim=-1)))

        # 方式1： 直接将门控机制运用在原始和Prompt上
        # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        final_features = gate * text_embeddings + (1 - gate) * (
                text_embeddings + enhanced_text)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控

        final_features = final_features.to(attention_mask.device)
        # 再把信息放回text_embedding的部分。
        # text_emb_for_image_attn = final_features.permute(1, 0, 2).to(
        #     attention_mask.device)  # 使用 Prompt 增强后的文本表示 final_features 作为 Query (需要调整维度顺序)

        #----------- 再增加与图像嵌入的修正部分
        # 注意力计算 [s,b,768]
        # image_cross_attn_output, _ = self.image_cross_attention(  # 图像交叉注意力计算
        #     query=text_emb_for_image_attn,  # Query: Prompt 增强后的文本表示
        #     key=image_emb,  # Key: 图像嵌入
        #     value=image_emb,  # Value: 图像嵌入
        #     key_padding_mask=None  # 图像部分是否需要 padding mask？ 根据实际情况添加
        # )  # image_cross_attn_output: [s, b, 768]  图像交叉注意力输出

        # image_cross_attn_output_result = image_cross_attn_output.permute(1, 0, 2)
        # gate_image = torch.sigmoid(self.gate_proj_image(torch.cat([final_features, image_cross_attn_output_result],
        #                                                 dim=-1)))
        # 把融合的特征也放入其中。
        # final_features = gate_image * final_features + (1 - gate) * (final_features + image_cross_attn_output_result)

        # print("用于放回encoder结果的维度", mask_expanded.repeat(1, 1, final_features.size(2)).shape)
        mask_expanded_encoder = mask_expanded.repeat(1, 1, final_features.size(2)).to(attention_mask.device)
        encoder_outputs = encoder_outputs.masked_scatter(
            mask_expanded_encoder,
            final_features[mask_expanded_encoder]
        )
        # -----------------
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,  # 情感提示模板的token索引（如特殊占位符序列）
            encoder_hidden_states=encoder_outputs,  # 多模态编码器的综合输出
            encoder_padding_mask=attention_mask.eq(0),  # 将padding位置设为True
            decoder_padding_mask=decoder_attention_mask.eq(0),  # 过滤padding位置
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        senti_prompt_logits = self.senti_prompt_linear(prompt_logits)

        # --- 新增：计算和返回损失函数 并将损失直接返回---
        diversity_loss = self.diversity_loss_cosine_distance()
        l2_reg_loss = self.l2_regularization_loss()

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
