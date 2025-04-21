import os
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
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

from transformers import AutoConfig, AutoModel, CLIPVisionModel, CLIPVisionConfig
import timm
from src.model.attention import Attention_for_Senti_Prompt

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
    d_image_encoder = _d_image_encoder(image_encoder_type, image_model_name, image_encoder)

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
    # print("输入形状:", pixel_values.shape)  # 应为[batch,3,224,224]
    # print("数据类型:", pixel_values.dtype)  # 应为float32
    # print("数值范围:", pixel_values.min(), pixel_values.max())  # 应在合理范围内

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
    def __init__(self, image_dim, final_dim, image_encoder_type, image_model_name, frozen_image_encoder=False,
                 num_image_tokens=2):
        super(ImageEmbedding, self).__init__()
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
        # 这里也可以换成一个MLP的
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
                 cls_token_id, num_image_tokens, use_different_aspect_prompt, aspect_prompt_token_front_id, aspect_prompt_token_end_id):
        super().__init__()

        self.use_generated_prompt = use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.aspect_prompt_token_front_id = aspect_prompt_token_front_id
        self.aspect_prompt_token_end_id = aspect_prompt_token_end_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_aspect_prompt = use_different_aspect_prompt

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop
        self.num_image_tokens = num_image_tokens

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

        new_input_ids = []
        # import ipdb; ipdb.set_trace()
        for i in range(len(aspects_num)):

            aspect_num = aspects_num[i]
            # print('the aspect_num is {}'.format(aspect_num))

            input_id = input_ids[i]
            # 相应的，aspect_num 数量的增加，让 end_index 全部+3 因为多了两个索引token和一个sep token
            if self.num_image_tokens == 0:
                prompt_begin_index = 25
                # prompt_end_index = 39
                prompt_end_index = 42
            elif self.num_image_tokens == 1:
                prompt_begin_index = 26
                # prompt_end_index = 40
                prompt_end_index = 43
            elif self.num_image_tokens == 2:
                prompt_begin_index = 27
                # prompt_end_index = 41
                prompt_end_index = 44
            elif self.num_image_tokens == 3:
                prompt_begin_index = 28
                # prompt_end_index = 42
                prompt_end_index = 45
            elif self.num_image_tokens == 4:
                prompt_begin_index = 29
                # prompt_end_index = 43
                prompt_end_index = 46
            elif self.num_image_tokens == 5:
                prompt_begin_index = 30
                # prompt_end_index = 44
                prompt_end_index = 47
            elif self.num_image_tokens == 6:
                prompt_begin_index = 31
                # prompt_end_index = 45
                prompt_end_index = 48
            elif self.num_image_tokens == 7:
                prompt_begin_index = 32
                # prompt_end_index = 46
                prompt_end_index = 49

            # print('before')
            # print(len(input_id))
            # import ipdb; ipdb.set_trace()
            reserve_aspect_id = input_id[prompt_begin_index:prompt_begin_index + 3 * aspect_num]
            # 同样修改 把 5 -> 6 因为这里的部分就是因为 aspect的数量预测，导致这里的索引token个数产生了不同
            # 对于不足6个的部分，需要用1补足这些部分、
            if aspect_num == 6:
                # print('aspect_num is 5')
                # print(reserve_aspect_id)
                new_input_id = torch.cat(
                    [input_id[:prompt_begin_index], reserve_aspect_id, input_id[prompt_end_index + 1:]])
            else:
                cut_aspect_id = torch.ones_like(input_id[prompt_begin_index + 3 * aspect_num:prompt_end_index])
                new_input_id = torch.cat(
                    [input_id[:prompt_begin_index], reserve_aspect_id, cut_aspect_id, input_id[prompt_end_index:]])
            # print("++++++++++++++++++++cut_aspect_id++++++++++++++++++++++++")
            # print(cut_aspect_id)
            # print(input_id[58:])
            new_input_ids.append(new_input_id)
            # print('the aspect num is {}'.format(aspect_num))
            # print('the shape of new_input_id is {}'.format(new_input_id.shape))
            # print(new_input_id[58:])
            # print("+++++++++++++++++++++++input_id length is {}+++++++++++++++++++++++".format(len(input_id)))
            # print(input_id)
            # print("+++++++++++++++++++++++new_input_id length is {}+++++++++++++++++++++++".format(len(input_id)))
            # print(new_input_id)
        new_input_ids = torch.stack(new_input_ids)
        # 多出指示Aspect前后的索引位置
        prompt_mask = aspect_prompt_mask = (new_input_ids == self.aspect_prompt_token_id) | (new_input_ids == self.aspect_prompt_token_front_id)| (new_input_ids == self.aspect_prompt_token_end_id)
        ##[29:58]: 一共5组:[50288, 50288,     9, 50289,  5702, 50284,]
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
                        # 新增部分：
                        aspect_dropout = nn.Dropout(0.2).to(
                            generated_aspect_prompt.device)
                        aspect_linear_2 = nn.Linear(768, 768).to(
                            generated_aspect_prompt.device)

                        prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        prompt_embedding = aspect_relu(prompt_embedding)

                        # 新增部分：
                        prompt_embedding = aspect_dropout(prompt_embedding)
                        prompt_embedding = aspect_linear_2(prompt_embedding)

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

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, use_generated_prompt,
                 config: MultiModalBartConfig, encoder, img_feat_id, aspect_prompt_token_id, senti_prompt_token_id,
                 cls_token_id, num_image_tokens, use_different_senti_prompt):
        super().__init__()

        self.use_generated_prompt = use_generated_prompt
        self.aspect_prompt_token_id = aspect_prompt_token_id
        self.senti_prompt_token_id = senti_prompt_token_id
        self.use_different_senti_prompt = use_different_senti_prompt

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

        # self.aspect_linear = nn.Linear(768, 768)
        # self.aspect_relu = nn.LeakyReLU()

    def _embed_multi_modal(self, generated_senti_prompt, aspects_num, input_ids, image_features):
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

        if self.use_generated_prompt:
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
                 NEU_id, POS_id, NEG_id, aspect_prompt_token_front_id, aspect_prompt_token_end_id
            ):
        super().__init__()

        self.use_generated_aspect_prompt = use_generated_aspect_prompt
        self.use_generated_senti_prompt = use_generated_senti_prompt
        self.aspect_prompt_token_front_id = aspect_prompt_token_front_id
        self.aspect_prompt_token_end_id = aspect_prompt_token_end_id
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
        self.num_image_tokens = num_image_tokens

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

        new_input_ids = []
        for i in range(len(aspects_num)):

            aspect_num = aspects_num[i]
            # print('the aspect_num is {}'.format(aspect_num))

            input_id = input_ids[i]
            # 在5个aspect的情况下 Prompt长度为 (5 - 1) * 6 + 5 = 29 现在加到6个就是  (6 - 1) * 6 + 5 = 35
            if self.num_image_tokens == 0:
                prompt_begin_index = 25
                # prompt_end_index = 54
                prompt_end_index = 60
            elif self.num_image_tokens == 1:
                prompt_begin_index = 26
                # prompt_end_index = 55
                prompt_end_index = 61
            elif self.num_image_tokens == 2:
                prompt_begin_index = 27
                # prompt_end_index = 56
                prompt_end_index = 62
            elif self.num_image_tokens == 3:
                prompt_begin_index = 28
                # prompt_end_index = 57
                prompt_end_index = 63
            elif self.num_image_tokens == 4:
                prompt_begin_index = 29
                # prompt_end_index = 58
                prompt_end_index = 64
            elif self.num_image_tokens == 5:
                prompt_begin_index = 30
                # prompt_end_index = 59
                prompt_end_index = 65
            elif self.num_image_tokens == 6:
                prompt_begin_index = 31
                # prompt_end_index = 60
                prompt_end_index = 66
            elif self.num_image_tokens == 7:
                prompt_begin_index = 32
                # prompt_end_index = 61
                prompt_end_index = 67

                # print('before')
            # print(len(input_id))
            # import ipdb; ipdb.set_trace()
            reserve_aspect_id = input_id[prompt_begin_index:prompt_begin_index + 6 * aspect_num]
            # 同理 5->6 做出修改
            if aspect_num == 6:
                # print('aspect_num is 5')
                # print(reserve_aspect_id)
                new_input_id = torch.cat(
                    [input_id[:prompt_begin_index], reserve_aspect_id, input_id[prompt_end_index + 1:]])
            else:
                cut_aspect_id = torch.ones_like(input_id[prompt_begin_index + 6 * aspect_num:prompt_end_index])
                new_input_id = torch.cat(
                    [input_id[:prompt_begin_index], reserve_aspect_id, cut_aspect_id, input_id[prompt_end_index:]])
            # print("++++++++++++++++++++cut_aspect_id++++++++++++++++++++++++")
            # print(cut_aspect_id)
            # print(input_id[58:])
            new_input_ids.append(new_input_id)
            # print('the shape of new_input_id is {}'.format(new_input_id.shape))
            # print(new_input_id[58:])

        new_input_ids = torch.stack(new_input_ids)

        if self.use_generated_aspect_prompt:
            ##aspect_prompt

            # import ipdb; ipdb.set_trace()
            # 多添加了具体指示前后Aspect的 token 同时保留 之前的部分
            # aspect_prompt_mask = (
            #         new_input_ids in [self.aspect_prompt_token_id, self.aspect_prompt_token_front_id
            #                           , self.aspect_prompt_token_end_id])
            ##[29:58]: 一共5组:[50288, 50288,     9, 50289,  5702, 50284,]
            aspect_prompt_mask = (new_input_ids == self.aspect_prompt_token_id) | (new_input_ids == self.aspect_prompt_token_front_id)| (new_input_ids == self.aspect_prompt_token_end_id)
            if self.use_different_aspect_prompt:
                # self.aspect_linear = self.aspect_linear.to(device)
                # self.aspect_relu = self.aspect_relu.to(device)
                for index in range(len(aspects_num)):
                    aspect_num = aspects_num[index]
                    # prompt_embedding_ = generated_prompt[index].repeat(aspect_num, 1)
                    aspect_prompt_embedding_list = []
                    for j in range(aspect_num):
                        # 这样的方法在数据量上的情况下会有问题，本文一开始还是少样本，可能都是这个问题，考虑扩大数据集
                        aspect_linear = nn.Linear(768, 768).to(
                            generated_aspect_prompt.device)  ##每个aspect有自己的变换，为每个aspect设计特定的prompt
                        aspect_relu = nn.LeakyReLU().to(generated_aspect_prompt.device)
                        # 新增部分：
                        # aspect_dropout = nn.Dropout(0.2).to(
                        #     generated_aspect_prompt.device)
                        # aspect_linear_2 = nn.Linear(768, 768).to(
                        #     generated_aspect_prompt.device)

                        aspect_prompt_embedding = aspect_linear(generated_aspect_prompt[index])
                        aspect_prompt_embedding = aspect_relu(aspect_prompt_embedding)
                        # 新增部分：
                        # aspect_prompt_embedding = aspect_dropout(aspect_prompt_embedding)
                        # aspect_prompt_embedding = aspect_linear_2(aspect_prompt_embedding)

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
            senti_prompt_mask = (new_input_ids == self.senti_prompt_token_id)
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
                 num_image_tokens=2,
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
        self.num_image_tokens = num_image_tokens
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
        # print(src_tokens.shape): (2, 34)
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
            if self.num_image_tokens == 0:
                end_index = 62
            elif self.num_image_tokens == 1:
                end_index = 63
            elif self.num_image_tokens == 2:
                end_index = 64
            elif self.num_image_tokens == 3:
                end_index = 65
            elif self.num_image_tokens == 4:
                end_index = 66
            elif self.num_image_tokens == 5:
                end_index = 67
            elif self.num_image_tokens == 6:
                end_index = 68
            elif self.num_image_tokens == 7:
                end_index = 69

            if hasattr(self, 'encoder_mlp') and not only_sc:
                src_outputs = self.encoder_mlp(src_outputs)

            if first is not None:
                mask = first.eq(0)
                src_outputs = src_outputs.gather(
                    index=first.unsqueeze(2).repeat(1, 1,
                                                    src_outputs.size(-1)),
                    dim=1)
            else:
                mask = state.encoder_mask[:, end_index:].eq(0)
                # src_outputs = self.decoder.embed_tokens(src_tokens)
            mask = mask.unsqueeze(1)  ## bsz x 1 x max_word_len: (2, 1, 34)
            input_embed = self.decoder.embed_tokens(
                src_tokens)  # bsz x max_word_len x hidden_size: (2, 34, 768); src_tokens: (2, 34)
            input_embed = self.dropout_layer(input_embed)
            if self.avg_feature:  # 先把feature合并一下
                src_outputs = (src_outputs[:, end_index:] + input_embed) / 2
            word_scores = torch.einsum(
                'blh,bnh->bln', hidden_state,
                src_outputs[:, end_index:])  # bsz x max_len x max_word_len: (2, 12, 34)
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
        # print("decoder_input_ids", decoder_input_ids.shape)
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)

        # print("encoder_outputs", encoder_outputs.shape)
        # print("attention_mask", attention_mask.shape)
        # print("decoder_padding_mask", decoder_padding_mask.shape)
        # print("decoder_causal_mask", causal_mask[:decoder_input_ids.size(1), :decoder_input_ids.size(1)].shape)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=None
            # decoder_causal_mask=causal_mask[:decoder_input_ids.size(1), :decoder_input_ids.size(1)],
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     # encoder_hidden_states=encoder_outputs,
        #     encoder_hidden_states=fused_encoder_outputs,  # 使用融合后的 encoder_outputs
        #     # encoder_hidden_states=final_features,  # 或者使用门控融合的结果。
        #     encoder_padding_mask=attention_mask.eq(0),
        #     decoder_padding_mask=decoder_attention_mask.eq(0),
        #     decoder_causal_mask=None,
        # )
        # 得到decoder预测的结果 形状为[batch_size,seq_len, vocab_size]
        lm_logits = F.linear(decoder_outputs[0][:, 1:],
                             self.decoder.embed_tokens.weight,
                             bias=self.final_logits_bias)
        # print("input_ids", input_ids.shape)
        # print("decoder_output的维度", decoder_outputs[0].shape)
        # print("self.decoder.embed_tokens.weight的大小", self.decoder.embed_tokens.weight)
        lm_loss = 0
        # compute lm loss if labels is given
        if labels is not None:
            labels = labels.clone()  # labels 应该是原始的未被掩码的TokenID序列，
            loss_fct = nn.CrossEntropyLoss()

            # print("Shape of lm_logits (input to CrossEntropyLoss):", lm_logits.shape)  # <-- Print lm_logits shape
            # print("Shape of labels (target for CrossEntropyLoss):", labels.shape)  # <-- Print labels shape
            # print("label", labels)
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


class FrozenLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bart_pretrained_linear_layer):  # 添加 bart_pretrained_linear_layer 参数
        super().__init__(in_features, out_features)
        if bart_pretrained_linear_layer is not None:  # 如果提供了预训练线性层
            self.weight = nn.Parameter(bart_pretrained_linear_layer.weight.data.clone(),
                                       requires_grad=False)  # 复制预训练权重并冻结
            self.bias = nn.Parameter(bart_pretrained_linear_layer.bias.data.clone(), requires_grad=False)  # 复制预训练偏置并冻结
        else:  # 如果未提供预训练线性层，则使用默认随机初始化
            for param in self.parameters():
                param.requires_grad = False  # 冻结参数 (即使是随机初始化，也要冻结)


# class FrozenLinearLayer(nn.Linear): #  继承 nn.Linear
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         for param in self.parameters(): # 遍历所有参数
#             param.requires_grad = False #  设置 requires_grad=False，冻结参数

class SingleHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # 线性变换矩阵
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # 缩放因子
        self.scale = 1 / (hidden_dim ** 0.5)

    def forward(self, query, key_value, mask=None):
        """
        query: [batch, q_len, hid]
        key_value: [batch, kv_len, hid]
        mask: [batch, q_len, kv_len]
        """
        Q = self.query(query)
        K = self.key(key_value)
        V = self.value(key_value)

        # 注意力得分
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 掩码处理
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 上下文向量
        context = torch.matmul(attn_weights, V)

        return context


class EnhancedSingleHeadCrossAttn(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.attn = SingleHeadCrossAttention(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, mask=None):
        residual = query
        context = self.attn(query, key_value, mask)
        output = self.layer_norm(residual + self.dropout(context))
        return output

#  预先定义一个交叉注意力类，后面在APD部分使用
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout,
                                                     batch_first=True)  # batch_first=True

    def forward(self, query, key_value):
        """
        Args:
            query:  [batch_size, seq_len, hidden_size] -  作为 Query 的特征
            key_value: [batch_size, key_value_seq_len, hidden_size] - 作为 Key 和 Value 的特征 (可以和 Query 长度不同)
        Returns:
            attention_output: [batch_size, seq_len, hidden_size] -  交叉注意力计算结果
        """
        attention_output, attention_weight = self.cross_attention(query, key_value,
                                                   key_value)  # Self-attention if query == key_value, else cross-attention
        return attention_output, attention_weight


# 融合这些所有的Prompt组合成一个Prompt
class FusionAttention(nn.Module):  # 将注意力融合模块定义为一个独立的 nn.Module，方便复用和管理
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.linear = nn.Linear(input_dim, input_dim)  # 可选的线性层，用于进一步处理注意力输出

    def forward(self, features):
        """
        Args:
            features: [batch_size, num_features, seq_len, dim] -  假设输入特征形状, num_features 是要融合的特征数量 (L+2)
        Returns:
            fused_feature: [batch_size, seq_len, dim] - 融合后的特征
        """
        # 将特征展平成 [batch_size * seq_len, num_features, dim] 方便 Self-Attention 处理
        reshaped_features = features.transpose(1, 2).reshape(features.size(0) * features.size(2), features.size(1),
                                                             features.size(3))  # [b*seq_len, N, dim]  N=num_features
        attn_output, _ = self.attention(reshaped_features, reshaped_features, reshaped_features)  # Self-Attention
        # attn_output: [b*seq_len, N, dim]
        fused_feature = attn_output.mean(dim=1).reshape(features.size(0), features.size(2),
                                                        features.size(3))  # [b*seq_len, dim] -> [b, seq_len, dim]
        fused_feature = self.linear(fused_feature)  # 可选的线性层
        return fused_feature


class MultiModalBartDecoder_generate_aspect_prompt(nn.Module):
    def __init__(self, config: MultiModalBartConfig, decoder, encoder, prompt_pool_num, diversity_loss_weight, l2_reg_weight):
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.aspect_prompt_linear = nn.Linear(768, 768)
        """
        因为本文的方法是在 few-shot上执行的操作， 每个样本的信息相对集中，关键的 aspect 词语和情感词语可能比较突出 所以反而需要简单的方法
        """
        # 新增部分：
        self.layers = len(encoder.layers)  # BART模型的层数这样获取， 他只有6层。
        # 可以考虑不用多头，防止少样本训练的过拟合。
        self.cross_attention_image = CrossAttentionLayer(hidden_size=768, num_attention_heads=6,
                                                         dropout=0.1)  # 用于和 图像嵌入 做交叉注意力
        self.fusion_attention = FusionAttention(input_dim=768, num_heads=4, dropout=0.1)  # 初始化注意力融合模块
        # 单头的
        # self.cross_attention_image = EnhancedSingleHeadCrossAttn(hidden_dim=768)  # 用于和 图像嵌入 做交叉注意力
        # self.fusion_attention = EnhancedSingleHeadCrossAttn(hidden_dim=768)  # 初始化注意力融合模块

        self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层

        # 挑选合适的encoder层作为需要融合的信息 同样为每一个层定义一个交叉注意力层
        self.indices = [4, 5]

        self.cross_attention_layers = nn.ModuleList([  # 用于和 encoder 各层做交叉注意力
            CrossAttentionLayer(hidden_size=768, num_attention_heads=4, dropout=0.1)
            for _ in range(len(self.indices))  # encoder_layers 是 encoder 的层数
        ])

        # self.cross_attention_layers = nn.ModuleList([  # 用于和 encoder 各层做交叉注意力
        #     EnhancedSingleHeadCrossAttn(hidden_dim=768)
        #     # 假设 hidden_size=768, num_heads=12，可以根据你的config调整
        #     for _ in range(len(self.indices))  # encoder_layers 是 encoder 的层数
        # ])

        # 组合每一个Prompt的结果
        # self.gate_projs = nn.ModuleList(
        #     [nn.Linear(768 * 2, 1) for _ in range(len(self.indices) + 2)])  # 为每个特征来源设置一个门控层

        # 用另一种方式计算权重
        # self.fusion_weight_matrices = nn.ParameterList([
        #     nn.Parameter(torch.randn(768, 768)) for _ in range(len(self.indices) + 2)
        #     创建 nn.Parameter 矩阵，形状 [768, 768]
        # ])

        # 1、情绪Prompt池部分。这个维度保持与768一致
        # self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_num, 768))  # 假设最大5个aspect
        # # 2、用于修正Prompt的转换器MLP
        # self.text_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768)
        # )
        # # 用于将Prompt也转换的MLP
        # self.prompt_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768)
        # )
        # # 注意力的头数也是一个可以修改的指标
        # self.attention = nn.MultiheadAttention(768, 4)  # 4头注意力
        # self.gate_proj_image = nn.Linear(768 * 2, 1)  # 门控融合层
        # self.diversity_loss_weight = diversity_loss_weight
        # self.l2_reg_weight = l2_reg_weight
        # self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768

        # 在每次进行交叉注意力计算 和 Prompt融合的时候， 先把encoder的输出经过冻结参数的MLP处理 它的初始化可以从encoder中抽取特定层，不过需要注意输入输出维度哦~
        # self.frozen_mlp1_layer = FrozenLinearLayer(in_features=768, out_features=3072,
        #                                            bart_pretrained_linear_layer=encoder.layers[
        #                                                5].fc1)  # 假设使用 BART encoder 第 5 层的 fc1 层参数
        # self.frozen_mlp1_layer_2 = FrozenLinearLayer(in_features=3072, out_features=768,
        #                                              bart_pretrained_linear_layer=encoder.layers[
        #                                                  5].fc2)  # 假设使用 BART encoder 第 5 层的 fc1 层参数
        # self.mlp1 = nn.Sequential(
        #     self.frozen_mlp1_layer,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     self.frozen_mlp1_layer_2,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     nn.GELU()  # 激活函数
        # )
        #
        # self.frozen_mlp2_layer = FrozenLinearLayer(in_features=768, out_features=3072,
        #                                            bart_pretrained_linear_layer=encoder.layers[
        #                                                5].fc1)  # 假设使用 BART encoder 第 6 层的 fc1 层参数
        # self.frozen_mlp2_layer_2 = FrozenLinearLayer(in_features=3072, out_features=768,
        #                                              bart_pretrained_linear_layer=encoder.layers[
        #                                                  5].fc2)  # 假设使用 BART encoder 第 6 层的 fc1 层参数
        # self.mlp2 = nn.Sequential(
        #     self.frozen_mlp2_layer,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     self.frozen_mlp2_layer_2,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     nn.GELU()  # 激活函数
        # )
        self.mlp1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        self.sparsity_weight = 0.01
        #  使用示例:
        # self.frozen_fc = FrozenLinearLayer(768, 768)  # 创建一个输入和输出维度都为 768 的冻结线性层

    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask, sentence_mask, image_mask, encoder_outputs_all):
        """
        Args:
            encoder_outputs:
            attention_mask:
            decoder_input_ids:
            decoder_attention_mask:
            sentence_mask:  指示了文本嵌入的部分。
            image_mask : 图像表示部分
            encoder_outputs_all: 编码层各层的输出内容 (list of tensors, 每个 tensor 形状 [b, s, hidden_size]) 共六层
        Returns:
        在这里做出修改， 利用encoder各层的信息来得到一组Prompt
        """
        # print("各层编码器信息", len(encoder_outputs_all))
        # mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]

        # 1. 和 Encoder 各层做交叉注意力 (仅限文本部分)
        cross_attention_layer_outputs, sparsity_loss_layers = self.cross_attention_for_layers(encoder_outputs, encoder_outputs_all,
                                                                        sentence_mask)
        # 2. 和 图像嵌入做交叉注意力 (代码不变)
        cross_attention_image_output, sparsity_loss_image = self.cross_attention_for_image(encoder_outputs, encoder_outputs, image_mask,
                                                                      sentence_mask)

        # 3. 注意力融合所有特征
        fused_encoder_outputs = self.fuse_features(encoder_outputs, cross_attention_layer_outputs,
                                                   cross_attention_image_output)

        # 在融合了之后 只修改文本嵌入部分，其他的就不做修改
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, fused_encoder_outputs.size(2)).to(attention_mask.device)

        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     fused_encoder_outputs[mask_expanded_encoder]
        # )
        diversity_loss = l2_reg_loss = 0.0
        # 采用和情绪部分同样的Prompt池方法 --------------------
        # 使用广播机制提取文本特征 [b,s,768]
        # text_embeddings = encoder_outputs * mask_expanded.float()
        # text_emb = text_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]
        #
        # # 扩展prompt池到batch维度 [prompt_num, b, 768]
        # prompt_pool = self.prompt_pool.unsqueeze(1).expand(-1, text_emb.size(1), -1).to(attention_mask.device)
        #
        # # 注意力计算 [s,b,768]
        # attn_output, _ = self.attention(
        #     query=text_emb,
        #     key=prompt_pool,
        #     value=prompt_pool,
        #     key_padding_mask=None
        # )
        #
        # # MLP处理 ------------------------------------------------------
        # mlp_output = self.prompt_mlp(attn_output.permute(1, 0, 2)).to(attention_mask.device)  # [b,s,768]
        # text_mlp = self.text_mlp(text_embeddings).to(attention_mask.device)  # [b,s,768]
        # # print("计算权重前两者维度", mlp_output.shape, text_mlp.shape)
        # # 权重计算与融合 -------------------------------------------------
        # weights = self.compute_correlation_weights_learnable_1(text_mlp, mlp_output)  # [b, n, n]
        # # 加权融合原始文本特征
        # # print("两种维度大小", weights.shape, text_embeddings.shape)
        # enhanced_text = torch.matmul(weights, text_embeddings)  # [b,s,768]
        # enhanced_text = enhanced_text.to(attention_mask.device)
        # # 融合Prompt信息和原文的文本嵌入表示
        # # 在forward方法中添加：
        # gate = torch.sigmoid(self.gate_proj(torch.cat([text_embeddings, enhanced_text], dim=-1)))
        #
        # # 方式1： 直接将门控机制运用在原始和Prompt上
        # # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate * text_embeddings + (1 - gate) * (
        #         text_embeddings + enhanced_text)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # weights_image = self.compute_correlation_weights_learnable_1(text_mlp, cross_attention_image_output)  # [b, n, n]
        # # 加权融合原始文本特征
        # # print("两种维度大小", weights.shape, text_embeddings.shape)
        # enhanced_image = torch.matmul(weights_image, text_embeddings)  # [b,s,768]
        # enhanced_image = enhanced_image.to(attention_mask.device)
        # # 融合Prompt信息和原文的文本嵌入表示
        # # 在forward方法中添加：
        # gate_image = torch.sigmoid(self.gate_proj_image(torch.cat([text_embeddings, enhanced_image], dim=-1)))
        #
        # # 方式1： 直接将门控机制运用在原始和Prompt上
        # # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate_image * final_features + (1 - gate_image) * (
        #         final_features + enhanced_image)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # final_features = final_features.to(attention_mask.device)
        # # 再把信息放回text_embedding的部分。
        # # print("用于放回encoder结果的维度", mask_expanded.repeat(1, 1, final_features.size(2)).shape)
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, final_features.size(2)).to(attention_mask.device)
        #
        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     final_features[mask_expanded_encoder]
        # )

        # 注意力融合效果不太好， 试试 继续保持门控机制融合的方法，

        # 这里可以考虑同样来个门控机制，因为这些信息可能融合过多，参杂了冗余信息，也许需要一部分即可。

        # gate = torch.sigmoid(self.gate_proj(torch.cat([encoder_outputs, fused_encoder_outputs], dim=-1)))

        # 方式1： 直接将门控机制运用在原始和Prompt上
        # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate * encoder_outputs + (1 - gate) * (
        #         encoder_outputs + fused_encoder_outputs)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # final_features = final_features.to(attention_mask.device)

        # import ipdb; ipdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            # encoder_hidden_states=encoder_outputs,
            encoder_hidden_states=fused_encoder_outputs,  # 使用融合后的 encoder_outputs
            # encoder_hidden_states=final_features,  # 或者使用门控融合的结果。
            encoder_padding_mask=attention_mask.eq(0),
            decoder_padding_mask=decoder_attention_mask.eq(0),
            decoder_causal_mask=None,
        )

        prompt_logits = decoder_outputs[0]
        # 修改特征层
        # aspect_prompt_logits = self.text_mlp(prompt_logits)
        #
        aspect_prompt_logits = self.aspect_prompt_linear(prompt_logits)

        # --- 新增：计算和返回损失函数 并将损失直接返回---
        # diversity_loss = self.diversity_loss_cosine_distance()
        # l2_reg_loss = self.l2_regularization_loss()

        return aspect_prompt_logits, sparsity_loss_layers, sparsity_loss_image

    def cross_attention_for_layers(self, query, encoder_outputs_all, sentence_mask):
        """和 Encoder 各层做交叉注意力，仅限文本嵌入部分 (sentence_mask shape: [b, s])"""
        all_layer_outputs = []

        batch_size, seq_len = sentence_mask.shape
        hidden_size = query.size(-1)  # 获取 hidden_size
        # 挑选合适的层也是一个需要判断的依据
        # 再计算交叉注意力前，先对encoder最后的输出套一个MLP
        # print("query维度", query.shape)
        query = self.mlp1(query)
        # print("query维度", query.shape)
        encoder_outputs_all_choose = [encoder_outputs_all[i] for i in self.indices]
        total_loss = 0.0
        for i, encoder_layer_output in enumerate(
                encoder_outputs_all_choose):  # 遍历 encoder 各层输出 这里的encoder会包括最终的输出结果，但是我们只需要encoder每一层的结果。
            batch_cross_attn_outputs = []  # 存储当前层 batch 中每个样本的交叉注意力结果

            for b_idx in range(batch_size):  # 遍历 batch 维度
                current_sentence_mask = sentence_mask[b_idx]  # [s] - 当前 batch 样本的 sentence_mask
                current_query = query[b_idx]  # [s, hidden_size] - 当前 batch 样本的 query
                current_encoder_layer_output = encoder_layer_output[
                    b_idx]  # [s, hidden_size] - 当前 batch 样本的 encoder layer output

                text_mask = current_sentence_mask.bool()  # 转换为 boolean mask [s]
                text_query = current_query[text_mask]  # [num_text_tokens, hidden_size] - 当前样本的文本 query
                encoder_layer_output_text = current_encoder_layer_output[
                    text_mask]  # [num_text_tokens, hidden_size] - 当前样本的 encoder layer text output

                if text_query.numel() == 0:  # 当前样本没有文本
                    cross_attn_output = torch.zeros_like(current_query)  # 用零张量填充当前样本的交叉注意力输出
                else:
                    text_query = text_query.unsqueeze(0)  # [1, num_text_tokens, hidden_size] - 为交叉注意力增加 batch 维度
                    encoder_layer_output_text = encoder_layer_output_text.unsqueeze(
                        0)  # [1, num_text_tokens, hidden_size] - 为交叉注意力增加 batch 维度
                    cross_attn_output_text_part, cross_attn_weight = self.cross_attention_layers[i](text_query,
                                                                                 encoder_layer_output_text)  # [1, num_text_tokens, hidden_size] -  交叉注意力计算结果 (仅文本部分)
                    total_loss = total_loss + torch.mean(torch.abs(cross_attn_weight)) * self.sparsity_weight
                    cross_attn_output = torch.zeros_like(current_query).unsqueeze(
                        0)  # [1, s, hidden_size] - 初始化当前样本的完整交叉注意力输出为零张量
                    cross_attn_output[:, text_mask, :] = cross_attn_output_text_part  # 将计算出的文本部分交叉注意力结果填入完整输出的对应位置

                    cross_attn_output = cross_attn_output.squeeze(0)  # [s, hidden_size] - 移除增加的 batch 维度

                batch_cross_attn_outputs.append(cross_attn_output)  # 将当前样本的交叉注意力结果加入列表

            # 将当前层 batch 中所有样本的交叉注意力结果堆叠起来
            layer_cross_attn_output = torch.stack(batch_cross_attn_outputs, dim=0)  # [b, s, hidden_size]
            all_layer_outputs.append(layer_cross_attn_output)  # 将当前层的交叉注意力结果加入总列表
        total_loss = total_loss / (len(self.indices) * batch_size)
        return all_layer_outputs, total_loss  # 返回一个列表，包含和每一层交叉注意力的结果 (形状和原始query相同，只有文本部分更新)

    def cross_attention_for_image(self, query, encoder_outputs, image_mask, sentence_mask):
        """和 图像嵌入 做交叉注意力 (image_mask shape: [b, s])"""
        batch_size, seq_len = image_mask.shape
        hidden_size = query.size(-1)
        batch_cross_attn_outputs = []

        # 再计算交叉注意力前，先对encoder最后的输出套一个MLP
        query = self.mlp1(query)
        total_loss = 0.0
        for b_idx in range(batch_size):  # 遍历 batch 维度
            current_image_mask = image_mask[b_idx]  # [s] - 当前 batch 样本的 image_mask
            current_sentence_mask = sentence_mask[b_idx]  # [s] - 当前 batch 样本的 sentence_mask
            current_query = query[b_idx]  # [s, hidden_size] - 当前 batch 样本的 query
            current_encoder_outputs = encoder_outputs[b_idx]  # [s, hidden_size] - 当前 batch 样本的 encoder output

            image_mask_bool = current_image_mask.bool()  # 转换为 boolean mask [s]
            image_embedding = current_encoder_outputs[image_mask_bool]  # [num_image_tokens, hidden_size] - 当前样本的图像嵌入

            text_mask = current_sentence_mask.bool()  # 转换为 boolean mask [s]
            text_query = current_query[text_mask]  # [num_text_tokens, hidden_size] - 当前样本的文本 query

            if image_embedding.numel() == 0:  # 当前样本没有图像
                cross_attn_output = torch.zeros_like(current_query)  # 用零张量填充
            else:
                image_embedding = image_embedding.unsqueeze(0)  # [1, num_image_tokens, hidden_size] - 增加 batch 维度
                query_expanded = text_query.unsqueeze(0)  # [1, s, hidden_size] - 增加 batch 维度，query 也需要扩展维度匹配
                cross_attn_output_image_part, cross_attn_weight = self.cross_attention_image(query_expanded,
                                                                          image_embedding)  # [1, s, hidden_size] - 交叉注意力计算 (图像部分)
                total_loss = total_loss + torch.mean(torch.abs(cross_attn_weight)) * self.sparsity_weight
                cross_attn_output = torch.zeros_like(current_query).unsqueeze(0)  # [1, s, hidden_size] - 初始化完整输出
                cross_attn_output[:, text_mask, :] = cross_attn_output_image_part  # 将计算出的图像部分结果填入完整输出
                # 注意，我们是让文本做q，进而去做后续的计算

                cross_attn_output = cross_attn_output.squeeze(0)  # [s, hidden_size] - 移除 batch 维度

            batch_cross_attn_outputs.append(cross_attn_output)

        layer_cross_attn_output = torch.stack(batch_cross_attn_outputs, dim=0)  # [b, s, hidden_size]
        total_loss = total_loss / (1 * batch_size)
        return layer_cross_attn_output, total_loss

    # 尝试不要图片信息，因为经过了转文本，再交叉的学习，
    def fuse_features(self, last_layer_feature, cross_attention_layer_outputs, cross_attention_image_output):
        """注意力融合特征"""
        all_features = cross_attention_layer_outputs + [cross_attention_image_output,
                                                        last_layer_feature]  # 列表，包含所有要融合的特征
        # 将特征列表堆叠成 [L+2, batch_size, seq_len, dim] 的张量
        features_tensor = torch.stack(all_features, dim=0)  # [N, b, seq_len, dim]  N = L+2
        # 调整维度顺序为 [batch_size, N, seq_len, dim] -> [batch_size, N, seq_len, dim]  方便 FusionAttention 模块处理
        features_tensor = features_tensor.transpose(0, 1)  # [b, N, seq_len, dim]

        fused_feature = self.fusion_attention(features_tensor)  # 使用注意力融合模块
        # 门控机制融合。

        # 再计算融合机制前，先对encoder最后的输出套一个MLP
        # last_layer_feature = self.mlp2(last_layer_feature)
        #
        # # 尝试1、
        # all_features = cross_attention_layer_outputs + [cross_attention_image_output, last_layer_feature]
        # #
        # # all_features = cross_attention_layer_outputs + [last_layer_feature]
        # fused_feature = torch.zeros_like(last_layer_feature)
        #
        # fused_feature = 0.7 * last_layer_feature  # 还是保留一定程度的 encoder的 最后输出
        #
        # # for i, feature in enumerate(all_features):
        # #     gate = torch.sigmoid(self.gate_projs[i](torch.cat([last_layer_feature, feature], dim=-1)))  # 每个来源单独计算门控
        # #     fused_feature = fused_feature + gate * feature
        #
        # 尝试2、 使用和情绪计算中使用的加权求和方法
        # for i, feature in enumerate(all_features):
        #     weight = self.compute_correlation_weights_learnable(last_layer_feature, feature,
        #                                                         self.fusion_weight_matrices[i])
        #     fused_feature = fused_feature + torch.matmul(weight, feature)

        return fused_feature

    def compute_correlation_weights_learnable(self, tensor1, tensor2, learnable_weight_matrix):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活

        return weights

    def compute_correlation_weights_learnable_1(self, tensor1, tensor2):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          self.learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活

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

    def forward(self, encoder_outputs, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                sentence_mask):
        # 增加部分3、利用原文的index指示器，我们只使用文本模态嵌入信息计算Prompt
        # import ipdb; ipdb.set_trace()
        # ---------------- 新增部分：
        mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]

        # 使用广播机制提取文本特征 [b,s,768]
        text_embeddings = encoder_outputs * mask_expanded.float()
        text_emb = text_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]

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
        enhanced_text = torch.matmul(weights, text_embeddings)  # [b,s,768]
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
                 encoder, prompt_pool_num, diversity_loss_weight, l2_reg_weight,
                 # max_aspects_nums=5
                 max_aspects_nums=6
                 ):
        # 这里同样需要修改， 17的数据集中 是存在aspect_num=6的数据的
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.dropout_layer = nn.Dropout(0.1)
        self.aspects_num_head = BartClassificationHead(config.d_model,
                                                       config.d_model, max_aspects_nums,
                                                       config.classif_dropout)
        # 这里同样做出与 识别情绪索引 一样的修正Prompt方法，只不过挑选的层需要做出变化
        self.layers = len(encoder.layers)  # BART模型的层数这样获取， 他只有6层。

        self.cross_attention_image = CrossAttentionLayer(hidden_size=768, num_attention_heads=6,
                                                         dropout=0.1)  # 用于和 图像嵌入 做交叉注意力
        self.fusion_attention = FusionAttention(input_dim=768, num_heads=4, dropout=0.1)  # 初始化注意力融合模块

        self.gate_proj = nn.Linear(768 * 2, 1)  # 门控融合层

        # 挑选合适的encoder层作为需要融合的信息  这个时候需要挑选中间层
        self.indices = [5]
        self.cross_attention_layers = nn.ModuleList([  # 用于和 encoder 各层做交叉注意力
            CrossAttentionLayer(hidden_size=768, num_attention_heads=4, dropout=0.1)
            for _ in range(len(self.indices))  # encoder_layers 是 encoder 的层数
        ])
        # 添加部分：
        # 1、情绪Prompt池部分。这个维度保持与768一致
        # self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_num, 768))  # 假设最大5个aspect
        # # 2、用于修正Prompt的转换器MLP
        # self.text_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768)
        # )
        # # 用于将Prompt也转换的MLP
        # self.prompt_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768)
        # )
        # # 注意力的头数也是一个可以修改的指标
        # self.attention = nn.MultiheadAttention(768, 4)  # 4头注意力
        # self.gate_proj_image = nn.Linear(768 * 2, 1)  # 门控融合层
        # self.diversity_loss_weight = diversity_loss_weight
        # self.l2_reg_weight = l2_reg_weight
        # self.learnable_weight_matrix = nn.Parameter(torch.randn(768, 768))  # 假设特征维度是 768
        self.aspects_num_linear = nn.Linear(768, 768)
        self.sparsity_loss_weight = 0.01
        # 采用门控机制加权融合
        # self.gate_projs = nn.ModuleList(
        #     [nn.Linear(768 * 2, 1) for _ in range(len(self.indices) + 2)])  # 为每个特征来源设置一个门控层

        # self.frozen_mlp1_layer = FrozenLinearLayer(in_features=768, out_features=3072,
        #                                            bart_pretrained_linear_layer=encoder.layers[
        #                                                5].fc1)  # 假设使用 BART encoder 第 5 层的 fc1 层参数
        # self.frozen_mlp1_layer_2 = FrozenLinearLayer(in_features=3072, out_features=768,
        #                                              bart_pretrained_linear_layer=encoder.layers[
        #                                                  5].fc2)  # 假设使用 BART encoder 第 5 层的 fc1 层参数
        # self.mlp1 = nn.Sequential(
        #     self.frozen_mlp1_layer,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     self.frozen_mlp1_layer_2,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     nn.GELU()  # 激活函数
        # )
        #
        # self.frozen_mlp2_layer = FrozenLinearLayer(in_features=768, out_features=3072,
        #                                            bart_pretrained_linear_layer=encoder.layers[
        #                                                5].fc1)  # 假设使用 BART encoder 第 6 层的 fc1 层参数
        # self.frozen_mlp2_layer_2 = FrozenLinearLayer(in_features=3072, out_features=768,
        #                                              bart_pretrained_linear_layer=encoder.layers[
        #                                                  5].fc2)  # 假设使用 BART encoder 第 6 层的 fc1 层参数
        # self.mlp2 = nn.Sequential(
        #     self.frozen_mlp2_layer,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     self.frozen_mlp2_layer_2,  # 使用 FrozenLinearLayer，参数已用 BART 预训练参数初始化并冻结
        #     nn.GELU()  # 激活函数
        # )
        self.mlp1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        # 用另一种方式计算权重
        # self.fusion_weight_matrices = nn.ParameterList([
        #     nn.Parameter(torch.randn(768, 768)) for _ in range(len(self.indices) + 2)
            # 创建 nn.Parameter 矩阵，形状 [768, 768]
        # ])


    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, aspects_num_labels, encoder_outputs, attention_mask,
                aspects_num_decoder_input_ids, sentence_mask, image_mask, encoder_outputs_all):

        # mask_expanded = sentence_mask.unsqueeze(-1).to(attention_mask.device)  # [b,s,1]

        # 1. 和 Encoder 各层做交叉注意力 (仅限文本部分)
        cross_attention_layer_outputs, sparsity_loss_layers = self.cross_attention_for_layers(encoder_outputs, encoder_outputs_all,
                                                                        sentence_mask)
        # 2. 和 图像嵌入做交叉注意力 (代码不变)
        cross_attention_image_output, sparsity_loss_image = self.cross_attention_for_image(encoder_outputs, encoder_outputs, image_mask,
                                                                      sentence_mask)

        # 3. 注意力融合所有特征
        fused_encoder_outputs = self.fuse_features(encoder_outputs, cross_attention_layer_outputs,
                                                   cross_attention_image_output)

        # 在融合了之后 只修改文本嵌入部分，其他的就不做修改
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, fused_encoder_outputs.size(2)).to(attention_mask.device)
        #
        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     fused_encoder_outputs[mask_expanded_encoder]
        # )
        diversity_loss = 0.0
        l2_reg_loss = 0.0
        # 采用和情绪部分同样的Prompt池方法 --------------------
        # 使用广播机制提取文本特征 [b,s,768]
        # text_embeddings = encoder_outputs * mask_expanded.float()
        # text_emb = text_embeddings.permute(1, 0, 2).to(attention_mask.device)  # [s,b,768]
        #
        # # 扩展prompt池到batch维度 [prompt_num, b, 768]
        # prompt_pool = self.prompt_pool.unsqueeze(1).expand(-1, text_emb.size(1), -1).to(attention_mask.device)
        #
        # # 注意力计算 [s,b,768]
        # attn_output, _ = self.attention(
        #     query=text_emb,
        #     key=prompt_pool,
        #     value=prompt_pool,
        #     key_padding_mask=None
        # )
        #
        # # MLP处理 ------------------------------------------------------
        # mlp_output = self.prompt_mlp(attn_output.permute(1, 0, 2)).to(attention_mask.device)  # [b,s,768]
        # text_mlp = self.text_mlp(text_embeddings).to(attention_mask.device)  # [b,s,768]
        # # print("计算权重前两者维度", mlp_output.shape, text_mlp.shape)
        # # 权重计算与融合 -------------------------------------------------
        # weights = self.compute_correlation_weights_learnable_1(text_mlp, mlp_output)  # [b, n, n]
        # # 加权融合原始文本特征
        # # print("两种维度大小", weights.shape, text_embeddings.shape)
        # enhanced_text = torch.matmul(weights, text_embeddings)  # [b,s,768]
        # enhanced_text = enhanced_text.to(attention_mask.device)
        # # 融合Prompt信息和原文的文本嵌入表示
        # # 在forward方法中添加：
        # gate = torch.sigmoid(self.gate_proj(torch.cat([text_embeddings, enhanced_text], dim=-1)))
        #
        # # 方式1： 直接将门控机制运用在原始和Prompt上
        # # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate * text_embeddings + (1 - gate) * (
        #         text_embeddings + enhanced_text)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # weights_image = self.compute_correlation_weights_learnable_1(text_mlp,
        #                                                              cross_attention_image_output)  # [b, n, n]
        # # 加权融合原始文本特征
        # # print("两种维度大小", weights.shape, text_embeddings.shape)
        # enhanced_image = torch.matmul(weights_image, text_embeddings)  # [b,s,768]
        # enhanced_image = enhanced_image.to(attention_mask.device)
        # # 融合Prompt信息和原文的文本嵌入表示
        # # 在forward方法中添加：
        # gate_image = torch.sigmoid(self.gate_proj_image(torch.cat([text_embeddings, enhanced_image], dim=-1)))
        #
        # # 方式1： 直接将门控机制运用在原始和Prompt上
        # # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate_image * final_features + (1 - gate_image) * (
        #         final_features + enhanced_image)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # final_features = final_features.to(attention_mask.device)
        # # 再把信息放回text_embedding的部分。
        # # print("用于放回encoder结果的维度", mask_expanded.repeat(1, 1, final_features.size(2)).shape)
        # mask_expanded_encoder = mask_expanded.repeat(1, 1, final_features.size(2)).to(attention_mask.device)
        #
        # encoder_outputs = encoder_outputs.masked_scatter(
        #     mask_expanded_encoder,
        #     final_features[mask_expanded_encoder]
        # )

        # 注意力融合效果不太好， 试试 继续保持门控机制融合的方法，

        # 这里可以考虑同样来个门控机制，因为这些信息可能融合过多，参杂了冗余信息，也许需要一部分即可。

        # gate = torch.sigmoid(self.gate_proj(torch.cat([encoder_outputs, fused_encoder_outputs], dim=-1)))

        # 方式1： 直接将门控机制运用在原始和Prompt上
        # final_features = gate * text_embeddings + (1 - gate) * enhanced_text
        # 方式2: 在运用门控机制的同时，把原始信息放在两个部分，尽可能保证原始信息居多，不会被Prompt过多干扰。
        # final_features = gate * encoder_outputs + (1 - gate) * (
        #         encoder_outputs + fused_encoder_outputs)  # 门控加法，注意这里是 text_embeddings + enhanced_text 的加法结果参与门控
        #
        # final_features = final_features.to(attention_mask.device)

        # --- 新增：计算和返回损失函数 并将损失直接返回---
        # diversity_loss = self.diversity_loss_cosine_distance()
        # l2_reg_loss = self.l2_regularization_loss()

        decoder_outputs = self.decoder(
            input_ids=aspects_num_decoder_input_ids,
            # encoder_hidden_states=encoder_outputs,
            encoder_hidden_states=fused_encoder_outputs,  # 同样做出修改
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
        # 尝试先走一个线性层在做。
        # predict_aspects_num_logits = self.aspects_num_head(decoder_outputs[0][:, 1])

        aspect_num_logits = self.aspects_num_linear(decoder_outputs[0][:, 1])
        predict_aspects_num_logits = self.aspects_num_head(aspect_num_logits)

        loss_fct = nn.CrossEntropyLoss()
        aspects_num_labels = torch.tensor(aspects_num_labels).to(predict_aspects_num_logits.device)
        aspects_num_loss = loss_fct(predict_aspects_num_logits, aspects_num_labels)
        return aspects_num_loss, predict_aspects_num_logits, sparsity_loss_layers, sparsity_loss_image

    def cross_attention_for_layers(self, query, encoder_outputs_all, sentence_mask):
        """和 Encoder 各层做交叉注意力，仅限文本嵌入部分 (sentence_mask shape: [b, s])"""
        all_layer_outputs = []

        batch_size, seq_len = sentence_mask.shape
        hidden_size = query.size(-1)  # 获取 hidden_size
        # 挑选合适的层也是一个需要判断的依据
        query = self.mlp1(query)
        encoder_outputs_all_choose = [encoder_outputs_all[i] for i in self.indices]
        total_loss = 0.0
        for i, encoder_layer_output in enumerate(
                encoder_outputs_all_choose):  # 遍历 encoder 各层输出 这里的encoder会包括最终的输出结果，但是我们只需要encoder每一层的结果。
            batch_cross_attn_outputs = []  # 存储当前层 batch 中每个样本的交叉注意力结果

            for b_idx in range(batch_size):  # 遍历 batch 维度
                current_sentence_mask = sentence_mask[b_idx]  # [s] - 当前 batch 样本的 sentence_mask
                current_query = query[b_idx]  # [s, hidden_size] - 当前 batch 样本的 query
                current_encoder_layer_output = encoder_layer_output[
                    b_idx]  # [s, hidden_size] - 当前 batch 样本的 encoder layer output

                text_mask = current_sentence_mask.bool()  # 转换为 boolean mask [s]
                text_query = current_query[text_mask]  # [num_text_tokens, hidden_size] - 当前样本的文本 query
                encoder_layer_output_text = current_encoder_layer_output[
                    text_mask]  # [num_text_tokens, hidden_size] - 当前样本的 encoder layer text output

                if text_query.numel() == 0:  # 当前样本没有文本
                    cross_attn_output = torch.zeros_like(current_query)  # 用零张量填充当前样本的交叉注意力输出
                else:
                    text_query = text_query.unsqueeze(0)  # [1, num_text_tokens, hidden_size] - 为交叉注意力增加 batch 维度
                    encoder_layer_output_text = encoder_layer_output_text.unsqueeze(
                        0)  # [1, num_text_tokens, hidden_size] - 为交叉注意力增加 batch 维度
                    cross_attn_output_text_part, cross_attn_weight = self.cross_attention_layers[i](text_query,
                                                                                 encoder_layer_output_text)  # [1, num_text_tokens, hidden_size] -  交叉注意力计算结果 (仅文本部分)
                    total_loss = total_loss + torch.mean(torch.abs(cross_attn_weight)) * self.sparsity_loss_weight
                    cross_attn_output = torch.zeros_like(current_query).unsqueeze(
                        0)  # [1, s, hidden_size] - 初始化当前样本的完整交叉注意力输出为零张量
                    cross_attn_output[:, text_mask, :] = cross_attn_output_text_part  # 将计算出的文本部分交叉注意力结果填入完整输出的对应位置

                    cross_attn_output = cross_attn_output.squeeze(0)  # [s, hidden_size] - 移除增加的 batch 维度

                batch_cross_attn_outputs.append(cross_attn_output)  # 将当前样本的交叉注意力结果加入列表

            # 将当前层 batch 中所有样本的交叉注意力结果堆叠起来
            layer_cross_attn_output = torch.stack(batch_cross_attn_outputs, dim=0)  # [b, s, hidden_size]
            all_layer_outputs.append(layer_cross_attn_output)  # 将当前层的交叉注意力结果加入总列表
        total_loss = total_loss / (len(self.indices) * batch_size)
        return all_layer_outputs, total_loss  # 返回一个列表，包含和每一层交叉注意力的结果 (形状和原始query相同，只有文本部分更新)

    def cross_attention_for_image(self, query, encoder_outputs, image_mask, sentence_mask):
        """和 图像嵌入 做交叉注意力 (image_mask shape: [b, s])"""
        batch_size, seq_len = image_mask.shape
        hidden_size = query.size(-1)
        batch_cross_attn_outputs = []
        query = self.mlp1(query)
        total_loss = 0.0
        for b_idx in range(batch_size):  # 遍历 batch 维度
            current_image_mask = image_mask[b_idx]  # [s] - 当前 batch 样本的 image_mask
            current_sentence_mask = sentence_mask[b_idx]  # [s] - 当前 batch 样本的 sentence_mask
            current_query = query[b_idx]  # [s, hidden_size] - 当前 batch 样本的 query
            current_encoder_outputs = encoder_outputs[b_idx]  # [s, hidden_size] - 当前 batch 样本的 encoder output

            image_mask_bool = current_image_mask.bool()  # 转换为 boolean mask [s]
            image_embedding = current_encoder_outputs[image_mask_bool]  # [num_image_tokens, hidden_size] - 当前样本的图像嵌入

            text_mask = current_sentence_mask.bool()  # 转换为 boolean mask [s]
            text_query = current_query[text_mask]  # [num_text_tokens, hidden_size] - 当前样本的文本 query

            if image_embedding.numel() == 0:  # 当前样本没有图像
                cross_attn_output = torch.zeros_like(current_query)  # 用零张量填充
            else:
                image_embedding = image_embedding.unsqueeze(0)  # [1, num_image_tokens, hidden_size] - 增加 batch 维度
                query_expanded = text_query.unsqueeze(0)  # [1, s, hidden_size] - 增加 batch 维度，query 也需要扩展维度匹配
                cross_attn_output_image_part, cross_attn_weight = self.cross_attention_image(query_expanded,
                                                                          image_embedding)  # [1, s, hidden_size] - 交叉注意力计算 (图像部分)
                total_loss = total_loss + torch.mean(torch.abs(cross_attn_weight)) * self.sparsity_loss_weight
                cross_attn_output = torch.zeros_like(current_query).unsqueeze(0)  # [1, s, hidden_size] - 初始化完整输出
                cross_attn_output[:, text_mask, :] = cross_attn_output_image_part  # 将计算出的图像部分结果填入完整输出
                # 注意，我们是让文本做q，进而去做后续的计算

                cross_attn_output = cross_attn_output.squeeze(0)  # [s, hidden_size] - 移除 batch 维度

            batch_cross_attn_outputs.append(cross_attn_output)

        layer_cross_attn_output = torch.stack(batch_cross_attn_outputs, dim=0)  # [b, s, hidden_size]
        total_loss = total_loss / batch_size
        return layer_cross_attn_output, total_loss

    def fuse_features(self, last_layer_feature, cross_attention_layer_outputs, cross_attention_image_output):
        """注意力融合特征"""
        all_features = cross_attention_layer_outputs + [cross_attention_image_output,
                                                        last_layer_feature]  # 列表，包含所有要融合的特征
        # 将特征列表堆叠成 [L+2, batch_size, seq_len, dim] 的张量
        features_tensor = torch.stack(all_features, dim=0)  # [N, b, seq_len, dim]  N = L+2
        # 调整维度顺序为 [batch_size, N, seq_len, dim] -> [batch_size, N, seq_len, dim]  方便 FusionAttention 模块处理
        features_tensor = features_tensor.transpose(0, 1)  # [b, N, seq_len, dim]

        fused_feature = self.fusion_attention(features_tensor)  # 使用注意力融合模块
        # 门控机制融合。

        # 再计算融合机制前，先对encoder最后的输出套一个MLP
        # last_layer_feature = self.mlp2(last_layer_feature)
        #
        # # 尝试1、
        # all_features = cross_attention_layer_outputs + [cross_attention_image_output, last_layer_feature]
        #
        # all_features = cross_attention_layer_outputs + [last_layer_feature]
        # fused_feature = torch.zeros_like(last_layer_feature)
        #
        # fused_feature = 0.7 * last_layer_feature  # 还是保留一定程度的 encoder的 最后输出
        #
        # for i, feature in enumerate(all_features):
        #     gate = torch.sigmoid(self.gate_projs[i](torch.cat([last_layer_feature, feature], dim=-1)))  # 每个来源单独计算门控
        #     fused_feature = fused_feature + gate * feature
        #
        # 尝试2、 使用和情绪计算中使用的加权求和方法
        # for i, feature in enumerate(all_features):
        #     weight = self.compute_correlation_weights_learnable(last_layer_feature, feature,
        #                                                         self.fusion_weight_matrices[i])
        #     fused_feature = fused_feature + torch.matmul(weight, feature)

        return fused_feature

    def compute_correlation_weights_learnable(self, tensor1, tensor2, learnable_weight_matrix):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活

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

    def compute_correlation_weights_learnable_1(self, tensor1, tensor2):
        """使用可学习的权重矩阵计算权重"""
        # 使用可学习的权重矩阵进行线性变换和交互 后续的sigmoid是否还需要使用，再试试看。
        interaction_scores = torch.einsum('bsd,dp,bjd->bsj',
                                          tensor1,
                                          self.learnable_weight_matrix,
                                          tensor2)  # [b, s, j]

        weights = torch.sigmoid(interaction_scores)  # 仍然可以使用 Sigmoid 激活

        return weights
