import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalCompletionModule(nn.Module):
    """处理不同长度的模态输入并进行补全的模块，支持全局掩码"""

    def __init__(self, hidden_size, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length  # 设置最大长度为20

        # 文本到字幕的投影
        self.text_to_caption_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768)
        )

        # 图像到字幕的投影
        self.image_to_caption_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768)
        )

        # 多模态融合门控
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 长度感知的注意力
        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

        # 图像注意力
        self.image_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

        # 输出长度调整层
        self.length_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        # 质量评估器
        self.quality_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数，根据参数维度选择合适的初始化方法"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 检查参数的维度
                if len(param.shape) >= 2:
                    # 对于2维及以上的参数，使用xavier_normal_初始化
                    nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    # 对于1维参数（如某些偏置项），使用正态分布初始化
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    # 对于标量参数，使用常数初始化
                    nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                # 偏置项初始化为0
                nn.init.zeros_(param)

    def complete_caption(self, encoder_hidden_states, attention_mask=None,
                         sentence_mask=None, image_mask=None, image_caption_mask=None,
                         image_caption_valid=None, relevance_score=None, threshold=None):
        """
        根据文本和图像生成字幕表示，使用全局掩码

        Args:
            encoder_hidden_states: 编码器输出的隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 全局注意力掩码 [batch_size, seq_len]
            sentence_mask: 文本部分掩码 [batch_size, seq_len]
            image_mask: 图像部分掩码 [batch_size, seq_len]
            image_caption_mask: 字幕部分掩码 [batch_size, seq_len]
            image_caption_valid: 字幕有效性标志 [batch_size]
            relevance_score: 文本和图像的相关度分数 [batch_size, 1]

        Returns:
            caption_emb: 补全的字幕嵌入 [batch_size, seq_len, hidden_size]
            quality_score: 补全质量评分 [batch_size, 1]
            target_length: 实际生成的字幕长度
        """
        batch_size = encoder_hidden_states.size(0)
        device = attention_mask.device
        seq_len = encoder_hidden_states.size(1)
        sentence_mask = sentence_mask.to(device)
        image_mask = image_mask.to(device)
        image_caption_mask = image_caption_mask.to(device)
        threshold = threshold.to(device)
        relevance_score = relevance_score.to(device)
        # 提取文本、图像和字幕的表示
        text_emb = encoder_hidden_states * sentence_mask.unsqueeze(-1).float().to(device)
        image_emb = encoder_hidden_states * image_mask.unsqueeze(-1).float().to(device)
        # 确定当前批次中有效字幕的最大长度
        batch_max_length = 0
        begin_caption_emb = None
        end_caption_emb = None

        # 从有效字幕中提取特殊标记的嵌入和最大长度
        for i in range(batch_size):
            if image_caption_valid is not None and image_caption_valid[i]:
                # 找到字幕掩码的位置
                caption_indices = torch.where(image_caption_mask[i] > 0)[0]
                if len(caption_indices) > 0:
                    current_length = len(caption_indices)
                    batch_max_length = max(batch_max_length, current_length)

                    # 找到特殊标记的位置：掩码前一个和后一个位置
                    begin_pos = caption_indices[0] - 1
                    end_pos = caption_indices[-1] + 1

                    # 确保索引有效
                    if 0 <= begin_pos < seq_len and 0 <= end_pos < seq_len:
                        # 提取特殊标记的嵌入
                        if begin_caption_emb is None:
                            begin_caption_emb = encoder_hidden_states[i, begin_pos].unsqueeze(0)
                            end_caption_emb = encoder_hidden_states[i, end_pos].unsqueeze(0)

        # 如果没有有效字幕，使用默认长度
        if batch_max_length == 0:
            batch_max_length = 20

        # 确保不超过最大长度20 但是需要减去2 是两个特殊token
        target_length = min(batch_max_length, self.max_length - 2)
        # 文本池化 - 只考虑文本部分的有效token
        text_lengths = sentence_mask.sum(dim=1, keepdim=True).float()
        text_pooled = (text_emb * sentence_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(text_lengths, min=1e-9)

        # 图像池化 - 只考虑图像部分的有效token
        image_lengths = image_mask.sum(dim=1, keepdim=True).float()
        image_pooled = (image_emb * image_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(image_lengths, min=1e-9)

        # 从文本生成字幕表示
        text_caption_seed = self.text_to_caption_proj(text_pooled)

        # 从图像生成字幕表示
        image_caption_seed = self.image_to_caption_proj(image_pooled)

        # 使用相关度分数作为融合权重，如果没有则学习权重
        if relevance_score is not None:
            alpha = relevance_score
        else:
            alpha = self.modality_gate(torch.cat([text_pooled, image_pooled], dim=-1)).to(device)

        # 融合文本和图像种子
        caption_seed = alpha * image_caption_seed + (1 - alpha) * text_caption_seed

        # 创建位置编码
        position_embeddings = torch.zeros(1, target_length, self.hidden_size, device=device)
        for pos in range(target_length):
            for i in range(0, self.hidden_size, 2):
                position_embeddings[0, pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / self.hidden_size))))
                if i + 1 < self.hidden_size:
                    position_embeddings[0, pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (i / self.hidden_size))))

        # 将位置编码与种子表示结合
        query = caption_seed.unsqueeze(1) + position_embeddings

        # 准备文本注意力的掩码 - 需要转换为MultiheadAttention期望的格式
        text_padding_mask = (sentence_mask == 0)  # True表示padding位置

        # 使用文本注意力扩展到目标长度
        text_caption_emb, _ = self.text_attention(
            query=query,
            key=text_emb,
            value=text_emb,
            key_padding_mask=text_padding_mask
        )

        # 准备图像注意力的掩码
        image_padding_mask = (image_mask == 0)  # True表示padding位置

        # 使用图像注意力增强字幕表示
        image_caption_emb, _ = self.image_attention(
            query=query,
            key=image_emb,
            value=image_emb,
            key_padding_mask=image_padding_mask
        )

        # 使用相关度分数融合两种字幕表示
        if relevance_score is not None:
            beta = relevance_score.unsqueeze(1)  # [batch_size, 1, 1]
        else:
            beta = torch.sigmoid(torch.bmm(text_caption_emb, image_caption_emb.transpose(1, 2))).mean(dim=2,
                                                                                                      keepdim=True).to(device)

        caption_emb = beta * image_caption_emb + (1 - beta) * text_caption_emb

        # 应用长度适配器
        caption_emb = self.length_adapter(caption_emb)

        # 评估补全质量
        quality_score = self.quality_estimator(caption_emb.mean(dim=1))

        # caption_all = []
        # for i in range(batch_size):
        #     # 文本池化 - 只考虑文本部分的有效token
        #     text_lengths = sentence_mask[i, ].sum(dim=0, keepdim=True).float().to(device)
        #     # print("2", (text_emb * sentence_mask.unsqueeze(-1)).sum(dim=1), torch.clamp(text_lengths, min=1e-9))
        #     text_pooled = (text_emb[i, :, :] * sentence_mask[i].unsqueeze(-1)).sum(dim=0) / torch.clamp(text_lengths, min=1e-9)
        #
        #     text_pooled = text_pooled.to(device)
        #
        #     # 图像池化 - 只考虑图像部分的有效token
        #     image_lengths = image_mask[i, ].sum(dim=0, keepdim=True).float().to(device)
        #     image_pooled = (image_emb[i, :, :] * image_mask[i].unsqueeze(-1)).sum(dim=0) / torch.clamp(image_lengths, min=1e-9)
        #     image_pooled = image_pooled.to(device)
        #
        #     # 从文本生成字幕表示
        #     text_caption_seed = self.text_to_caption_proj(text_pooled).to(device)
        #
        #     # 从图像生成字幕表示
        #     image_caption_seed = self.image_to_caption_proj(image_pooled).to(device)
        #
        #     # 使用相关度分数作为融合权重，如果没有则学习权重
        #     if relevance_score[i, ] is not None:
        #         alpha = relevance_score[i, ].to(device)
        #     else:
        #         alpha = self.modality_gate(torch.cat([text_pooled, image_pooled], dim=-1)).to(device)
        #
        #     # 融合文本和图像种子
        #     # 在这里同样采用 阈值限制
        #     if alpha < threshold:
        #         alpha = 0.0
        #     caption_seed = alpha * image_caption_seed + (1 - alpha) * text_caption_seed
        #     # 创建位置编码
        #     position_embeddings = torch.zeros(1, target_length, self.hidden_size, device=device)
        #     for pos in range(target_length):
        #         for j in range(0, self.hidden_size, 2):
        #             # print("3", torch.tensor(pos / (10000 ** (j / self.hidden_size))), target_length)
        #             position_embeddings[0, pos, j] = torch.sin(torch.tensor(pos / (10000 ** (j / self.hidden_size))))
        #             if j + 1 < self.hidden_size:
        #                 position_embeddings[0, pos, j + 1] = torch.cos(torch.tensor(pos / (10000 ** (j / self.hidden_size))))
        #
        #     # 将位置编码与种子表示结合 为字幕表示增加长度维度
        #     query = caption_seed.unsqueeze(0).unsqueeze(1) + position_embeddings
        #     query = query.to(device)
        #
        #     # 准备文本注意力的掩码 - 需要转换为MultiheadAttention期望的格式
        #     text_padding_mask = (sentence_mask[i, ] == 0).to(device)  # True表示padding位置
        #     # 使用文本注意力扩展到目标长度
        #     text_caption_emb, _ = self.text_attention(
        #         query=query,
        #         key=text_emb[i].unsqueeze(0),
        #         value=text_emb[i].unsqueeze(0),
        #         key_padding_mask=text_padding_mask.unsqueeze(0)
        #     )
        #
        #     # 准备图像注意力的掩码
        #     image_padding_mask = (image_mask[i, ] == 0).to(device)  # True表示padding位置
        #
        #     # 使用图像注意力增强字幕表示
        #     image_caption_emb, _ = self.image_attention(
        #         query=query,
        #         key=image_emb[i].unsqueeze(0),
        #         value=image_emb[i].unsqueeze(0),
        #         key_padding_mask=image_padding_mask.unsqueeze(0)
        #     )
        #
        #     # 使用相关度分数融合两种字幕表示
        #     if relevance_score[i, ] is not None:
        #         beta = relevance_score[i, ].unsqueeze(1).to(device)  # [batch_size, 1, 1]
        #     else:
        #         beta = torch.sigmoid(torch.bmm(text_caption_emb, image_caption_emb.transpose(1, 2))).mean(dim=2,
        #                                                                                                   keepdim=True).to(device)
        #     if beta < threshold:
        #         beta = 0.0
        #     caption_emb = beta * image_caption_emb + (1 - beta) * text_caption_emb
        #
        #     # 应用长度适配器
        #     caption_emb = self.length_adapter(caption_emb).to(device)
        #     caption_all.append(caption_emb)
        # caption_all = torch.stack(caption_all, dim=0)
        # print(caption_all.size())
        # # 评估补全质量
        # quality_score = self.quality_estimator(caption_all.mean(dim=1)).to(device)

        return caption_emb, quality_score, target_length, begin_caption_emb, end_caption_emb