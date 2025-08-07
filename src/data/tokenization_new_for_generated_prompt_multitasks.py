import torch
import numpy as np
from transformers import BartTokenizer, AutoTokenizer
from itertools import chain
from functools import cmp_to_key
# from src.utils import TaskType
import spacy

def cmp(v1, v2):
    if v1[0] == v2[0]:
        return v1[1] - v2[1]
    return v1[0] - v2[0]


class ConditionTokenizer:
    """
    tokenizer for image features, event and task type
    this is NOT inherent from transformers Tokenizer
    考虑在Aspect的token中加上前后关系。
    """
    def __init__(self,
                 args,
                 pretrained_model_name=r"data/bart-base",
                 cls_token="<<cls>>",
                 mlm_token="<<mlm>>",
                 mrm_token="<<mrm>>",
                 begin_text="<<text>>",
                 end_text="<</text>>",
                 img_feat='<<img_feat>>',
                 begin_img="<<img>>",
                 end_img="<</img>>",
                 img_caption='<<img_caption>>',
                 begin_caption='<<cap>>',
                 end_caption='<</cap>>',
                 ae_token='<<AE>>',
                 sc_token='<<SC>>',
                 ae_oe_token="<<AOE>>",
                 sep_token="<<SEP>>",
                 aesc_token='<<AESC>>',
                 pos_token='<<POS>>',
                 neu_token='<<NEU>>',
                 neg_token='<<NEG>>',
                 aspect_prompt_token='<<AE_PROMPT>>',
                 senti_prompt_token='<<SENTI_PROMPT>>',
                 begin_prompt='<<prompt>>',
                 end_prompt='<</prompt>>',
                 senti_token='<<senti>>',
                 aspects_num_token='<<aspects_num>>',
                 ANP_token='<<ANP>>',
                 ANP_generate_token='<<AOG>>',
                 mask_token='<<MASK>>',
                 aspect_prompt_token_front='<<AE_PROMPT_FRONT>>',
                 aspect_prompt_token_end='<<AE_PROMPT_END>>',
                 invalid_caption_token='<<INVALID_CAPTION>>'):
        self._base_tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name, )
        # self._base_tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name)
        # 自定义Token的添加 后续需要添加Token 在

        self.additional_special_tokens = [
            cls_token, mlm_token, mrm_token, begin_text, end_text, img_feat, begin_img, end_img, 
            img_caption, begin_caption, end_caption, aspects_num_token,
            senti_token, ANP_token, ANP_generate_token, mask_token,
            pos_token, neu_token, neg_token, ae_oe_token, sep_token,
            aesc_token, ae_token, sc_token, 
            aspect_prompt_token, senti_prompt_token, begin_prompt, end_prompt,
            aspect_prompt_token_front, aspect_prompt_token_end, invalid_caption_token
        ]
        unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.unique_no_split_tokens = unique_no_split_tokens + self.additional_special_tokens
        self.unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        print(self.unique_no_split_tokens)

        self._base_tokenizer.add_tokens(self.additional_special_tokens)
        self.cls_token = cls_token
        self.mlm_token = mlm_token
        self.mrm_token = mrm_token
        self.begin_text = begin_text
        self.end_text = end_text
        self.img_feat = img_feat
        self.begin_img = begin_img
        self.end_img = end_img

        self.img_caption = img_caption
        self.begin_caption = begin_caption
        self.end_caption = end_caption

        self.ae_token = ae_token
        self.sc_token = sc_token
        self.ae_oe_token = ae_oe_token
        self.sep_token = sep_token
        self.senti_token = senti_token
        self.ANP_token = ANP_token
        self.ANP_generate_token = ANP_generate_token

        self.aesc_token = aesc_token
        self.pos_token = pos_token
        self.neu_token = neu_token
        self.neg_token = neg_token

        self.aspect_prompt_token = aspect_prompt_token
        self.aspect_prompt_token_front = aspect_prompt_token_front
        self.aspect_prompt_token_end = aspect_prompt_token_end
        self.senti_prompt_token = senti_prompt_token
        self.begin_prompt = begin_prompt
        self.end_prompt = end_prompt
        self.aspects_num_token = aspects_num_token
        self.invalid_caption_token = invalid_caption_token

        self.mask_token = mask_token
        self.mask_token_id = self.convert_tokens_to_ids(mask_token)

        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.mlm_token_id = self.convert_tokens_to_ids(mlm_token)
        self.mrm_token_id = self.convert_tokens_to_ids(mrm_token)
        self.begin_text_id = self.convert_tokens_to_ids(begin_text)
        self.end_text_id = self.convert_tokens_to_ids(end_text)
        self.img_feat_id = self.convert_tokens_to_ids(img_feat)
        self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        self.end_img_id = self.convert_tokens_to_ids(end_img)

        self.img_caption_id = self.convert_tokens_to_ids(img_caption)
        self.begin_caption_id = self.convert_tokens_to_ids(begin_caption)
        self.end_caption_id = self.convert_tokens_to_ids(end_caption)

        self.ae_token_id = self.convert_tokens_to_ids(ae_token)
        self.sc_token_id = self.convert_tokens_to_ids(sc_token)
        self.ae_oe_token_id = self.convert_tokens_to_ids(ae_oe_token)
        self.sep_token_id = self.convert_tokens_to_ids(sep_token)
        self.senti_token_id = self.convert_tokens_to_ids(senti_token)
        self.ANP_token_id = self.convert_tokens_to_ids(ANP_token)
        self.ANP_generate_token_id = self.convert_tokens_to_ids(
            ANP_generate_token)
        self.aesc_token_id = self.convert_tokens_to_ids(aesc_token)
        self.pos_token_id = self.convert_tokens_to_ids(pos_token)
        self.neu_token_id = self.convert_tokens_to_ids(neu_token)
        self.neg_token_id = self.convert_tokens_to_ids(neg_token)

        self.aspect_prompt_token_id = self.convert_tokens_to_ids(aspect_prompt_token)
        self.aspect_prompt_token_front_id = self.convert_tokens_to_ids(aspect_prompt_token_front)
        self.aspect_prompt_token_end_id = self.convert_tokens_to_ids(aspect_prompt_token_end)
        self.senti_prompt_token_id = self.convert_tokens_to_ids(senti_prompt_token)
        self.begin_prompt_id = self.convert_tokens_to_ids(begin_prompt)
        self.end_prompt_id = self.convert_tokens_to_ids(end_prompt)
        self.aspects_num_token_id = self.convert_tokens_to_ids(aspects_num_token)
        self.invalid_caption_token_id = self.convert_tokens_to_ids(invalid_caption_token)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id

        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id


        print('self.bos_token_id', self.bos_token_id)
        print('self.eos_token_id', self.eos_token_id)
        print('self.pad_token_id', self.pad_token_id)
        print('self.begin_caption_token_id', self.begin_caption_id)
        print('self.end_caption_token_id', self.end_caption_id)
        print('self.aspect_prompt_token_id', self.aspect_prompt_token_id)
        print('self.senti_prompt_token_id', self.senti_prompt_token_id)
        print('self.begin_prompt_id', self.begin_prompt_id)
        print('self.end_prompt_id', self.end_prompt_id)
        print('self.aspects_num_token_id', self.aspects_num_token_id)

        if args.task == 'pretrain':
            self.mapping = {'AE_OE': '<<AOE>>', 'SEP': '<<SEP>>'}
        else:
            if args.task == 'twitter_sc':
                self.mapping = {
                    'SC': '<<SC>>',
                    'POS': '<<POS>>',
                    'NEU': '<<NEU>>',
                    'NEG': '<<NEG>>'
                }
            elif args.task == 'twitter_ae':
                self.mapping = {
                    'AE': '<<AE>>',
                    'POS': '<<POS>>',
                    'NEU': '<<NEU>>',
                    'NEG': '<<NEG>>'
                }
            else:
                self.mapping = {
                    'AESC': '<<AESC>>',
                    'POS': '<<POS>>',
                    'NEU': '<<NEU>>',
                    'NEG': '<<NEG>>'
                }
        self.senti = {'POS': '<<POS>>', 'NEU': '<<NEU>>', 'NEG': '<<NEG>>'}
        self.senti2id = {}
        for key, value in self.senti.items():
            key_id = self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            # assert key_id[0] >= self.cur_num_tokens
            self.senti2id[key] = key_id[0]
        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            # assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid) + 2
        print(self.mapping2id)
        print("文本下标的偏移值", len(self.mapping2targetid) + 2)
        self.nlp = spacy.load("en_core_web_sm")

        # --- SenticNet 初始化 ---
        try:
            from senticnet.senticnet import SenticNet  # 使用正确的类名
            self.sn = SenticNet()
            self.senticnet_available = True
            print("SenticNet initialized successfully.")
        except ImportError:
            print(
                "Warning: SenticNet library not found or import error. Sentiment features from SenticNet will not be available.")
            self.sn = None
            self.senticnet_available = False
        except Exception as e:
            print(
                f"Warning: Error initializing SenticNet: {e}. Sentiment features from SenticNet will not be available.")
            self.sn = None
            self.senticnet_available = False

        # SenticNet 极性阈值 (需要仔细调整)
        # abs(polarity) < neutral_threshold -> neutral
        # polarity >= positive_threshold -> positive
        # polarity <= negative_threshold -> negative
        self.neutral_threshold = 0.2  # 例如，-0.2 到 0.2 之间被视为中性
        self.positive_threshold = 0.2  # 例如，大于等于 0.2 被视为正面 (且不为中性)
        self.negative_threshold = -0.2  # 例如，小于等于 -0.2 被视为负面 (且不为中性)

        self.negation_words = {"not", "no", "never", "n't", "isnt", "arent", "dont", "doesnt", "didnt", "wasnt",
                               "werent"}

        '''
        for AESC:
        {'AESC': 50281, 'POS': 50276, 'NEU': 50277, 'NEG': 50278}
        '''

        # ----------------
        # 添加的用senticNet识别每一个词的极性，先作为 总的 情绪Prompt的一个指导信息。
    def _get_senticnet_polarity_value(self, word_text):
        """ 辅助函数：从SenticNet获取单个词的原始极性值 """
        if not self.senticnet_available:
            return None
        try:
            # SenticNet通常对小写、下划线连接的格式更友好
            formatted_word = word_text.lower().replace(" ", "_")
            concept_info = self.sn.concept(formatted_word)
            if concept_info and 'polarity_value' in concept_info and concept_info['polarity_value'] is not None:
                return float(concept_info['polarity_value'])
            # 尝试不替换空格
            if " " in word_text:
                concept_info_no_replace = self.sn.concept(word_text.lower())
                if concept_info_no_replace and 'polarity_value' in concept_info_no_replace and \
                        concept_info_no_replace[
                            'polarity_value'] is not None:
                    return float(concept_info_no_replace['polarity_value'])
        except KeyError:
            pass  # 概念未找到
        except Exception as e:
            print(f"Error querying SenticNet for '{word_text}': {e}")
        return None

    def _determine_sentiment_tag_str(self, polarity_value):
        """ 将SenticNet的连续极性值转换为特殊标记字符串，中性不返回标记
            self.pos_token = pos_token
            self.neu_token = neu_token
            self.neg_token = neg_token
            """
        if polarity_value is None:
            return ""  # 没有获取到极性，不加标记

        # 使用定义的阈值
        if polarity_value >= self.positive_threshold:
            return self.pos_token_id
        elif polarity_value <= self.negative_threshold:
            return self.neg_token_id
        else:  # 中性或接近中性
            return ""  # 对于中性词不操作

    # ------------

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)
    # 新添加的部分： ------------
    def pad(self, encoded_inputs, padding='longest', max_length=None, pad_to_multiple_of=None,
            return_attention_mask=True, return_tensors='pt'):
        """
        Pads encoded inputs to the longest sequence in the batch, or to max_length if provided.
        Returns PyTorch tensors.
        """

        if padding not in ['longest', True, 'max_length', False]:  # Add 'max_length' and False options
            raise ValueError(f"Padding strategy '{padding}' not supported")

        input_ids_list = [example["input_ids"] for example in encoded_inputs]  # Extract input_ids lists
        device = input_ids_list[0].device
        if padding == 'longest' or padding is True:  # 'longest' or True: pad to max length in batch
            max_len = max(len(ids) for ids in input_ids_list)
        elif padding == 'max_length':  # 'max_length': pad to max_length
            max_len = max_length
        elif not padding or padding == False:  # False or None: no padding
            max_len = None
        else:
            max_len = None  # Default to no padding if padding strategy is not recognized

        if max_len is None:  # No padding needed
            padded_input_ids = [torch.tensor(ids, device=device) for ids in
                                input_ids_list]  # Move to device when creating tensor
            attention_masks = [torch.ones_like(ids, device=device).int() if return_attention_mask else None for ids in
                               padded_input_ids]  # Move to device
        else:  # Padding is needed
            padded_input_ids = []
            attention_masks = []
            for ids in input_ids_list:
                padding_len = max_len - len(ids)
                padding_id = self.pad_token_id  # Get pad_token_id

                padding_tensor = torch.tensor([padding_id] * padding_len, dtype=torch.long,
                                              device=device)  # <-- Specify device=device when creating padding_tensor!
                padded_ids = torch.cat([ids, padding_tensor])  # Tensor concatenation (both on GPU now)
                padded_input_ids.append(padded_ids)

                if return_attention_mask:  # Create attention mask
                    attention_mask = torch.cat([torch.ones_like(ids, device=device).int(),
                                                torch.zeros_like(padding_tensor,
                                                                 device=device).int()])  # Specify device for attention mask tensors too
                    attention_masks.append(attention_mask)
                else:
                    attention_masks.append(None)

        # Stack into batch tensors
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(attention_masks) if return_attention_mask else None

        # Prepare output dictionary
        output = {"input_ids": input_ids_batch}
        if return_attention_mask:
            output["attention_mask"] = attention_mask_batch

        return output

    def get_special_tokens_mask(self, token_ids_list, already_has_special_tokens=False):
        """
        Returns a list indicating special tokens. 1 for special tokens, 0 for regular tokens.
        """
        special_token_ids = set()  # 初始化为空集合

        # 遍历 self.additional_special_tokens 列表，获取每个 special token 的 ID，并添加到 special_token_ids 集合
        for special_token in self.additional_special_tokens:
            if special_token is not None:  # 确保 special_token 不是 None
                token_id = self.convert_tokens_to_ids(special_token)  # 获取 special token 的 ID
                if token_id is not None:  # 确保 token_id 有效
                    special_token_ids.add(token_id)  # 添加到 special_token_ids 集合

        special_token_ids.add(self.pad_token_id)  # 始终包含 pad_token_id
        special_token_ids.add(self.unk_token_id)  # 始终包含 unk_token_id
        # 不需要再手动添加 bos_token_id, eos_token_id, sep_token_id, cls_token_id, mask_token_id，因为 additional_special_tokens 已经包含了大部分特殊 token

        special_tokens_mask = [1 if token_id in special_token_ids else 0 for token_id in token_ids_list]
        return special_tokens_mask

    # ---------------------
    def pad_tokens(self, tokens):
        max_len = max([len(x) for x in tokens])
        pad_result = torch.full((len(tokens), max_len),
                                self.pad_token_id,
                                dtype=torch.long)
        mask = torch.zeros(pad_result.size(), dtype=torch.bool)
        for i, x in enumerate(tokens):
            pad_result[i, :len(x)] = torch.tensor(tokens[i], dtype=torch.long)
            mask[i, :len(x)] = True
        return pad_result, mask
    
    def pad_tokens_with_maxlength(self, tokens, max_len):
        pad_result = torch.full((len(tokens), max_len),
                                self.pad_token_id,
                                dtype=torch.long)
        mask = torch.zeros(pad_result.size(), dtype=torch.bool)

        for i, x in enumerate(tokens):
            # print(x)
            pad_result[i, :len(x)] = torch.tensor(tokens[i], dtype=torch.long)
            mask[i, :len(x)] = True
        # print("=====================pad_result=========================")
        # print(pad_result)
        # print(mask)
        return pad_result, mask
    

    def encode_mlm_sentence(self, labels):
        label_split = [x.split() for x in labels]
        input_tokens = []
        for split in label_split:
            cur_num = 0
            bpes = [self.bos_token_id]
            for x in split:
                tokens = self._base_tokenizer(x, add_prefix_space=True)
                bpes = bpes + tokens
            bpes.append(self.eos_token_id)
            input_tokens.append(input_tokens)
        return input_tokens

    def encode_condition(self, task, img_num=None, use_caption=True, caption=None, sentence=None, has_prompt=False, max_aspects_num=None,  text_only=False):
        """
        tokenize text, image features and event
        the output format (after decoded back):
        task_type [<img> <img_feat> ... <img_feat> </img>] [<event> EVENT </event>] [<mlm> MLM </mlm>]

        :param task_type: str or list[str]
        :param img_num: int or list[int], the number of image features
        :param event: str or list[str], event descriptions
        :param mlm: str or list[str], sentence for masked language modeling
        :return: dict {str: Tensor}, {
                "input_ids": ...,
                "attention_mask": ...,
                "event_mask": ...,          only exist if event is given. 1 for the position with event tokens
                "mlm_mask": ...,            only exist if mlm is given. 1 for the position with mlm tokens
                "img_mask":...,             only exist if img_num is given. 1 for the position with img tokens
            }
        """
        
        '''
        [image_features] + is + [image_caption] + [text] + [aspect_prompt_token]*len_1(最多为5) + has + [senti_prompt_token]*len_2 'sentiment'

        prompt_1: [image_features] + is + [image_caption]  
        + 'There is' <prompt> ([aspect_prompt_token]*len_1(最多为5) + of [senti_prompt_token]*len_2 + 'sentiment' + <sep>)*n </prompt> + 'in'
        + [text]

        prompt_2: [image_features] + is + [image_caption]  
        + <prompt> ([aspect_prompt_token]*len_1(最多为5) + of [senti_prompt_token]*len_2)*n </prompt> 
        + [text]

        prompt_3: [image_features] + is + [image_caption]  
        + 'There is' <prompt> ([aspect_prompt_token]*len_1(最多为5) + of <<NEU>> or <<POS>> or <<NEG>> + 'sentiment' + <sep>)*n </prompt> + 'in'
        + [text]

        '''

        image_text = None
        if img_num is not None:
            if not isinstance(img_num, list):
                img_num = [img_num]
            image_text = []
            for index, value in enumerate(img_num):
                image_text.append(self.begin_img + self.img_feat * value +  ###引入image_caption token
                                  self.end_img)
                # 这里得到的是一个特殊字段

        # import ipdb; ipdb.set_trace()
        image_caption_valid = []  # 用0 1 判断字幕是否是正确的信息
        if caption is not None:
            if not isinstance(caption, list):
                caption = [caption]
            caption_split = [x.split() for x in caption]
            image_caption_tokens = []
            batch_caption_nouns = []  # 新增：用于存储每个caption的名词列表
            for split in caption_split:
                '''
                print(split)
                为方便起见，固定caption文本长度
                '''
                # print("+++++++++++++++++++++split before ++++++++++++++++++++++++")
                # print(len(split))
                if len(split) > 12:
                    split = split[:12]
                # print("+++++++++++++++++++++split after ++++++++++++++++++++++++")
                # print(len(split))
                # print(split)
                is_bpes = self._base_tokenizer.tokenize('is',
                                                         add_prefix_space=True)
                is_bpes = self._base_tokenizer.convert_tokens_to_ids(is_bpes) ##[16]
                caption_word_bpes = [is_bpes]

                caption_word_bpes.append([self.begin_caption_id])

                for caption_word in split:
                    caption_bpes = self._base_tokenizer.tokenize(caption_word,
                                                         add_prefix_space=True)
                    caption_bpes = self._base_tokenizer.convert_tokens_to_ids(caption_bpes)
                    caption_word_bpes.append(caption_bpes)
                caption_word_bpes.append([self.end_caption_id])

                _caption_word_bpes = list(chain(*caption_word_bpes))
                if split[0] == self.invalid_caption_token:
                    image_caption_valid.append(0)
                    batch_caption_nouns.append([])  # 无效，返回空

                elif len(_caption_word_bpes) > 20:
                    print("出现了错误的字幕信息，直接丢弃")
                    is_bpes = self._base_tokenizer.tokenize('is',
                                                            add_prefix_space=True)
                    is_bpes = self._base_tokenizer.convert_tokens_to_ids(is_bpes)  ##[16]
                    caption_word_bpes = [is_bpes]

                    caption_word_bpes.append([self.begin_caption_id])
                    _caption_word_bpes.append([self.invalid_caption_token_id])
                    caption_word_bpes.append([self.end_caption_id])
                    _caption_word_bpes = list(chain(*caption_word_bpes))
                    image_caption_valid.append(0)
                    batch_caption_nouns.append([])
                else:
                    image_caption_valid.append(1)
                    caption_text = " ".join(split)
                    doc = self.nlp(caption_text)
                    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
                    batch_caption_nouns.append(nouns)
                image_caption_tokens.append(_caption_word_bpes.copy())

        if sentence is not None:
            if not isinstance(sentence, list):
                sentence = [sentence]
            sentence_split = [x.split() for x in sentence]
            input_sentence_tokens = []
            batch_sentence_nouns = []  # 新增：用于存储每个句子的名词
            for split in sentence_split:
                word_bpes = [[self.bos_token_id]]

                previous_word_for_negation = None  # 用于否定检测
                for idx, word in enumerate(split):
                    # --- SenticNet情感处理 ---
                    word_polarity_value = self._get_senticnet_polarity_value(word)
                    # 简单否定检查
                    # if previous_word_for_negation and previous_word_for_negation.lower() in self.negation_words and word_polarity_value is not None:
                    #     word_polarity_value = -word_polarity_value  # 反转极性

                    word_sentiment_category_id = self._determine_sentiment_tag_str(word_polarity_value)

                    bpes = self._base_tokenizer.tokenize(word,
                                                         add_prefix_space=(idx > 0 or not word.startswith(" ")))
                    bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                    word_bpes.append(bpes)
                    # 将当前词的情感类别ID赋给其所有的BPE tokens
                    # if word_sentiment_category_id:
                    #     word_bpes.append([word_sentiment_category_id])
                word_bpes.append([self.eos_token_id])

                _word_bpes = list(chain(*word_bpes))
                input_sentence_tokens.append(_word_bpes.copy())
                # 新增：提取名词
                sentence_text = " ".join(split)
                doc = self.nlp(sentence_text)
                nouns = [token.text for token in doc if token.pos_ == "NOUN"]
                batch_sentence_nouns.append(nouns)
        
        '''
        for prompt_1:  prompt_1: [image_features] + is + [image_caption]  
        + 'There is' <prompt> ([aspect_prompt_token]*len_1(最多为5) + of [senti_prompt_token]*len_2 + 'sentiment' + <sep>)*n </prompt> + 'in'
        + [text] 效果很好 
        in input_ids: input_ids[:][24:64] is the related to aspect prompts, so in generated prompt stage, the attention_mask[:][24:64]=0
        '''
        # if has_prompt:
        #     aspect_prompts_tokens = []     
        #     for index, value in enumerate(aspects_num):
        #         # aspect_prompts.append(self.begin_prompt + (self.aspect_prompt_token*2 + 'has' +  self.senti_prompt_token + 'sentiment' + self.sep_token) * value  +
        #         #                   self.end_prompt)
        #         aspect_prompt_bpes = [[self.begin_prompt_id]]
        #         if value == 1:
        #             _be = 'is'
        #         else:
        #             _be = 'are'
                
        #         there_bpes = self._base_tokenizer.tokenize('there',
        #                                                 add_prefix_space=True)
        #         there_bpes = self._base_tokenizer.convert_tokens_to_ids(there_bpes)
        #         aspect_prompt_bpes.append(there_bpes)

        #         _be_bpes = self._base_tokenizer.tokenize(_be,
        #                                                 add_prefix_space=True)
        #         _be_bpes = self._base_tokenizer.convert_tokens_to_ids(_be_bpes)
        #         aspect_prompt_bpes.append(_be_bpes)

        #         for i in range(value):
        #             aspect_prompt_bpes.append([self.aspect_prompt_token_id]*2)

        #             of_bpes = self._base_tokenizer.tokenize('of',
        #                                                     add_prefix_space=True)
        #             of_bpes = self._base_tokenizer.convert_tokens_to_ids(of_bpes)
        #             aspect_prompt_bpes.append(of_bpes)
                    
        #             aspect_prompt_bpes.append([self.senti_prompt_token_id])

        #             senti_bpes = self._base_tokenizer.tokenize('sentiment',
        #                                                     add_prefix_space=True)
        #             senti_bpes = self._base_tokenizer.convert_tokens_to_ids(senti_bpes)
        #             aspect_prompt_bpes.append(senti_bpes)

        #             if i <(value-1):
        #                 aspect_prompt_bpes.append([self.sep_token_id])


        #         aspect_prompt_bpes.append([self.end_prompt_id])
        #         in_bpes =  self._base_tokenizer.tokenize('in',
        #                                                 add_prefix_space=True)
        #         in_bpes = self._base_tokenizer.convert_tokens_to_ids(in_bpes)
        #         aspect_prompt_bpes.append(in_bpes)
        #         _aspect_prompt_bpes = list(chain(*aspect_prompt_bpes))
        #         aspect_prompts_tokens.append(_aspect_prompt_bpes.copy())

        '''
        for prompt_4:  prompt_1: [image_features] + is + [image_caption]  
        + <prompt> ([aspect_prompt_token]*len_1(最多为5) + of [senti_prompt_token]*len_2)*n </prompt> + 'in'
        + [text] 效果很好 
        in input_ids: input_ids[:][24:64] is the related to aspect prompts, so in generated prompt stage, the attention_mask[:][24:64]=0
        '''
        if has_prompt:
            aspect_prompts_tokens = []     
            for index, value in enumerate(max_aspects_num):
                # aspect_prompts.append(self.begin_prompt + (self.aspect_prompt_token*2 + 'has' +  self.senti_prompt_token + 'sentiment' + self.sep_token) * value  +
                #                   self.end_prompt)
                aspect_prompt_bpes = [[self.begin_prompt_id]]
                if value == 1:
                    _be = 'is'
                else:
                    _be = 'are'
                
                there_bpes = self._base_tokenizer.tokenize('there',
                                                        add_prefix_space=True)
                there_bpes = self._base_tokenizer.convert_tokens_to_ids(there_bpes)
                aspect_prompt_bpes.append(there_bpes)

                _be_bpes = self._base_tokenizer.tokenize(_be,
                                                        add_prefix_space=True)
                _be_bpes = self._base_tokenizer.convert_tokens_to_ids(_be_bpes)
                aspect_prompt_bpes.append(_be_bpes)

                for i in range(value):
                    if task == 'AESC':
                        #  考虑为aspect增加前后的区别
                        # aspect_prompt_bpes.append([self.aspect_prompt_token_id]*2)
                        aspect_prompt_bpes.append([self.aspect_prompt_token_front_id])
                        aspect_prompt_bpes.append([self.aspect_prompt_token_end_id])
                        of_bpes = self._base_tokenizer.tokenize('of',
                                                                add_prefix_space=True)
                        of_bpes = self._base_tokenizer.convert_tokens_to_ids(of_bpes)
                        aspect_prompt_bpes.append(of_bpes)
                        
                        aspect_prompt_bpes.append([self.senti_prompt_token_id])

                        senti_bpes = self._base_tokenizer.tokenize('sentiment',
                                                                add_prefix_space=True)
                        senti_bpes = self._base_tokenizer.convert_tokens_to_ids(senti_bpes)
                        aspect_prompt_bpes.append(senti_bpes)

                        if i <(value-1):
                            aspect_prompt_bpes.append([self.sep_token_id])
                        # 总长度(max_num - 1) * 6 + 5
                    elif task == 'twitter_sc':
                        aspect_prompt_bpes.append([self.senti_prompt_token_id])

                        senti_bpes = self._base_tokenizer.tokenize('sentiment',
                                                                add_prefix_space=True)
                        senti_bpes = self._base_tokenizer.convert_tokens_to_ids(senti_bpes)
                        aspect_prompt_bpes.append(senti_bpes)

                        if i <(value-1):
                            aspect_prompt_bpes.append([self.sep_token_id])
                    elif task == 'twitter_ae':
                        # aspect_prompt_bpes.append([self.aspect_prompt_token_id]*2)
                        aspect_prompt_bpes.append([self.aspect_prompt_token_front_id])
                        aspect_prompt_bpes.append([self.aspect_prompt_token_end_id])
                        if i <(value-1):
                            aspect_prompt_bpes.append([self.sep_token_id])
                        # 总长度(max_num - 1) * 3 + 2
                    else:
                        print('Not is right task, please check code!!!')


                aspect_prompt_bpes.append([self.end_prompt_id])
                in_bpes =  self._base_tokenizer.tokenize('in',
                                                        add_prefix_space=True)
                in_bpes = self._base_tokenizer.convert_tokens_to_ids(in_bpes)
                aspect_prompt_bpes.append(in_bpes)
                _aspect_prompt_bpes = list(chain(*aspect_prompt_bpes))
                aspect_prompts_tokens.append(_aspect_prompt_bpes.copy())
        # import ipdb; ipdb.set_trace()

        '''
        for prompt_2
        prompt_2: [image_features] + is + [image_caption]  
        + <prompt> ([aspect_prompt_token]*len_1(最多为5) + [senti_prompt_token]*len_2)*n </prompt> 
        + [text] 试一试
        '''

        # if has_prompt:
        #     aspect_prompts_tokens = []     
        #     for index, value in enumerate(aspects_num):
        #         # aspect_prompts.append(self.begin_prompt + (self.aspect_prompt_token*2 + 'has' +  self.senti_prompt_token + 'sentiment' + self.sep_token) * value  +
        #         #                   self.end_prompt)
        #         aspect_prompt_bpes = [[self.begin_prompt_id]]

        #         for i in range(value):
        #             aspect_prompt_bpes.append([self.aspect_prompt_token_id]*2)
                    
        #             aspect_prompt_bpes.append([self.senti_prompt_token_id])

        #             if i <(value-1):
        #                 aspect_prompt_bpes.append([self.sep_token_id])

        #         aspect_prompt_bpes.append([self.end_prompt_id])
        #         _aspect_prompt_bpes = list(chain(*aspect_prompt_bpes))
        #         aspect_prompts_tokens.append(_aspect_prompt_bpes.copy())
        # import ipdb; ipdb.set_trace()
        '''
        prompt_3: [image_features] + is + [image_caption]  
        + 'There is' <prompt> ([aspect_prompt_token]*len_1(最多为5) + of <<NEU>> or <<POS>> or <<NEG>> + 'sentiment' + <sep>)*n </prompt> + 'in'
        + [text]
        '''
        # if has_prompt:
        #     aspect_prompts_tokens = []     
        #     for index, value in enumerate(aspects_num):
        #         # aspect_prompts.append(self.begin_prompt + (self.aspect_prompt_token*2 + 'has' +  self.senti_prompt_token + 'sentiment' + self.sep_token) * value  +
        #         #                   self.end_prompt)
        #         aspect_prompt_bpes = [[self.begin_prompt_id]]
        #         if value == 1:
        #             _be = 'is'
        #         else:
        #             _be = 'are'
                
        #         there_bpes = self._base_tokenizer.tokenize('there',
        #                                                 add_prefix_space=True)
        #         there_bpes = self._base_tokenizer.convert_tokens_to_ids(there_bpes)
        #         aspect_prompt_bpes.append(there_bpes)

        #         _be_bpes = self._base_tokenizer.tokenize(_be,
        #                                                 add_prefix_space=True)
        #         _be_bpes = self._base_tokenizer.convert_tokens_to_ids(_be_bpes)
        #         aspect_prompt_bpes.append(_be_bpes)

        #         for i in range(value):
        #             aspect_prompt_bpes.append([self.aspect_prompt_token_id]*2)

        #             of_bpes = self._base_tokenizer.tokenize('of',
        #                                                     add_prefix_space=True)
        #             of_bpes = self._base_tokenizer.convert_tokens_to_ids(of_bpes)
        #             aspect_prompt_bpes.append(of_bpes)
                    
                    
        #             or_bpes = self._base_tokenizer.tokenize('or',
        #                                                     add_prefix_space=True)
        #             or_bpes = self._base_tokenizer.convert_tokens_to_ids(or_bpes)
                    
        #             aspect_prompt_bpes.append([self.neu_token_id])
        #             aspect_prompt_bpes.append(or_bpes)
        #             aspect_prompt_bpes.append([self.pos_token_id])
        #             aspect_prompt_bpes.append(or_bpes) 
        #             aspect_prompt_bpes.append([self.neg_token_id])

        #             senti_bpes = self._base_tokenizer.tokenize('sentiment',
        #                                                     add_prefix_space=True)
        #             senti_bpes = self._base_tokenizer.convert_tokens_to_ids(senti_bpes)
        #             aspect_prompt_bpes.append(senti_bpes)

        #             if i <(value-1):
        #                 aspect_prompt_bpes.append([self.sep_token_id])


        #         aspect_prompt_bpes.append([self.end_prompt_id])
        #         in_bpes =  self._base_tokenizer.tokenize('in',
        #                                                 add_prefix_space=True)
        #         in_bpes = self._base_tokenizer.convert_tokens_to_ids(in_bpes)
        #         aspect_prompt_bpes.append(in_bpes)
        #         _aspect_prompt_bpes = list(chain(*aspect_prompt_bpes))
        #         aspect_prompts_tokens.append(_aspect_prompt_bpes.copy())

        if image_text is not None:
            image_sentence = self.encode(image_text,
                                         add_special_tokens=False,
                                         return_tensors='pt',
                                         padding=True)
            # 这里是 换成Token 的形式
            image_ids = image_sentence['input_ids']
            image_attention_mask = image_sentence['attention_mask']
            
            image_caption_tokens, image_caption_mask = self.pad_tokens_with_maxlength(
                image_caption_tokens, max_len=20)

            # print("========================image_caption_tokens====================================")
            # print(image_caption_tokens)
            # print('the length of image_caption_tokens is {}'.format(image_caption_tokens.shape))
            # print(image_caption_mask)

            input_sentence_tokens, input_sentence_mask = self.pad_tokens(
                input_sentence_tokens)
            
            aspect_prompts_tokens, aspect_prompts_mask = self.pad_tokens_with_maxlength(aspect_prompts_tokens, max_len=40)


            if text_only:
                image_attention_mask = torch.zeros(image_ids.size())
                image_caption_mask = torch.zeros(image_caption_tokens.size())
            if not use_caption:
                image_caption_mask = torch.zeros(image_caption_tokens.size())
                image_attention_mask = image_attention_mask
            


            input_ids = torch.cat((image_ids, image_caption_tokens, aspect_prompts_tokens, input_sentence_tokens), 1)
            attention_mask = torch.cat(
                (image_attention_mask, image_caption_mask, aspect_prompts_mask, input_sentence_mask), 1)
        else:
            input_sentence_tokens, input_sentence_mask = self.pad_tokens(
                input_sentence_tokens)
            input_ids = input_sentence_tokens
            attention_mask = input_sentence_mask
        # import ipdb; ipdb.set_trace()
        encoded = {}
        encoded['input_ids'] = input_ids
        encoded['attention_mask'] = attention_mask
        # build mlm mask
        if sentence is not None:
            sentence_mask = torch.zeros(input_ids.size(), dtype=torch.bool)
            for index, value in enumerate(input_ids):
                start = (value == self.bos_token_id).nonzero(as_tuple=True)[0]
                end = (value == self.eos_token_id).nonzero(as_tuple=True)[0]
                sentence_mask[index, start + 1:end] = True
            encoded['sentence_mask'] = sentence_mask

        # build img mask
        if img_num is not None:
            encoded['img_mask'] = encoded['input_ids'] == self.img_feat_id
            image_mask = torch.zeros(input_ids.size(), dtype=torch.bool)
            for index, value in enumerate(input_ids):
                start = (value == self.begin_img_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_img_id).nonzero(as_tuple=True)[0]
                image_mask[index, start + 1:end] = True
            encoded['my_image_mask'] = image_mask
        # 此处的特殊标记id是 begin_img_id 和 end_img_id  为了防止可能的重名，换个名字

        # 为了截取 image 在文本模态上的嵌入表示， 和文本的sentence_mask一样，做同样的操作
        caption_mask = torch.zeros(input_ids.size(), dtype=torch.bool)
        if caption is not None:
            caption_mask = torch.zeros(input_ids.size(), dtype=torch.bool)
            for index, value in enumerate(input_ids):
                start = (value == self.begin_caption_id).nonzero(as_tuple=True)[0]
                end = (value == self.end_caption_id).nonzero(as_tuple=True)[0]
                caption_mask[index, start + 1:end] = True
        image_caption_valid = torch.tensor(image_caption_valid, dtype=torch.long)
        encoded['image_caption_valid'] = image_caption_valid
        encoded['image_caption_mask'] = caption_mask

        # 提取字幕和文本中的名词
        encoded['caption_nouns'] = batch_caption_nouns
        encoded['sentence_nouns'] = batch_sentence_nouns

        return encoded

    def encode_label(self, label, img_num=None):  #generate labels for MLM task

        # build text label
        if not isinstance(label, list):
            label = [label]

        label_split = [x.split() for x in label]
        label_tokens = []
        for split in label_split:
            word_bpes = [[self.bos_token_id], [self.mlm_token_id]]
            for word in split:
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.eos_token_id])
            _word_bpes = list(chain(*word_bpes))
            label_tokens.append(_word_bpes)
        input_ids, attention_mask = self.pad_tokens(label_tokens)

        output_shape = input_ids[:, 2:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(input_ids[:, 1:].shape,
                                        dtype=torch.long)
        decoder_attention_mask = torch.empty(input_ids[:, 1:].shape,
                                             dtype=torch.long)

        for i in range(labels.size(0)):
            labels[i] = input_ids[i][(input_ids[i] != self.bos_token_id)
                                     & (input_ids[i] != self.mlm_token_id)]
            decoder_input_ids[i] = input_ids[i][
                input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][
                input_ids[i] != self.eos_token_id]
        labels[(labels == self.pad_token_id) | (labels == self.begin_img_id) |
               (labels == self.end_img_id) | (labels == self.mlm_token_id) |
               (labels == self.img_feat_id)] = -100
        output = {
            'mlm_labels': labels,
            'mlm_decoder_input_ids': decoder_input_ids,
            'mlm_decoder_attention_mask': decoder_attention_mask
        }

        return output
    


    def encode_senti(self, sentis):  #generate label for MSP task
        senti_input_text = [
            self.bos_token + self.senti_token for i in range(len(sentis))
        ]
        senti_input_text = self.encode(senti_input_text,
                                       add_special_tokens=False,
                                       return_tensors='pt',
                                       padding=True)
        senti_decoder_input_ids = senti_input_text['input_ids']
        senti_decoder_attention_mask = senti_input_text['attention_mask']

        sentiment = []
        for senti in sentis:
            sentiment.append(senti)
            # else:
            #     raise ValueError('sentiment label error!!')
        output = {
            'senti_labels': torch.from_numpy(np.array(sentiment)),
            'senti_decoder_input_ids': senti_decoder_input_ids,
            'senti_decoder_attention_mask': senti_decoder_attention_mask
        }
        return output

    def encode_anp_dis(self, batch_size):
        ANP_input_text = [
            self.bos_token + self.ANP_token for i in range(batch_size)
        ]
        ANP_input_text = self.encode(ANP_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output = {}
        output['ANP_decoder_input_ids'] = ANP_input_text['input_ids']
        output['ANP_decoder_attention_mask'] = ANP_input_text['attention_mask']

        return output

    def encode_anp_generate(self, ANP_words):  #generate label for AOG task
        label_split = [x.split() for x in ANP_words]
        label_tokens = []
        for split in label_split:
            word_bpes = [[self.bos_token_id], [self.ANP_generate_token_id]]
            for word in split:
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.eos_token_id])
            _word_bpes = list(chain(*word_bpes))
            label_tokens.append(_word_bpes)
        input_ids, attention_mask = self.pad_tokens(label_tokens)

        output_shape = input_ids[:, 2:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(input_ids[:, 1:].shape,
                                        dtype=torch.long)
        decoder_attention_mask = torch.empty(input_ids[:, 1:].shape,
                                             dtype=torch.long)

        for i in range(labels.size(0)):
            labels[i] = input_ids[i][
                (input_ids[i] != self.bos_token_id)
                & (input_ids[i] != self.ANP_generate_token_id)]
            decoder_input_ids[i] = input_ids[i][
                input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][
                input_ids[i] != self.eos_token_id]

        labels[(labels == self.pad_token_id) | (labels == self.begin_img_id) |
               (labels == self.end_img_id) |
               (labels == self.ANP_generate_token_id) |
               (labels == self.img_feat_id)] = -100

        output = {
            'anp_generate_labels': labels,
            'anp_generate_decoder_input_ids': decoder_input_ids,
            'anp_generate_decoder_attention_mask': decoder_attention_mask
        }
        return output

    def encode_aesc(self, label, aesc_spans, aesc_max_len):
        # import ipdb; ipdb.set_trace()
        target_shift = len(self.mapping2targetid) + 2
        aesc_text = []
        masks = []
        gt_spans = []

        flag = True
        for text, span in zip(label, aesc_spans):
            span = sorted(span, key=cmp_to_key(cmp))
            word_bpes = [[self.begin_text_id]]
            for word in text.split():
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.end_text_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            # print("self.mapping2targetid is {}".format(self.mapping2targetid))
            '''
            self.mapping2targetid: {'AESC': 2, 'POS': 3, 'NEU': 4, 'NEG': 5} 
            '''
            cur_text = [
                0, self.mapping2targetid['AESC'], self.mapping2targetid['AESC']
            ]
            # print("====================cur_text is {}=============================".format(cur_text))
            mask = [0, 0, 0]
            gt = []
            for x in span:
                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1] - 1] + target_shift
                polarity = self.mapping2targetid[x[2]]
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                cur_text.append(polarity)
                gt.append((s_bpe, e_bpe, polarity))
                mask.append(1)
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)

            aesc_text.append(cur_text)
            gt_spans.append(gt)
            masks.append(mask)
        span_max_len = max([len(x) for x in aesc_text])
        for i in range(len(masks)):
            add_len = span_max_len - len(masks[i])
            masks[i] = masks[i] + [0 for ss in range(add_len)]
            aesc_text[i] = aesc_text[i] + [1 for ss in range(add_len)]
        
       
        output = {}
        output['labels'] = torch.tensor(aesc_text)
        output['masks'] = torch.tensor(masks)
        output['spans'] = gt_spans
        # 考虑增加前后的区别
        aspect_prompt_input_text = [
            self.bos_token + self.aspect_prompt_token + self.aspect_prompt_token for i in range(len(label))
        ]
        # aspect_prompt_input_text = [
        #     self.bos_token + self.aspect_prompt_token_front + self.aspect_prompt_token_end for i in range(len(label))
        # ]
        aspect_prompt_input_text = self.encode(aspect_prompt_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['aspect_prompt_decoder_input_ids'] = aspect_prompt_input_text['input_ids']
        output['aspect_prompt_decoder_attention_mask'] = aspect_prompt_input_text['attention_mask']
        
        senti_prompt_input_text = [
            self.bos_token + self.senti_prompt_token for i in range(len(label))
        ]
        senti_prompt_input_text = self.encode(senti_prompt_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['senti_prompt_decoder_input_ids'] = senti_prompt_input_text['input_ids']
        output['senti_prompt_decoder_attention_mask'] = senti_prompt_input_text['attention_mask']

        aspects_num_input_text = [
            self.bos_token + self.aspects_num_token for i in range(len(label))
        ]

        aspects_num_input_text = self.encode(aspects_num_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['aspects_num_decoder_input_ids'] = aspects_num_input_text['input_ids']
        output['aspects_num_decoder_attention_mask'] = aspects_num_input_text['attention_mask']
        

        # print("---------------------output is {}------------------".format(output))
        # print('++++++++++++++++++++++output[labels] is {}+++++++++++++++++++++++++++++'.format(output['labels']))
        return output

    def encode_ae_oe(self, label, aspect_spans,
                     opinion_spans):  #generate labels of AOE task
        target_shift = len(self.mapping2targetid) + 2
        ae_oe_text = []
        masks = []
        gt_spans = []

        for text, ae_span, oe_span in zip(label, aspect_spans, opinion_spans):
            word_bpes = [[self.begin_text_id]]
            for word in text.split():
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.end_text_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            cur_text = [
                0, self.mapping2targetid['AE_OE'],
                self.mapping2targetid['AE_OE']
            ]
            mask = [0, 0, 0]

            gt = []
            for x in ae_span:
                # print(x[0])
                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1]] + target_shift
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                gt.append((s_bpe, e_bpe))
                mask.append(1)
                mask.append(1)
            cur_text.append(self.mapping2targetid['SEP'])
            mask.append(1)
            for x in oe_span:
                # print(x[0])
                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1]] + target_shift
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                gt.append((s_bpe, e_bpe))
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)

            ae_oe_text.append(cur_text)
            masks.append(mask)
            gt_spans.append(gt)
        span_max_len = max(len(x) for x in ae_oe_text)
        for i in range(len(masks)):
            add_len = span_max_len - len(masks[i])
            masks[i] = masks[i] + [0 for ss in range(add_len)]
            ae_oe_text[i] = ae_oe_text[i] + [1 for ss in range(add_len)]
        output = {}
        output['labels'] = torch.tensor(ae_oe_text)
        output['masks'] = torch.tensor(masks)
        output['spans'] = gt_spans
        return output

    def encode_mrm(self, box_cls):
        mrm_input_text = [
            self.bos_token + self.mrm_token + self.img_feat * 36
            for i in range(len(box_cls))
        ]
        mrm_input_text = self.encode(mrm_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        mrm_decoder_input_ids = mrm_input_text['input_ids']
        mrm_decoder_attention_mask = mrm_input_text['attention_mask']

        output = {
            'mrm_labels': torch.from_numpy(np.array(box_cls)),
            'mrm_decoder_input_ids': mrm_decoder_input_ids,
            'mrm_decoder_attention_mask': mrm_decoder_attention_mask
        }
        return output

    def encode_twitter_ae(self, label, aspect_spans, ae_max_len):
        target_shift = len(self.mapping2targetid) + 2
        ae_text = []
        masks = []
        gt_spans = []
        for text, span in zip(label, aspect_spans):
            word_bpes = [[self.begin_text_id]]
            for word in text.split():
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.end_text_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            # self.all_cum_lens.append(cum_lens)
            # print(len(cum_lens), len(split))
            cur_text = [
                0, self.mapping2targetid['AE'], self.mapping2targetid['AE']
            ]
            mask = [0, 0, 0]
            # print(text)
            # print(len(cum_lens), len(text.split()))
            gt = []
            # print("文本", text)
            # print("总长度", len(cum_lens))
            for x in span:
                # print("首尾索引", x[0], x[1], x[1] - 1)
                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1] - 1] + target_shift
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                gt.append((s_bpe, e_bpe))
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)
            # cur_text = cur_text + [
            #     1 for i in range(ae_max_len - len(cur_text))
            # ]
            # mask = mask + [0 for i in range(ae_max_len - len(mask))]
            # print(cur_text)
            ae_text.append(cur_text)
            masks.append(mask)
            gt_spans.append(gt)
        span_max_len = max(len(x) for x in ae_text)
        for i in range(len(masks)):
            add_len = span_max_len - len(masks[i])
            masks[i] = masks[i] + [0 for ss in range(add_len)]
            ae_text[i] = ae_text[i] + [1 for ss in range(add_len)]
        output = {}
        output['labels'] = torch.tensor(ae_text)
        output['masks'] = torch.tensor(masks)
        output['spans'] = gt_spans
        # output['AE_masks'][:, 2] = 1
        # 考虑增加前后Token的区别
        aspect_prompt_input_text = [
            self.bos_token + self.aspect_prompt_token + self.aspect_prompt_token for i in range(len(label))
        ]
        # aspect_prompt_input_text = [
        #     self.bos_token + self.aspect_prompt_token_front + self.aspect_prompt_token_end for i in range(len(label))
        # ]
        aspect_prompt_input_text = self.encode(aspect_prompt_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['aspect_prompt_decoder_input_ids'] = aspect_prompt_input_text['input_ids']
        output['aspect_prompt_decoder_attention_mask'] = aspect_prompt_input_text['attention_mask']
        # 使用num的token
        aspects_num_input_text = [
            self.bos_token + self.aspects_num_token for i in range(len(label))
        ]

        aspects_num_input_text = self.encode(aspects_num_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['aspects_num_decoder_input_ids'] = aspects_num_input_text['input_ids']
        output['aspects_num_decoder_attention_mask'] = aspects_num_input_text['attention_mask']
        
        return output

    def encode_twitter_sc(self, label, aesc_spans, aesc_max_len):
        target_shift = len(self.mapping2targetid) + 2
        aesc_text = []
        masks = []
        gt_spans = []
        # print(len(opinion_spans))
        # print(len(self.all_cum_lens), len(opinion_spans))

        flag = True
        for text, span in zip(label, aesc_spans):
            span = sorted(span, key=cmp_to_key(cmp))
            word_bpes = [[self.begin_text_id]]
            for word in text.split():
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.end_text_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()

            # if flag:
            #     # print(word_bpes)
            #     print(cum_lens)
            #     flag = False
            cur_text = [
                0, self.mapping2targetid['SC'], self.mapping2targetid['SC']
            ]
            mask = [0, 0, 0]
            # print(text)
            # print(len(cum_lens), len(text.split()))
            gt = []
            for x in span:

                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1] - 1] + target_shift
                # if s_bpe >= cum_lens[-1] or e_bpe >= cum_lens[-1]:
                #     break
                polarity = self.mapping2targetid[x[2]]
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                cur_text.append(polarity)
                gt.append((s_bpe, e_bpe, polarity))
                mask.append(1)
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(0)
            # cur_text = cur_text + [
            #     1 for i in range(aesc_max_len - len(cur_text))
            # ]
            # mask = mask + [0 for i in range(aesc_max_len - len(mask))]
            # print(cur_text)
            aesc_text.append(cur_text)
            gt_spans.append(gt)
            masks.append(mask)
        span_max_len = max([len(x) for x in aesc_text])
        for i in range(len(masks)):
            add_len = span_max_len - len(masks[i])
            masks[i] = masks[i] + [0 for ss in range(add_len)]
            aesc_text[i] = aesc_text[i] + [1 for ss in range(add_len)]
            # masks[i].extend([0 for ss in range(add_len)])
            # aesc_text[i].extend([1 for ss in range(add_len)])

        output = {}
        # print(oe_text[0], len(oe_text))
        # for xx in oe_text:
        #     if xx == None:
        #         print('opinion shit!!!!!!!!!!!!!!!')
        # print(aesc_text[0])
        # print(masks[0])
        output['labels'] = torch.tensor(aesc_text)
        output['masks'] = torch.tensor(masks)
        output['spans'] = gt_spans

        senti_prompt_input_text = [
            self.bos_token + self.senti_prompt_token for i in range(len(label))
        ]
        senti_prompt_input_text = self.encode(senti_prompt_input_text,
                                     add_special_tokens=False,
                                     return_tensors='pt',
                                     padding=True)
        output['senti_prompt_decoder_input_ids'] = senti_prompt_input_text['input_ids']
        output['senti_prompt_decoder_attention_mask'] = senti_prompt_input_text['attention_mask']

        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)