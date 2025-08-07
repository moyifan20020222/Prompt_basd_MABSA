import argparse
import json
import os
from collections import Counter
from datetime import datetime
from torch import optim, nn
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import random
from src.data.collation_for_prompt_multitasks import Collator
from src.data.dataset_for_prompt import MVSA_Dataset, Twitter_Dataset
from src.data.tokenization_new_for_generated_prompt_multitasks import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model_for_generated_dual_prompts_multitasks_Aspect import MultiModalBartModel_AESC
from src.model.model_for_prompt import MultiModalBartModelForPretrain
from src.training_multitasks import fine_tune
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
from src.model.metrics import AESCSpanMetric
from src.model.generater_for_generated_prompt_multitasks import SequenceGeneratorModel
import src.eval_utils_multitasks as eval_utils
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# from thop import profile
from src.model.modules_for_prompt_multitasks import image_model_name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 尽量抑制TensorFlow的日志
# os.environ['CUDA_VISIBLE_DEVICES'] = ''   # 告诉TensorFlow没有可见的GPU。注意是空字符串，有时比'-1'更有效。
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 防止TensorFlow预分配所有GPU内存

import logging

# 设置日志级别为WARNING，这样INFO级别的日志就不会显示
logging.basicConfig(level=logging.WARNING)
# 特别设置fastNLP相关模块的日志级别
for logger_name in logging.root.manager.loggerDict:
    if 'fastNLP' in logger_name or 'instantiator' in logger_name:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_parameter_number(model):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:  # 只统计可训练的参数
    #         print(f"Name: {name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(rank, args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    tb_writer = None
    add_name = ''
    log_dir = os.path.join(args.log_dir, timestamp + add_name)

    # make log dir and tensorboard writer if log_dir is specified
    if rank == 0 and args.log_dir is not None or not args.distributed:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=(rank == 0))

    # make checkpoint dir if not exist
    if args.is_check == 1 and not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    logger.info("The vision model use: {}".format(image_model_name))
    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:{}".format(rank))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    tokenizer = ConditionTokenizer(args=args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(
            json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    bos_token_id = 0  # 因为是特殊符号
    eos_token_id = 1

    # import ipdb; ipdb.set_trace()

    if args.checkpoint:
        pretrain_model = MultiModalBartModelForPretrain.from_pretrained(
            args.checkpoint,
            config=bart_config,
            bart_model=args.bart_model,
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False)
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        seq2seq_model.encoder.load_state_dict(
            pretrain_model.encoder.state_dict())
        seq2seq_model.decoder.load_state_dict(
            pretrain_model.span_decoder.state_dict())
        model = SequenceGeneratorModel(seq2seq_model, seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
    else:
        print('++++++++++++++++++ No Pretrain ++++++++++++++++++++++++++++++++')
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        model = SequenceGeneratorModel(seq2seq_model, seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
        # model = MultiModalBartModel_AESC(bart_config, args.bart_model,
        #                                  tokenizer, label_ids)

    model.to(device)

    parameters = get_parameter_number(model)  ##{'Total': 169351685, 'Trainable': 169351685}
    print(parameters)
    logger.info("The parameters of model use: {}".format(parameters))
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    scaler = GradScaler() if args.amp else None

    epoch = 0
    logger.info('Loading data...')
    collate_aesc = Collator(
        args.task,
        tokenizer,
        mlm_enabled=args.mlm_enabled,
        senti_enabled=False,
        ae_enabled=False,
        oe_enabled=False,
        aesc_enabled=True,
        anp_enabled=False,
        max_img_num=args.num_image_tokens,
        has_prompt=args.has_prompt,
        text_only=args.text_only,
        use_caption=args.use_caption)

    train_dataset = Twitter_Dataset(args.dataset[0][1], split='train', image_model_name=image_model_name)
    dev_dataset = Twitter_Dataset(args.dataset[0][1], split='dev', image_model_name=image_model_name)
    test_dataset = Twitter_Dataset(args.dataset[0][1], split='test', image_model_name=image_model_name)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              collate_fn=collate_aesc)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=collate_aesc)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_aesc)

    callback = None
    metric = AESCSpanMetric(eos_token_id,
                            num_labels=len(label_ids),
                            conflict_id=-1)
    model.train()
    start = datetime.now()
    best_dev_res = None
    best_dev_test_res = None
    best_test_res = None
    # res_dev = eval_utils.eval(model, dev_loader, metric, device)

    pos_num = 0
    neg_num = 0
    neu_num = 0
    for i, batch in enumerate(dev_loader):
        aesc_infos = {
            key: value
            for key, value in batch['AESC'].items()
        }
        sentiment_label_ids = [3, 4, 5]

        # 调用函数统计
        sentiment_counts = get_sentiment_proportions(aesc_infos['spans'], sentiment_label_ids)

        # 遍历情感ID列表，获取每个ID的计数
        pos_num += sentiment_counts[3]
        neu_num += sentiment_counts[4]
        neg_num += sentiment_counts[5]
    print("验证集各个情绪数量 pos neu neg", pos_num, neu_num, neg_num)

    pos_num = 0
    neg_num = 0
    neu_num = 0
    for i, batch in enumerate(test_loader):
        aesc_infos = {
            key: value
            for key, value in batch['AESC'].items()
        }
        sentiment_label_ids = [3, 4, 5]

        # 调用函数统计
        sentiment_counts = get_sentiment_proportions(aesc_infos['spans'], sentiment_label_ids)

        # 遍历情感ID列表，获取每个ID的计数
        pos_num += sentiment_counts[3]
        neu_num += sentiment_counts[4]
        neg_num += sentiment_counts[5]
    print("测试集各个情绪数量 pos neu neg", pos_num, neu_num, neg_num)

    sentiment_counts = Counter()
    sentiment_label_ids = [3, 4, 5]
    for i, batch in enumerate(train_loader):
        aesc_infos = {
            key: value
            for key, value in batch['AESC'].items()
        }
        for sample_result in aesc_infos['spans']:  # 遍历 batch 中的每个样本
            for aspect_senti_pair in sample_result:  # 遍历样本中的每个三元组
                sentiment_id = aspect_senti_pair[2]  # 情感 ID 是三元组的第三个元素
                if sentiment_id in sentiment_label_ids:  # 确保是有效的情感 ID
                    sentiment_counts[sentiment_id] += 1
    print("总数", sentiment_counts)
    while epoch < args.epochs:
        # --------- 新增分布式架构部分：
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        fine_tune(epoch=epoch,
                  model=model,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  metric=metric,
                  optimizer=optimizer,
                  args=args,
                  device=device,
                  logger=logger,
                  callback=callback,
                  log_interval=1,
                  tb_writer=tb_writer,
                  tb_interval=1,
                  scaler=scaler,
                  # sentiment_counts=sentiment_counts
                  )

        print('test!!!!!!!!!!!!!!')
        if (epoch + 1) % args.eval_every == 0:
            # train_dev = eval_utils.eval(model, train_loader, metric, device)
            res_dev, dev_aspects_num_acc = eval_utils.eval(args, model, dev_loader, metric, device)
            res_test, test_aspects_num_acc = eval_utils.eval(args, model, test_loader, metric,
                                                             device)
            if rank == 0 or not args.distributed:
                logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}, dev_aspects_num_acc: {:.4f}'.format(
                    res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f'], dev_aspects_num_acc))

                logger.info('TEST  aesc_p:{} aesc_r:{} aesc_f:{}, test_aspects_num_acc: {:.4f}'.format(
                    res_test['aesc_pre'], res_test['aesc_rec'],
                    res_test['aesc_f'], test_aspects_num_acc))

            save_flag = False
            if best_dev_res is None:
                best_dev_res = res_dev
                best_dev_test_res = res_test

            else:
                if best_dev_res['aesc_f'] < res_dev['aesc_f']:
                    best_dev_res = res_dev
                    best_dev_test_res = res_test

            if best_test_res is None:
                best_test_res = res_test
                save_flag = True
            else:
                if best_test_res['aesc_f'] < res_test['aesc_f']:
                    best_test_res = res_test
                    save_flag = True

            if args.is_check == 1 and save_flag:
                current_checkpoint_path = os.path.join(checkpoint_path,
                                                       args.check_info)
                model.seq2seq_model.save_pretrained(current_checkpoint_path)
                print('save model!!!!!!!!!!!')
        epoch += 1

    logger.info("Training complete in: " + str(datetime.now() - start),
                pad=True)
    logger.info('---------------------------')
    logger.info('BEST DEV:-----')
    logger.info('BEST DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_res['aesc_pre'], best_dev_res['aesc_rec'],
        best_dev_res['aesc_f']))

    logger.info('BEST DEV TEST:-----')
    logger.info('BEST DEV--TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_test_res['aesc_pre'], best_dev_test_res['aesc_rec'],
        best_dev_test_res['aesc_f']))

    logger.info('BEST TEST:-----')
    logger.info('BEST TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_test_res['aesc_pre'], best_test_res['aesc_rec'],
        best_test_res['aesc_f']))
    # 清理进程组
    # dist.destroy_process_group()
    # if not args.cpu:
    #     cleanup_process()


def get_sentiment_proportions(batch_results, sentiment_label_ids):
    """
    统计一个 batch 中每个情感 ID 的出现比例。

    :param batch_results: 一个 batch 的 Aspect-Sentiment 三元组列表。
                          例如：[[[s,e,senti_id], ...], [[s,e,senti_id], ...], ...]
    :param sentiment_label_ids: list, 包含所有有效情感 ID 的列表，例如 [3, 4, 5]。
    :return: Counter, 每个情感 ID 及其出现次数。
    """
    sentiment_counts = Counter()
    for sample_result in batch_results:  # 遍历 batch 中的每个样本
        for aspect_senti_pair in sample_result:  # 遍历样本中的每个三元组
            sentiment_id = aspect_senti_pair[2]  # 情感 ID 是三元组的第三个元素
            if sentiment_id in sentiment_label_ids:  # 确保是有效的情感 ID
                sentiment_counts[sentiment_id] += 1
    return sentiment_counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        metavar=('DATASET_NAME', 'DATASET_PATH'),
                        required=True,
                        help='')
    # required

    parser.add_argument('--checkpoint_dir',
                        required=True,
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='data/bart-base',
                        type=str,
                        help='bart pretrain model')
    # path
    parser.add_argument(
        '--log_dir',
        default=None,
        type=str,
        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config',
                        default=None,
                        type=str,
                        help='path to load model config')
    parser.add_argument('--text_only', type=str, default=False, help='whether only use text')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='name or path to load weights')
    parser.add_argument('--lr_decay_every',
                        default=4,
                        type=int,
                        help='lr_decay_every')
    parser.add_argument('--lr_decay_ratio',
                        default=0.8,
                        type=float,
                        help='lr_decay_ratio')
    # training and evaluation
    parser.add_argument('--epochs',
                        default=35,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    # dropout
    parser.add_argument(
        '--dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the transformer. This overwrites the model config')
    parser.add_argument(
        '--classif_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the classification layers. This overwrites the model config'
    )
    parser.add_argument(
        '--attention_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the attention layers. This overwrites the model config'
    )
    parser.add_argument(
        '--activation_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the activation layers. This overwrites the model config'
    )

    # hardware and performance
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--gpu_num',
                        default=1,
                        type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port',
                        type=str,
                        default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='#workers for data loader')
    parser.add_argument('--max_len', type=int, default=10, help='max_len')
    parser.add_argument('--num_image_tokens', type=int, default=2, help='the length of image_tokens')
    parser.add_argument('--max_len_a',
                        type=float,
                        default=0.6,
                        help='max_len_a')

    parser.add_argument('--bart_init',
                        type=int,
                        default=1,
                        help='use bart_init or not')

    parser.add_argument('--check_info',
                        type=str,
                        default='',
                        help='check path to save')
    parser.add_argument('--is_check',
                        type=int,
                        default=0,
                        help='save the model or not')
    parser.add_argument('--task', type=str, default='AESC', help='task type')

    parser.add_argument('--has_prompt', action='store_true', default=True, help='whether has prompt')
    parser.add_argument('--use_generated_aspect_prompt', action='store_true', default=True,
                        help='whether use the generated aspect prompt')
    parser.add_argument('--use_generated_senti_prompt', action='store_true', default=True,
                        help='whether use the generated sentiment prompt')
    parser.add_argument('--use_different_senti_prompt', type=str, default=True,
                        help='whether use different prompt for different aspects in an instance')
    parser.add_argument('--use_different_aspect_prompt', action='store_true', default=True,
                        help='whether use different prompt for different aspects in an instance')
    parser.add_argument('--use_caption', type=str, default=True, help='whether use image caption')
    parser.add_argument('--use_multitasks', action='store_true', default=True, help='whether use multitasks')
    parser.add_argument('--loss_lambda', default=0.1, type=float, help='the weight of aspect_num classification loss')
    # 新增部分：1、情绪Prompt池大小 2、用于更新Prompt池部分的损失函数， 包括多样性损失权重和正则化损失权重
    parser.add_argument('--Prompt_Pool_num', type=int, default=8, help="The number of PromptPool")
    parser.add_argument('--diversity_loss_weight', type=float, default=0.1, help='The weight of diversity_loss')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0001, help='The weight of l2_reg')
    # 新增关于mlm损失的部分：
    parser.add_argument('--mlm_enabled', type=str, default=True, help='MLM Loss in CTTA')

    # 添加分布式训练参数
    # parser.add_argument('--distributed', action='store_true', default=True, help='是否使用分布式训练')
    # parser.add_argument('--world_size', type=int, default=3, help='使用的GPU数量')
    # # 这个部分是服务器的地址和 端口，不同服务器 请改变这两个部分
    # # parser.add_argument('--dist_url', default='tcp://10.154.45.17:12355', help='分布式训练的URL')
    # parser.add_argument('--dist_url', type=str, default='tcp://10.200.1.3:12355', help='分布式训练的URL')
    # parser.add_argument('--dist_backend', default='nccl', type=str, help='分布式训练的后端')
    # parser.add_argument('--master_port', type=int, default=12355, help='端口')
    # 是否是少样本
    parser.add_argument('--is_few_shot', type=str, default=True, help='当前是否是少样本数据集')

    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')

    return args


if __name__ == '__main__':
    args = parse_args()

    # mp.spawn(main, args=(args, ), nprocs=args.gpu_num, join=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    # if args.distributed:
    #     # 使用torch.multiprocessing启动多进程
    #     # 使用spawn方法启动进程，确保每个进程有独立的环境
    #     mp.set_start_method('spawn', force=True)
    #     mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    # else:
    #     # 单进程模式
    main(0, args)
