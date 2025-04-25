from datetime import datetime

import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import src.model.utils as utils
import src.eval_utils as eval_utils
# from src.utils import TaskType
import torch

def pretrain(task_list,
             epoch,
             model,
             train_loaders,
             optimizer_dict,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):

    # assert len(task_list) == len(train_loaders)

    total_step = len(train_loaders[0])
    model.train()
    total_loss = 0

    start_time = datetime.now()

    for i, batchs in enumerate(zip(*train_loaders)):
        # Forward pass
        with autocast(enabled=args.amp):
            loss_all = []
            total_loss = 0
            for cnt, task in enumerate(task_list):
                batch = batchs[cnt]
                # print(batch.keys())
                if task == 'Sentiment':
                    loss, prelogits = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        })
                else:
                    loss = model.forward(
                        task,
                        input_ids=batch['input_ids'].to(device),
                        image_features=list(
                            map(lambda x: x.to(device),
                                batch['image_features'])),
                        attention_mask=batch['attention_mask'].to(device),
                        mlm_infos={
                            key: value.to(device)
                            for key, value in batch['MLM'].items()
                        } if 'MLM' in batch else None,
                        mrm_infos={
                            key: value
                            for key, value in batch['MRM'].items()
                        } if 'MRM' in batch else None,
                        senti_infos={
                            key: value.to(device)
                            for key, value in batch['Sentiment'].items()
                        } if 'Sentiment' in batch else None,
                        ANP_infos={
                            key: value.to(device)
                            for key, value in batch['ANP'].items()
                        } if 'ANP' in batch else None,
                        ANP_generate_infos={
                            key: value.to(device)
                            for key, value in batch['ANP_generate'].items()
                        } if 'ANP_generate' in batch else None,
                        ae_oe_infos={
                            key: value
                            for key, value in batch['AE_OE'].items()
                        } if 'AE_OE' in batch else None)

                # print(loss.dtype)
                loss_all.append(loss)
                optimizer_dict.zero_grad()

                loss.backward()
                optimizer_dict.step()

            for k, v in zip(task_list, loss_all):
                print(k + ':', v.item(), end=' ')
            print()
        # Backward and optimize

        if logger is not None and i % log_interval == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}]'.format(
                epoch + 1, args.epochs, i + 1, total_step))
            loss_text = ' '.join(
                [k + ':' + str(v.item()) for k, v in zip(task_list, loss_all)])
            logger.info(loss_text + '\n')


def fine_tune(epoch,
              model,
              train_loader,
              test_loader,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)
    model.train()
    total_loss = 0
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    start_time = datetime.now()
    num_correct = 0

    for i, batch in enumerate(train_loader):
        # Forward pass
        # ---- 第一部分是 准备数据，获取的是数据集分词
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
            # import ipdb; ipdb.set_trace()
            # print("+++++++++++++++++++++++++++++++++++++++++++")
            # print('aesc_infos is {}'.format(aesc_infos))
        with autocast(enabled=args.amp):
            # print("模型的输入信息-inputs_id", batch['input_ids'].shape)
            # print("模型的输入图像信息-image_features", list(
            #         map(lambda x: x.to(device), batch['image_features'])))
            # print("模型输入附加信息", aesc_infos)
            # ASC任务包括了 labels （三元组结果） mask labels的 掩码 spans 三元组提取的结果
            # senti_prompt_decoder_input_ids 和 senti_prompt_decoder_attention_mask 指示最后生成的情绪Prompt
            # print("当前使用的模型是:", model)
            # print("image_caption_mask:", batch['image_caption_mask'].size())
            # print("image_caption_valid:", batch['image_caption_valid'])

            loss, predict_aspects_num, pseudo_loss = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos, 
                aspects_num=batch['aspects_num'],
                sentence_mask=batch['sentence_mask'],  # 识别句子位置
                image_mask=batch['my_image_mask'],  # 识别图片token部分
                mlm_message=batch['MLM'],  # 用于计算MLM损失的信息
                image_caption_valid=batch['image_caption_valid'],
                image_caption_mask=batch['image_caption_mask'],
                score=batch['score']
            )
            target_aspects_num = torch.tensor(batch['aspects_num']).to(predict_aspects_num.device)
            if args.task == 'twitter_sc':
                print("在只识别情绪的任务下，aspect数量已知，所以正确数目固定为1")
            num_correct += torch.eq(predict_aspects_num, target_aspects_num).sum().float().item()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # 这不是应该训练完一轮再输出吗，怎么会在这。
        # train_acc = num_correct / len(train_loader.dataset)
        # print('The accuracy of aspects_num is {:.4f} !!!!'.format(train_acc))
        # Backward and optimize

        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        utils.set_lr(optimizer, liner_warm_rate * args.lr)

        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()
        scheduler.step()

    train_acc = num_correct / len(train_loader.dataset)
    print('The accuracy of aspects_num is {:.4f} !!!!'.format(train_acc))
