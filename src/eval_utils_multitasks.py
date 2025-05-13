from copy import deepcopy

import torch
import torch.nn as nn


def eval(args, model, loader, metric, device):
    num_correct = 0
    model.eval()
    for i, batch in enumerate(loader):
        # Forward pass
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
        # import ipdb; ipdb.set_trace()  分布式需要修改如下 model -> model.module
        if args.world_size == 1:
            predict, predict_aspects_num, pseudo_loss = model.predict(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos,
                aspects_num=batch['aspects_num'],
                sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
                image_mask=batch['my_image_mask'],
                mlm_message=batch['MLM'],
                image_caption_valid=batch['image_caption_valid'],
                image_caption_mask=batch['image_caption_mask'],
                score=batch['score'],
                caption_nouns=batch['caption_nouns'],
                sentence_nouns=batch['sentence_nouns']
            )
        else:
            predict, predict_aspects_num, pseudo_loss = model.module.predict(
                input_ids=batch['input_ids'].to(device),
                image_features=list(
                    map(lambda x: x.to(device), batch['image_features'])),
                attention_mask=batch['attention_mask'].to(device),
                aesc_infos=aesc_infos,
                aspects_num=batch['aspects_num'],
                sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
                image_mask=batch['my_image_mask'],
                mlm_message=batch['MLM'],
                image_caption_valid=batch['image_caption_valid'],
                image_caption_mask=batch['image_caption_mask'],
                score=batch['score'],
                caption_nouns=batch['caption_nouns'],
                sentence_nouns=batch['sentence_nouns']
            )
        target_aspects_num = torch.tensor(batch['aspects_num']).to(predict_aspects_num.device)
        num_correct += torch.eq(predict_aspects_num, target_aspects_num).sum().float().item()

        # print('predict is {}'.format(predict))

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break
    aspects_num_eval_acc = num_correct / len(loader.dataset) * args.world_size
    res = metric.get_metric()
    model.train()
    return res, aspects_num_eval_acc


# 修正 ： 在测试阶段增加一个 MLM损失的修正，


def get_ctta_params(model):
    decoder_aspect_params = list(model.BART_model.prompt_decoder.parameters())  # 获取APD模块的内容
    params_to_tune = decoder_aspect_params
    # print("参数列表", decoder_aspect_params)
    return params_to_tune


def eval_with_pseudo_ctta(args, model, loader, metric, device, num_ctta_steps=1,
                          ctta_lr=1e-5):  # 添加 tokenizer, mlm_loss_module, num_ctta_steps, ctta_lr
    num_correct = 0
    # 现在先把MLM损失单独加载在测试部分，就不像CTTA任务那样操作先，

    # original_model_state = deepcopy(model.state_dict())  # 保存原始模型参数 这个部分只保留我们训练的参数即可。

    tuned_patameter_names = []
    # --- Pseudo-CTTA 自适应阶段 ---
    print("开始 Pseudo-CTTA 自适应...")
    model.train()  # 设置模型为训练模式，以进行自适应
    pseudo_loss = 0.0
    # --- 定义自适应阶段的优化器 (关键部分) ---
    params_to_tune = get_ctta_params(model)  # 调用函数获取用于自适应的参数 (见下文解释)
    optimizer = torch.optim.AdamW(params_to_tune, lr=ctta_lr)
    #
    # optimizer = torch.optim.AdamW(model.parameters(), lr=ctta_lr)
    for ctta_step in range(num_ctta_steps):  # Pseudo-CTTA 迭代步骤
        for i, batch in enumerate(loader):  # 遍历 *测试集* loader
            optimizer.zero_grad()
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
            if args.world_size == 1:
                predict, predict_aspects_num, pseudo_loss = model.predict(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(
                        map(lambda x: x.to(device), batch['image_features'])),
                    attention_mask=batch['attention_mask'].to(device),
                    aesc_infos=aesc_infos,
                    aspects_num=batch['aspects_num'],
                    sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
                    image_mask=batch['my_image_mask'],
                    mlm_message=batch['MLM'],
                    image_caption_valid=batch['image_caption_valid'],
                    image_caption_mask=batch['image_caption_mask'],
                    score=batch['score'],
                    caption_nouns=batch['caption_nouns'],
                    sentence_nouns=batch['sentence_nouns']
                )
            else:
                predict, predict_aspects_num, pseudo_loss = model.module.predict(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(
                        map(lambda x: x.to(device), batch['image_features'])),
                    attention_mask=batch['attention_mask'].to(device),
                    aesc_infos=aesc_infos,
                    aspects_num=batch['aspects_num'],
                    sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
                    image_mask=batch['my_image_mask'],
                    mlm_message=batch['MLM'],
                    image_caption_valid=batch['image_caption_valid'],
                    image_caption_mask=batch['image_caption_mask'],
                    score=batch['score'],
                    caption_nouns=batch['caption_nouns'],
                    sentence_nouns=batch['sentence_nouns']
                )
            pseudo_loss.backward()
            optimizer.step()
        print(f"Pseudo-Loss: {pseudo_loss.item():.4f}")

    print("Pseudo-CTTA 自适应完成.")

    # --- 自适应后的评估阶段 ---
    print("在 Pseudo-CTTA 后进行评估...")
    model.eval()  # 设置模型为评估模式，进行性能评估

    # metric.reset()  # 重置评估指标

    num_correct = 0  # 重置 aspect 数量准确率计数

    for i, batch in enumerate(loader):  # 再次遍历测试集 loader，进行评估
        # --- ABSA 任务的前向传播 (与原始 eval 函数相同) ---
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

        predict, predict_aspects_num, _ = model.module.predict(  # 评估阶段不需要 pseudo_loss
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos,
            aspects_num=batch['aspects_num'],
            sentence_mask=batch['sentence_mask'],
            image_mask=batch['my_image_mask'],
            mlm_message=batch['MLM'],
            image_caption_valid=batch['image_caption_valid'],
            image_caption_mask=batch['image_caption_mask'],
            score=batch['score'],
            caption_nouns=batch['caption_nouns'],
            sentence_nouns=batch['sentence_nouns']
        )
        target_aspects_num = torch.tensor(batch['aspects_num']).to(predict_aspects_num.device)
        num_correct += torch.eq(predict_aspects_num, target_aspects_num).sum().float().item()

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))

    aspects_num_eval_acc = num_correct / len(loader.dataset)
    res = metric.get_metric()

    # model.load_state_dict(original_model_state)  # 恢复模型参数到原始状态! 非常重要!

    print("模型参数已恢复到原始状态.")

    return res, aspects_num_eval_acc
