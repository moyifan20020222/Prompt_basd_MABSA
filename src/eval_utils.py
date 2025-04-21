import torch
import torch.nn as nn


def eval(args, model, loader, metric, device):
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
        # import ipdb; ipdb.set_trace()
        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos, 
            aspects_num=batch['aspects_num'],
            sentence_mask=batch['sentence_mask'],  # 用于实现文本部分的截取，完成我们的Prompt修正SPD模块的内容、
            image_mask=batch['my_image_mask'],
            mlm_message=batch['MLM']
        )
        # 返回结果为序列生成的结果。
        # print('predict is {}'.format(predict))

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res
