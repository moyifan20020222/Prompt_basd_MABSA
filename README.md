# Prompt_basd_MABSA

本部分先记录使用 以GMP为基础的代码上，增加Prompt Tuning 以提升模型在 MATE MASC JAMSA任务上的性能， 目前的方法已经较之原文有了部分性能的提升

不足：
原始论文的数据集的选取过于偏差，导致MATE任务的性能受到较大限制，进而影响了JMASA任务的性能， 目前需要在MASAD数据集上 验证 数据集中的Aspect的多样性是否会影响本篇提出方法的性能。

为此我们需要使用Blip模型重新处理Twitter15和17的数据集 以及 MASAD数据集，以保证数据格式和GMP代码的处理格式一致。

之后在twiiter数据集上，验证样本数量的阈值，检验我们的方法在何种程度下的 少样本会出现过拟合。

而如果我们的方法确实能提升MATE性能，则考虑将MASAD数据集的格式 往持续学习方面修改任务定义。


-----
4.25
增加关于图像与文本相关度的部分，通过BLIP-ITM标注，让字幕信息和文本信息做交叉注意力融合后学习相关度的计算，对于无法提取字幕的部分，用一个特定值表示。看看效果能提升吗。 可以生成字幕，但是缺失字幕没必要补充了，因为训练的样本并不足以完成这个内容，但是可以对于缺失字幕，用图像信息顶替。

--------
4.27
把情绪Prompt和Aspect各自的总Prompt生成中使用的融合部分同样扩展到 每一个具体Aspect的情绪Prompt的构建当中，验证提升效果 ，（确实是有用的，并且可以把GMP中的情绪Prompt变成对每一个Aspect都生成一个。）

------------
5.13
增加Spacy提取字幕和文本的名词集合，并重新在GMP给出的少样本上训练，目前已经在MASC上得到了较大的提升，目前在MATE上继续测试

------------------
5.19
--
7.4
为情绪部分添加对比学习，关于情绪三类别的对比学习，而在MABSA中添加句子的Aspect总Prompt与情绪的总Prompt的对比学习。
目前三个任务在 GMP 给出的Few-Shot 数据集下的最优值分别如下：（twitter17所需的Prompt池大小更大一些）(best_dev_test)
如果以最优测试集，所有任务的性能还会更高一些
| MASC任务       | 准确率（%） | 训练时间（分） | F1(%)       |
|:--------------|------------:|:----------------:|---------------|
| Twitter15     |  70.25  |  1h  | 63.26 |
| Twitter17     |  71.25  |  1h  | 69.31 |


| MATE任务       | Precision（%） | Recall（%） | F1 (%)         |
|:--------------|------------:|:----------------:|---------------|
| Twitter15     |  77.24  |  73.00  | 75.06 |
| Twitter17     |  80.91  |  81.04  | 80.97 |


| MABSA任务       | Precision（%） | Recall（%） | F1 (%)         |
|:--------------|------------:|:----------------:|---------------|
| Twitter15     |  53.83  |  50.14  | 51.92 |
| Twitter17     |  56.93  |  54.62  | 55.75 |


使用的模型：
本文使用的视觉模型为vit_base_patch32_224 字幕模型为BLIP 相关度部分为 BLIP-ITM 三个模型可以在Huggingface下载
也可以在镜像网站下载：
[https://hf-mirror.com/models?sort=trending&search=vit_base_patch32_224](https://hf-mirror.com/google/vit-base-patch32-224-in21k)
https://hf-mirror.com/Salesforce/blip-image-captioning-base
https://hf-mirror.com/Salesforce/blip-itm-base-coco
本文使用的数据集的下载：
https://drive.google.com/drive/folders/1NgPH5xhz5dF-Zwxe-8CjjsgQJ5VaQ8KL?usp=sharing
但原始部分缺失了部分图像信息，所以需要单独去找完整的Twitter数据集：
IJCAI2019_data
