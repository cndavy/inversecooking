## Inverse Cooking: Recipe Generation from Food Images

Code supporting the paper:

*Amaia Salvador, Michal Drozdzal, Xavier Giro-i-Nieto, Adriana Romero.
[Inverse Cooking: Recipe Generation from Food Images. ](https://arxiv.org/abs/1812.06164)
CVPR 2019*


If you find this code useful in your research, please consider citing using the
following BibTeX entry:

```
@InProceedings{Salvador2019inversecooking,
author = {Salvador, Amaia and Drozdzal, Michal and Giro-i-Nieto, Xavier and Romero, Adriana},
title = {Inverse Cooking: Recipe Generation From Food Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

### Installation

This code uses Python 3.6 and PyTorch 0.4.1 cuda version 9.0.

- Installing PyTorch:
```bash
$ conda install pytorch=0.4.1 cuda90 -c pytorch
```

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Pretrained model

- Download ingredient and instruction vocabularies [here](https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl) 
and [here](https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl), respectively.
- Download pretrained model [here](https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt).

### Demo

You can use our pretrained model to get recipes for your images.

Download the required files (listed above), place them under the ```data``` directory, and try our demo notebook ```src/demo.ipynb```.

Note: The demo will run on GPU if a device is found, else it will use CPU.
# Recipe1M Dataset

## Layers

### layer1.json

```js
{
  id: String,  // unique 10-digit hex string
  title: String,
  instructions: [ { text: String } ],
  ingredients: [ { text: String } ],
  partition: ('train'|'test'|'val'),
  url: String
}
```

### layer2.json

```js
{
  id: String,   // refers to an id in layer 1
  images: [ {
    id: String, // unique 10-digit hex + .jpg
    url: String
  } ]
}
```

## Images

The images in each of the partitions, train/val/test, are arranged in a four-level hierarchy corresponding to the first four digits of the image id.

For example: `val/e/f/3/d/ef3dc0de11.jpg`

The images are in RGB JPEG format and can be loaded using standard libraries.
### Data

- Download [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download) (registration required)
- Extract files somewhere (we refer to this path as ```path_to_dataset```).
- The contents of ```path_to_dataset``` should be the following:
```
det_ingrs.json
layer1.json
layer2.json
images/
images/train
images/val
images/test
```

*Note: all python calls below must be run from ```./src```*
### Build vocabularies

```bash
$ python build_vocab.py --recipe1m_path path_to_dataset
```

### Images to LMDB (Optional, but recommended)

For fast loading during training:

```bash
$ python utils/ims2file.py --recipe1m_path path_to_dataset
```

If you decide not to create this file, use the flag ```--load_jpeg``` when training the model.

### Training

Create a directory to store checkpoints for all models you train
(e.g. ```../checkpoints``` and point ```--save_dir``` to it.)

We train our model in two stages:

1. Ingredient prediction from images

```bash
python train.py --model_name im2ingr --batch_size 150 --finetune_after 0 --ingrs_only \
--es_metric iou_sample --loss_weight 0 1000.0 1.0 1.0 \
--learning_rate 1e-4 --scale_learning_rate_cnn 1.0 \
--save_dir ../checkpoints --recipe1m_dir path_to_dataset
```

2. Recipe generation from images and ingredients (loading from 1.)

```bash
python train.py --model_name model --batch_size 256 --recipe_only --transfer_from im2ingr \
--save_dir ../checkpoints --recipe1m_dir path_to_dataset
```

Check training progress with Tensorboard from ```../checkpoints```:

```bash
$ tensorboard --logdir='../tb_logs' --port=6006
```

### Evaluation

- Save generated recipes to disk with
```python sample.py --model_name model --save_dir ../checkpoints --recipe1m_dir path_to_dataset --greedy --eval_split test```.
- This script will return ingredient metrics (F1 and IoU)

### License

inversecooking is released under MIT license, see [LICENSE](LICENSE.md) for details.



从图像到食谱，如何实现？

从图片中生成食谱需要同时理解组成食材和制作的过程（如切片、和其他材料搅拌等）。传统方法将这个问题视为检索任务，基于输入图片和数据集图片的相似度计算，将食谱从一个固定的数据集中检索出来。很明显，传统方法在数据集缺少某种食物制作方法的情况下就会失败。

有一种方法可以克服这一数据局限，即将图片到菜谱的问题视为一个条件生成任务。研究人员认为，与其直接从图片中获取菜谱，不如首先预测食物的材料，然后基于图像和食材生成食物制作方法。这样可以利用图片和食材的中间过程获取一些额外信息。

模型

模型主要由两部分构成，首先研究人员预训练一个图片编码器和一个食材解码器（ingredients decoder），提取输入图像的视觉特征来预测食材。然后训练一个食材编码器（ingredient encoder）和烹饪流程解码器（instruction decoder），根据输入图片的图像特征和已经预测到的食材，生成食物的名称和烹饪流程。

模型架构如下图所示：

食物图片变菜谱：这篇CVPR论文让人人都可以学习新料理
图 2：模型的结构。模型的输入是食物图片，输出的是烹饪方法序列，而中间一步是基于图像生成食材清单。

具体来讲，烹饪流程解码器使用了三种不同的注意力策略：



食物图片变菜谱：这篇CVPR论文让人人都可以学习新料理
图 3：烹饪流程解码器使用的注意力策略。Transformer 模型（a）中的注意力模块被替换成了三种不同的注意力模块（b-d），用于多种条件下的烹饪说明。

效果如何？

研究人员使用 Recipe1M [45] 数据集来训练和评估模型。该数据集包括从烹饪网站上爬取的 1,029,720 个食谱。在实验中，研究者仅使用了包含图片的食谱，并移除了使用少于两种食材或两道流程的食物。最终，实验使用了 252,547 个训练样本、54,255 个验证样本和 54,506 个测试样本。

研究人员对比了传统的检索方法和该研究提出的新方法，结果如下：

食物图片变菜谱：这篇CVPR论文让人人都可以学习新料理
表 3：基线方法和论文方法的对比。左图为 IoU 和 F1 分数，右图为食材在烹饪指南上的精确率和召回率。

研究人员还进行了用户测试。他们从测试集中随机选择了 15 张图片，让用户根据提供的图片选择 20 种食材，并写下可能图片对应的菜谱。为了减少人类任务的复杂度，研究人员提高食材使用频率的阈值，减少了食材的选择数量。

食物图片变菜谱：这篇CVPR论文让人人都可以学习新料理
表 4：用户测试。左图为基线方法、人类和论文方法判断食材的 IoU 和 F1 分数，右图为根据人类判断，这三种方法生成食谱的成功率。

实验结果说明，使用 AI 生成的食谱比检索方法生成的食谱效果更好。

这样的研究只是造福吃货吗？

这项研究通过对食物图片的研究，可以进一步猜测其食材和加工方式。这可以进一步方便人们学习新的食物制作、协助计算食物中每种成分的卡路里、创造新的菜谱。同时，该研究采用的方法可以进一步启发「根据图片预测长文本」的研究。

更何况，再也不用看着社交媒体上的美食流口水了。扫图出菜谱，人人都可以学着做~

参考链接：https://ai.facebook.com/blog/inverse-cooking/

https://www.reddit.com/r/MachineLearning/comments/c1tb5m/p_using_ai_to_generate_recipes_from_food_images/