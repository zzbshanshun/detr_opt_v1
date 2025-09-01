**Position Detached DETR**
========
optimized detr from original detr and conditional detr.

# 参考(References)
* [ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR/tree/main)，下文简称CDter；
* [DETR](https://github.com/facebookresearch/detr)
* [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR)，下文简称DAB；


[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

# 1.背景
Detr 算法存在收敛较慢问题，按照其论文结果需要迭代500个ep才能完成，针对该问题后续研究者提出较多的有效方法，其中ConditionalDETR以较为简单直接的方式大幅提升了训练效率。这里参考后者的思路，在detr原有代码上进行对比试验，得出几个影响收敛速度的因素，并做了针对性修改。在单卡（4080s）上获得较好的收敛速度。  

# 2.收敛影响因素
> **通过对比试验，得到如下几点影响因:**：
* query的position embedding使用方式，主要是是否参与显示的box预测；
* decoder交叉注意力计算时context与position是否进行分离，以及分离方式；
* loss计算中class 部分的权重；
* 分类loss使用；
* query的数量；
> CDetr和DAB中的已有方法中改进点：
* query position embedding改进：
  * CDetr中基于原有embedding对每个query的中心坐标预测，并将其编码后作为解码器中每层position embedding输入；最有使用预测中心作为output预测的偏移矫正；
  * DAB中采用直接生成参考点box=(x,y,w,h)，在解码器中通过参考点编码生成query的position embedding，并通过每层output输出进行微调，作为下一层输入；
* decoder交叉注意力计算改进：
  * DAB中沿用了CDetr的方式，将context与position以concat方式进行解耦计算；
* loss修改：
  * CDetr和DAB相同都采用Sigmoid Focal Loss替换原有softmax；
* query数量：
  * 两者均将query数量从原有的100提升到300，配合focal loss对训练收敛有一定程度的优化效果；


* 解码器中，要使用每层的output输出进行box预测，其预测分支尽量避免对output计算梯度；
