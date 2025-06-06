**Position Detached DETR**
========
optimized detr from original detr and conditional detr.

# 参考(References)
* [ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR/blob/main/.github/convergence-curve.png)
* [DETR](https://github.com/facebookresearch/detr)


[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

# 背景
Detr 算法存在收敛较慢问题，按照其论文结果需要迭代500个ep才能完成，针对该问题后续研究者提出较多的有效方法，其中ConditionalDETR以较为简单直接的方式大幅提升了训练效率。这里参考后者的思路，在detr原有代码上进行对比试验，得出几个影响收敛速度的因素，并做了针对性修改。在单卡（4080s）上获得较好的收敛速度。  
**首先，通过对比试验，得到影响因素有如下几点，重要程度由高到低**：  
* query的position embedding是否显示参与训练；
* decoder交叉注意力计算时context与position是否进行分离，以及分离方式；
* 使用class 的cost和loss_coef系数；
* 分类loss使用；
* query的数量在一定程度上会影响训练结果；
