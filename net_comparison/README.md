# 有名なネットワークの評価・比較

## 目次
1. [概要](#概要)
1. [用いたデータセット](#用いたデータセット)
1. [用いたネットワークの種類](#用いたネットワークの種類)
  - LeNet-5
  - MobileNet
  - MobileNetV2
  - VGG16
  - ResNet50
1. [結果](#結果)

## 概要

LeNet，MobileNet，ResNetなど有名なネットワークを使ってCifar10のデータセットを評価した．

## 用いたデータセット

今回，Cifar10を用いて実験を行なった．  
また，正規化を行う際には[ここの](https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py)数値を参考にした．

## 用いたネットワークの種類

### LeNet-5
[こちら](https://engmrk.com/lenet-5-a-classic-cnn-architecture/)を参考に実装

### MobileNet

Depthwise Separable Covolutionと呼ばれる畳み込みニューラルネットワークを採用．Googleが2017，2018に発表した論文．  
精度を追求すると，モデルが膨張してしまう問題を解決．  

畳み込みの計算をする際に，チャンネル数，高さ，そして横幅を分割し，計算することによって，通常の畳み込みと同じ計算をするが，1つ1つの計算量が小さくなるため時間が短縮される．  

<p align='center'>
  <img height='300' src='./assets/mobilenet_desc.png?raw=true'>
</p>

モデル自体は精度の改良ではなく，計算量の削減に特化したモデルである．しかし，精度は当時存在した他のモデルに負け劣らずの数値であった．

今回は，[論文](https://arxiv.org/abs/1704.04861)を参考に実装を行なった．以下，ネットワークの構造を抽出した

<p align='center'>
  <img height='300' src='./assets/MobileNet_structure.png?raw=true'>
</p>

### MobileNet V2

<p align='center'>
  <img height='300' src='./assets/mobilenetv2_details.svg'>
</p>


MobileNet V2はMobileNetの改良版である．  

MobileNetと同様に，[論文](https://arxiv.org/pdf/1801.04381.pdf)を参考に実装を行なった．  

MobileNetよりも精度は少し高く，かつ実行速度は早い．


### VGG16

<p align='center'>
  <img height='300' src='./assets/vgg16_details.svg'>
</p>


### ResNet50

ResNetとはDeep Residual Networksの略であり，Microsoft Researchから2015に提案された最大1000層以上のニューラルネットワークである．  

ResNetはShortcut Connectionを導入している．層が深いニューラルネットワークは，精度が学習できる最大値を越えると，層が進むにつれ逆に精度が下がるという問題がある．これは，過学習で引き起こされる問題ではなく，長く研究がされていた．  
そこで，ResNetで提案された手法として，「残差関数」というものがある．  
残差関数とは，本来求める最適な出力と，層の入力との差分である．  

Shortcut Connectionを導入することによって，深い層でも性能は保たれ，高い精度を得ることができるようになった．


## 結果


## 参照

- Mark Sandler; Andrew Howard; Menglong Zhu; Andrey Zhmoginov; Liang-Chieh Chen et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861

- Mark Sandler; Andrew Howard; Menglong Zhu; Andrey Zhmoginov; Liang-Chieh; Chen et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. 

- [Source code for torchvision.models.vgg](https://pytorch.org/docs/0.4.0/_modules/torchvision/models/vgg.html)

- [Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification: From Microsoft to Facebook [Part 1]](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)
