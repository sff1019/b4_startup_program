# 有名なネットワークの評価・比較

## 目次
1. [概要](#概要)
1. [用いたネットワークの種類](#用いたネットワークの種類)
1. [結果](#結果)

## 概要

LeNet，MobileNet，ResNetなど有名なネットワークを使ってCifar10のデータセットを評価した．

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


## 結果


## 参照

- Mark Sandler; Andrew Howard; Menglong Zhu; Andrey Zhmoginov; Liang-Chieh Chen et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications 

