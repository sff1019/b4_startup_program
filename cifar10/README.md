# CIFAR10の学習・評価（Pytorch）

## 目次

1. [用いた最適化手法について](#用いた最適化手法について)
    - SGD
    - Momentum SGD
    - AdaGrad
    - RMSprop
    - Adam
2. [MLPを用いた実装](#mlpを用いた実装)
    - [考察](#mlpの考察)
      - [CPUにおける実行](#cpuにおける実行)
      - [GPUにおける実行](#gpuにおける実行)
3. [CNNを用いた実装](#cnnを用いた実装)
    - [考察](#cnnの考察)
      - [CPUにおける実行](#cpuにおける実行)
      - [GPUにおける実行](#gpuにおける実行)

---

## 用いた最適化手法について

今回，5つの最適化手法を用いて，実験・考察を行なった．  
最もシンプルなSGDから，改良を重ねたAdamを通して，精度や実行時間をみた．  
以下，それぞれの手法の簡易的説明，そして式を記す．

### SGD

<img height='50' src='./assets/sgd.png'>

勾配降下法と呼ばれる手法であり，多くはミニバッチで行われるため，確率的勾配降下法とも呼ばれる．  
パラメタの勾配を求め，それらを用いて，最適化を行う最もシンプルな方法である．  
しかし，収束の不安定性・遅さから高次元の問題で使われることはない．

### Momentum SGD

<img height='100' src='./assets/momentum_sgd.png'>

物理のモーメンタムを用いた手法．vは速度を表し，ボールが池面の傾斜を転がるように動く．  
SGDに比べて，x軸方向に受ける力が小さく，y軸方向には受ける力が大きいが，速度は安定しないので，SGDに比べてx軸方向へ早く近づくことができる．

### AdaGrad

<img height='100' src='./assets/adagrad.png'>

AdaGradは，パラメタの要素ごとに適応的に学習係数を調整しながら学習を行う手法である．  
hはこれまで経験した勾配の値を2乗和として保持することによって，学習のスケールを調整．

### RMSprop

<img height='100' src='./assets/rmsprop.png'>

RMSpropはAdaGradを改良したアルゴリズムである．  
AdaGradは学習率が0に十分近くなってしまうと，まだ坂があったとしてもほとんど更新されなくなってしまうという問題があった．  
そこで提案されたRMSpropは初期の影響がαに応じて指数的に減衰する．

### Adam
<img height='200' src='./assets/adam.png'>

RMSpropの改良版である．  
勾配に関しては，RMSpropのように，指数的減衰をするが，これに加え過去の勾配の指数関数的減数平均を保持する．

---

## MLPを用いた実装

[Pytorchのドキュメント](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)を参考に実装

<p align='center'>
  <img height='300' src='./assets/chainer_mnist_mlp_detailed.png?raw=true'>
</p>

入力層: 100ノード  
中間層: 100ノード  
出力層: 10ノード

### MLPの考察

#### CPUにおける実行

#### GPUにおける実行

---

## CNNを用いた実装

<p align='center'>
  <img height='300' src='./assets/chainer_mnist_cnn_detailed.png?raw=true'>
</p>

入力層: 100ノード  
中間層: 100ノード  
出力層: 10ノード

### CNNの考察

#### CPUにおける実行

#### GPUにおける実行

ノード：turing  
ノード数：1  
CUDA：10.1

---
