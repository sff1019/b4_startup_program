# 新入生スタートアッププログラム

## 項目
- [x] 横田研クラスタを利用できるようにする
- [x] Pythonの環境構築
- [x] Pythonの基本的な文法を学ぶ
- [x] 深層学習の基本を学ぶ
- [x] MNISTの学習・評価 (Chainer or Pytorch)
   - [x] MLPを用いた実装
   - [x] CNNを用いた実装
   - [x] それぞれの手法における考察
- [ ] CIFAR10の学習・評価 (Chainer or Pytorch)
   - [ ] MLPを用いた実装
   - [ ] CNNを用いた実装
   - [ ] それぞれの手法における考察
- [ ] (Optional*) 有名なネットワークの評価・比較 (Lenet, Mobilenet, VGG16, Resnet50 etc.)
- [x] (Optional*) Optimizerごとの学習曲線を比較 (SGD, MomentumSGD, RMSProp, Adam etc.)
- [ ] (Optional**) データの水増し手法ごとの評価・比較 (平滑化, 部分マスク, 回転, mix-up)
- [ ] (Optional**) バッチ正規化, 勾配ノイズなどの手法の評価・比較
- [x] (Optional***) 深層学習フレームワークを用いずに, フルスクラッチでMLP(&CNN)を実装してみる


## 目的
- 横田研の計算環境の利用や研究を行う上で必要な知識の習得

## 最低限身につけて欲しいこと
- 研究室クラスタやTSUBAME3.0等の計算機環境の利用方法
- 深層学習の基盤となる仕組みの理解
- 深層学習フレームワーク(Chainer, Pytorch)の使い方
    - MLP (Multi Layer Perceptron)
    - CNN

## 課題
### 目次

1. 横田研クラスタを利用できるようにする
1. Pythonの環境構築
1. Pythonの基本的な文法を学ぶ
1. 深層学習の基本を学ぶ
1. MNISTの学習・評価 (Chainer or Pytorch)
  1. MLPを用いた実装
  1. CNNを用いた実装
1. CIFAR10の学習・評価 (Chainer or Pytorch)
  1. MLPを用いた実装
  1. CNNを用いた実装
1. (Optional*) 有名なネットワークの評価・比較 (Lenet, Mobilenet, VGG16, Resnet50 etc.)
1. (Optional*) Optimizerごとの学習曲線を比較 (SGD, MomentumSGD, RMSProp, Adam etc.)
1. (Optional**) データの水増し手法ごとの評価・比較 (平滑化, 部分マスク, 回転, mix-up)
1. (Optional**) バッチ正規化, 勾配ノイズなどの手法の評価・比較
1. (Optional***) 深層学習フレームワークを用いずに, フルスクラッチでMLP(&CNN)を実装してみる

注: Optionalは興味があればやってください. *の数が増えるほど難易度が上がります. (難しい分, 深層学習への理解が深まると思います.)  
注: Pythonは3系(できれば>=3.6)を使ってください.  
注: プログラミング環境は好みがあると思うので, 各自好きなものを使ってください.  

6までできれば十分ですが, 興味がある場合は6以降も挑戦してみてください.  
いずれも活発に研究されている領域なので, 今後の研究を行う上で良い踏み台になると思います.  

何かわからないことがあれば, 先輩に質問をしてください. 先輩を困らせるような質問をたくさんしましょう!!

### 横田研クラスタを利用できるようにする
研究室クラスタは10台のGPUマシンおよび1台のCPUマシンから成る計算機環境です.  
SSH経由でどこからでもアクセスできるようになっているので, 好きなときにGPU環境を利用できます.  

SSHの設定はドキュメントを参考にして行ってください.  
(アカウントがない場合は@y1r君に発行をお願いする.)  

[横田研クラスタの使い方](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Frioyokotalab%2Fserver-admin%2Fwiki&sa=D&sntz=1&usg=AFQjCNGqJF7s_0LWkm3hjnp3luFhR6CrVA)

[Slurmを用いたJobの投入方法](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Frioyokotalab%2Fserver-admin%2Fwiki%2F%25E6%25A8%25AA%25E7%2594%25B0%25E7%25A0%2594%25E3%2582%25AF%25E3%2583%25A9%25E3%2582%25B9%25E3%2582%25BF%25E3%2581%25AE%25E4%25BD%25BF%25E3%2581%2584%25E6%2596%25B9(Slurm%25E3%2581%25A7%25E3%2581%25AE%25E3%2582%25B8%25E3%2583%25A7%25E3%2583%2596%25E3%2581%25AE%25E6%258A%2595%25E5%2585%25A5%25E3%2581%25AE%25E4%25BB%2595%25E6%2596%25B9)%23%25E3%2583%258E%25E3%2583%25BC%25E3%2583%2589%25E6%2595%25B0%25E3%2582%2592%25E5%25A4%2589%25E3%2581%2588%25E3%2582%258B&sa=D&sntz=1&usg=AFQjCNH0lPtZxw0zJHY47ViyjjdKsWtkAg)

### Pythonの環境構築
既にわかるようなら飛ばしてしまって構いません.  

PythonはMac版の場合[こちら](https://www.python.org/downloads/)からインストールできます. (Windowsはググってください)  
pyenv, virtualenvを使うと複数のPython環境を簡単に構築できるので余力がある場合は使ってみてください.  
下記のサイトを参考にすると簡単に導入きます.  

[pyenv と pyenv-virtualenv で環境構築](https://qiita.com/Kodaira_/items/feadfef9add468e3a85b)

ChainerやPytorch, Numpy, Scipyのインストール方法は各自調べてください.  
もしかしたらChainer, Pytorchのインストールではまるかもしれないので, そのときはためらわずに先輩に質問してください.  

### Pythonの基本的な文法を学ぶ
既にわかるようなら飛ばしてしまって構いません.  

下記の資料が参考になると思います.  

- Python3基礎文法
- Python入門
- NumpyはCPUでの行列演算においてよく利用されるので, NumPy 入門 などを軽く読んでみてください.
(Numpyの知識はChainer, Pytorchの行列演算を行う上で役に立つ場合が多いです)
詰まりやすい, Numpyにおけるaxisやndimsなどの概念は「NumPyの軸(axis)と次元数(ndim)とは何を意味するのか」が参考になると思います.

### 深層学習の基本を学ぶ
次以降の章と平行しながらやるといいです.

参考資料は以下です. 上3つがおすすめです.  

- ディープラーニング チュートリアル
- ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装
- イラストで学ぶ ディープラーニング (KS情報科学専門書)
- 深層学習 (機械学習プロフェッショナルシリーズ)(少しむずかしい. 横田研の蔵書にあります)
- 深層学習リンク集 - DeepLearning (好きなものを読んでみてください)

### MNISTの学習・評価 (Chainer or Pytorch)
MNISTは画像認識分野の中で最も有名な白黒画像データセットで, 手書き数字画像60,000枚と、テスト画像10,000枚が収録されています.  
前章で得た知識を元に, MLP(Multi Layer Perceptron)およびCNN(Convolutional Neural Network)をそれぞれ用いて数字を判定するモデルを構築してください.  

もし余力があれば, ChainerUI, Tensorboard, Matplotlibなどを使ってTraining Loss, Validation Lossの可視化ができるようにすると面白いです.  

### CIFAR10の学習・評価 (Chainer or Pytorch)
CIFAR10は画像認識分野で(おそらく)MNISTの次に有名なデータセットです.  
10クラスに分けられた32x32ピクセルのカラー画像60000枚からなります. (1クラス6000枚)  

MNISTの問題よりも高次の特徴量を抽出する必要があるため, 層を深く(deepに)したときに, 汎化性能や学習の速度がどのように変化したか評価してください.

### (Optional*) 有名なネットワークの評価・比較 (Lenet, Mobilenet, VGG16, Resnet50 etc.)
深層学習では精度を向上させるために, 様々なモデルが提案されています.
モデルが提案された背景について調べながら, 様々なモデルを使ってMNISTやCIFAR10の学習・評価を行ってみてください.

- Lenet
- Mobilenet
- VGG16
- Resnet50, Resnet101
などがよく知られている有名なネットワークです.

### (Optional*) Optimizerごとの学習曲線を比較 (SGD, MomentumSGD, RMSProp, Adam, K-FAC etc.)
モデルを過学習を起こさずに, 効率よく学習するために様々なOptimizerが提案されています.
Optimizerが提案された背景について調べながら, 様々なモデルを使ってMNISTやCIFAR10の学習・評価を行ってみてください.
余力があれば, An overview of gradient descent optimization algorithmsや深層学習の最適化アルゴリズムを読んでみてください.

### (Optional**) データの水増し手法ごとの評価・比較 (平滑化, 部分マスク, 回転, mix-up)
汎化性能を向上させるために用いられる, データの水増しという手法があります.
データの水増し(Data Augumentation)について調べつつ, 色々な手法を試してみてください.
個人的にはmix-upが気になる(2019/04現在).

### (Optional**) 正規化, バッチ正規化, 勾配ノイズ, SmoothOutなどの手法の評価・比較
深層学習においては, 学習を効率よく行うためにデータの正規化がよく行われます.
また, 一般的にシャープな解ではなくフラットな解を得ることによって汎化性能が向上すると言われるため, 勾配ノイズやSmoothOutなどの手法が用いられています.
これらの手法を適用したモデルの学習・評価を行ってください. (特に学習の速度・安定性・汎化性能がどうなったか評価する)

### (Optional***) 深層学習フレームワークを用いずに, フルスクラッチでMLP(&CNN)を実装してみる
深層学習のフレームワークは高度に抽象化されているため, 中でどのような計算が行われているかを意識する必要がありません.
それだけでなく, GPUかCPUかということすらほとんど意識する必要がありません.
実際に誤差逆伝搬やOptimizerをフルスクラッチで実装することによって深層学習の気持ちに寄り添えるようになりましょう!!!

