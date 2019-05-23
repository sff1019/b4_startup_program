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
4. [実行方法](#実行方法)

---

## 用いた最適化手法について

今回，5つの最適化手法を用いて，実験・考察を行なった．  
最もシンプルなSGDから，改良を重ねたAdamを通して，精度や実行時間をみた．  
以下，それぞれの手法の簡易的説明，そして式を記す．

### SGD

<img height='50' src='./assets/SGD.png'>

勾配降下法と呼ばれる手法であり，多くはミニバッチで行われるため，確率的勾配降下法とも呼ばれる．  
パラメタの勾配を求め，それらを用いて，最適化を行う最もシンプルな方法である．  
しかし，収束の不安定性・遅さから高次元の問題で使われることはない．

### Momentum SGD

<img height='100' src='./assets/momentum_SGD.png'>

物理のモーメンタムを用いた手法．vは速度を表し，ボールが池面の傾斜を転がるように動く．  
SGDに比べて，x軸方向に受ける力が小さく，y軸方向には受ける力が大きいが，速度は安定しないので，SGDに比べてx軸方向へ早く近づくことができる．

### AdaGrad

<img height='100' src='./assets/AdaGrad.png'>

AdaGradは，パラメタの要素ごとに適応的に学習係数を調整しながら学習を行う手法である．  
hはこれまで経験した勾配の値を2乗和として保持することによって，学習のスケールを調整．

### RMSprop

<img height='100' src='./assets/RMSprop.png'>

RMSpropはAdaGradを改良したアルゴリズムである．  
AdaGradは学習率が0に十分近くなってしまうと，まだ坂があったとしてもほとんど更新されなくなってしまうという問題があった．  
そこで提案されたRMSpropは初期の影響がαに応じて指数的に減衰する．

### Adam
<img height='200' src='./assets/Adam.png'>

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

精度

<p align='center'>
  <img height='300' src='./assets/cifar_mlp_cpu_accuracy.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|42.338|46.434|48.776|50.37|51.9|53.332|54.552|55.552|56.266|56.978|42.52|45.79|46.9|48.18|48.65|50.01|50.74|51.07|50.96|51.32|
|Momentum SGD|45.236|50.192|52.344|54.964|55.664|56.296|57.354|60.284|60.406|60.15|44.01|46.43|47.84|48.8|49.1|48.36|49.01|50.13|49.45|48.83|
|AdaGrad|43.212|44.754|45.674|46.236|46.906|47.308|47.7|47.958|48.474|48.762|43.15|44.58|45.22|46.01|46.34|46.69|46.9|46.82|47.09|47.26|
|RMSprop|43.666|47.14|49.222|49.738|51.076|50.918|52.668|54.922|54.77|55.216|41.74|44.54|45.19|45.04|45.71|44.82|44.94|46.05|45.69|45.53|
|Adam|40.168|45.71|48.818|49.512|51.608|52.31|53.564|55.89|53.374|55.05|39.11|43.85|45.02|45.28|46.09|46.37|46.73|47.62|45.74|46.76|

損失

<p align='center'>
  <img height='300' src='./assets/cifar_mlp_cpu_loss.png?raw=true'>
</p>

|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|1.823|1.609|1.53|1.475|1.43|1.39|1.356|1.326|1.3|1.274|
|Momentum SGD|1.665|1.512|1.439|1.387|1.349|1.312|1.278|1.253|1.224|1.197|
|AdaGrad|1.747|1.633|1.596|1.572|1.555|1.541|1.529|1.519|1.511|1.503|
|RMSprop|1.801|1.658|1.599|1.55|1.515|1.491|1.465|1.44|1.418|1.398|
|Adam|1.782|1.645|1.582|1.541|1.502|1.47|1.445|1.416|1.396|1.377|


実行時間


<p align='center'>
  <img height='300' src='./assets/cifar_mlp_cpu_elapsed_time.png?raw=true'>
</p>

|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|22.255|42.347|63.223|85.221|107.002|128.972|150.287|171.88|195.438|216.7|
|Momentum SGD|23.515|47.023|70.282|93.517|116.816|139.875|162.791|185.092|207.407|229.515|
|AdaGrad|30.259|60.39|92.6|123.503|153.971|184.547|214.907|244.68|275.88|307.382|
|RMSprop|31.253|62.478|93.776|124.663|155.544|188.937|222.575|254.491|286.799|319.131|
|Adam|34.408|69.024|102.166|135.875|171.338|205.248|237.989|270.731|304.286|341.85|


#### GPUにおける実行

ノード：pascal   
ノード数：1  
CUDA：10.1


精度

<p align='center'>
  <img height='300' src='./assets/cifar_mlp_gpu_accuracy.png?raw=true'>
</p>

|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|42.926|46.562|49.318|51.59|53.358|54.97|56.796|57.572|58.696|60.662|43.06|45.62|47.81|49.36|50.55|51.23|52.08|51.52|52.33|52.51|
|Momentum SGD|49.932|52.614|55.282|56.344|59.324|59.042|62.498|63.36|64.812|67.396|47.7|47.61|49.28|49.45|49.28|49.24|50.3|49.94|49.71|51.74|
|AdaGrad|45.274|47.148|48.236|49.218|49.88|50.496|50.842|51.314|51.596|52.058|44.88|46.27|46.83|47.57|48.03|48.27|48.56|48.97|49.02|49.26|
|RMSprop|42.532|45.308|47.216|49.506|51.67|50.282|51.604|51.694|52.228|55.366|40.05|42.42|43.35|44.49|45.54|44.33|44.95|43.12|43.98|46.37|
|Adam|42.602|42.582|45.454|46.846|50.49|50.756|52.748|51.506|52.544|53.238|41.81|40.6|41.95|42.78|45.48|45.68|46.05|45.16|44.45|45.64|

損失


<p align='center'>
  <img height='300' src='./assets/cifar_mlp_gpu_loss.png?raw=true'>
</p>

|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|1.809|1.598|1.515|1.454|1.401|1.356|1.316|1.279|1.245|1.215|
|Momentum SGD|1.652|1.483|1.408|1.349|1.294|1.256|1.217|1.182|1.139|1.108|
|AdaGrad|1.68|1.564|1.525|1.499|1.479|1.463|1.449|1.436|1.426|1.416|
|RMSprop|1.98|1.834|1.768|1.719|1.686|1.649|1.631|1.607|1.581|1.566|
|Adam|1.948|1.819|1.743|1.692|1.651|1.614|1.588|1.556|1.534|1.511|


実行時間

<p align='center'>
  <img height='300' src='./assets/cifar_mlp_gpu_elapsed_time.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|27.048|48.207|69.001|89.73|110.287|131.034|151.826|172.527|193.483|214.264|
|Momentum SGD|23.699|44.962|66.355|87.766|109.313|130.778|152.349|174.312|196.097|217.579|
|AdaGrad|24.577|47.076|69.636|92.34|114.872|137.435|160.296|182.673|204.935|227.264|
|RMSprop|25.251|48.304|71.25|94.353|117.358|140.036|163.201|186.675|209.604|232.8|
|Adam|25.879|49.915|73.979|98.15|122.113|146.315|170.183|194.417|218.234|242.177|

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

精度

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_cpu_accuracy.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|37.142|45.472|49.474|52.268|53.346|53.258|55.532|56.124|56.016|57.782|37.12|45.03|49.23|50.93|52.27|51.85|53.71|53.83|54.41|55.72|
|Momentum SGD|49.344|53.258|56.028|56.736|57.422|58.626|58.322|59.36|59.47|59.15|47.8|52.18|53.49|54.63|55.52|57.01|55.83|57.05|56.83|56.99|
|AdaGrad|36.13|38.332|40.008|41.294|41.826|42.49|43.002|43.326|43.776|44.216|36.2|38.38|39.66|40.7|41.43|42.27|42.72|43.31|43.67|43.86|
|RMSprop|45.502|52.706|51.804|56.306|57.092|58.002|57.362|58.504|58.542|60.278|45.43|50.99|50.87|54.94|54.93|55.91|55.56|56.46|56.11|57.79|
|Adam|51.43|54.674|57.026|57.554|59.238|60.126|60.04|58.686|61.33|62.132|50.29|53.28|55.19|55.3|57.7|57.73|57.68|56.31|58.76|59.27|

損失

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_cpu_loss.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|1.946|1.633|1.489|1.415|1.369|1.331|1.304|1.28|1.263|1.246|
|Momentum SGD|1.633|1.411|1.337|1.297|1.267|1.244|1.231|1.217|1.206|1.199|
|AdaGrad|1.884|1.744|1.697|1.669|1.642|1.627|1.61|1.599|1.585|1.576|
|RMSprop|1.634|1.427|1.354|1.314|1.283|1.263|1.243|1.225|1.208|1.2|
|Adam|1.605|1.371|1.293|1.256|1.231|1.208|1.189|1.177|1.167|1.155|

実行時間

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_cpu_elapsed_time.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|46.741|95.319|148.565|199.509|246.546|294.55|343.183|392.104|439.554|486.772|
|Momentum SGD|49.897|100.565|153.861|202.562|250.803|299.17|348.553|396.714|446.283|494.693|
|AdaGrad|51.306|102.183|153.852|204.674|255.487|306.315|357.099|407.879|458.784|509.547|
|RMSprop|52.356|104.435|156.115|208.332|260.29|312.128|364.126|419.126|473.624|526.96|
|Adam|55.88|110.97|168.923|227.283|281.763|336.683|390.786|445.209|500.467|558.147|

#### GPUにおける実行

ノード：pascal   
ノード数：1  
CUDA：10.1


精度

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_gpu_accuracy.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|39.066|45.858|48.862|50.152|52.54|54.264|54.998|54.528|56.628|57.676|38.89|44.7|47.62|49.44|51.52|52.98|53.47|53.13|55.05|55.28|
|Momentum SGD|48.684|52.626|55.488|57.046|57.15|56.094|57.612|58.522|58.972|59.586|47.81|51.35|53.82|55.43|54.83|53.98|55.49|55.67|56.69|56.73|
|AdaGrad|36.528|38.936|40.794|41.702|42.838|43.568|44.314|44.636|45.214|45.754|36.5|39.09|41.01|41.86|42.89|43.47|44.34|44.7|45.31|45.7|
|RMSprop|48.974|51.988|54.594|54.128|57.116|56.764|58.596|57.982|58.242|58.594|47.72|50.53|53.14|52.93|55.7|55.02|57.17|55.75|55.52|55.87|
|Adam|50.492|54.116|54.384|56.604|56.458|58.454|59.194|59.92|59.288|59.938|49.92|52.8|52.7|54.75|54.65|56.4|57.17|57.52|56.9|57.76|


損失

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_gpu_loss.png?raw=true'>
</p>


|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|1.934|1.614|1.496|1.432|1.382|1.34|1.306|1.283|1.256|1.239|
|Momentum SGD|1.625|1.407|1.338|1.294|1.264|1.249|1.231|1.219|1.21|1.196|
|AdaGrad|1.854|1.723|1.676|1.641|1.611|1.59|1.574|1.555|1.544|1.534|
|RMSprop|1.623|1.416|1.346|1.313|1.287|1.266|1.251|1.232|1.225|1.22|
|Adam|1.6|1.394|1.322|1.283|1.251|1.23|1.215|1.2|1.191|1.174|


実行時間

<p align='center'>
  <img height='300' src='./assets/cifar_cnn_gpu_elapsed_time.png?raw=true'>
</p>

|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|
|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGD|41.999|77.281|112.234|147.129|182.602|217.88|252.928|288.046|323.595|358.221|
|Momentum SGD|40.212|78.091|115.807|153.621|191.378|229.291|267.046|304.407|341.787|379.691|
|AdaGrad|42.56|83.326|124.162|164.57|204.89|245.725|286.313|326.79|367.894|408.575|
|RMSprop|45.053|87.707|130.43|172.719|215.418|258.378|300.513|343.468|386.217|428.926|
|Adam|47.173|92.217|137.239|182.334|227.364|272.627|317.843|363.545|408.396|453.587|


---

## 実行方法

### main.py

それぞれ，`./mlp`または`./cnn`内で実行  

```
$ python main.py --optimizer_type=$MY_OPTIMIZER --device=$MY_DEVICE
```

ただし，`--optimizer_type`は`[sgd, msgd, adagrad, rmsprop, adam]`のいずれか  
また，`--device`は`[cpu, gpu]`のいずれかから選ぶ

`$ python main.py --help`にて，引数の詳細が表示される．

### plot.py

`./`内で実行

`$ python plot.py --plot_type=$MY_PLOT_TYPE --net_type=$MY_NET_TYPE --device=$MY_DEVICE`

ただし，`--plot_type`は`[accuracy ,loss, elapsed_time]`のいずれか，`--device`は`[cpu, gpu]`，そして`net_type`は`[mlp, cnn]`のいずれかから選ぶ
