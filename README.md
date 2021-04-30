# ML_predictionone
AI予測分析ツールPrediction Oneのチュートリアル例(機器の故障予測による故障の未然防止)を
pythonの`RandomForest`で実装した。
開発環境としてzipファイルで持ち運べるembedded版を用いた

# データ整理
1. python環境をzipファイルで持ち運べるembedded版を使うため、`python-3.8.9-embed-win32.zip`を解凍する。  
2. `input_file.zip`を解凍した`input_file`フォルダと、`include.zip`を解凍した`include`ファイルを1.で解凍したフォルダ内に置く。  
3. 学習モデル作成プログラム`RF_train_05.py`と、モデルから推測を行う`RF_predict_05.py`も1.で解凍したフォルダ内に置く。  
4. 3.のプログラムを実行するために`pandas`,`sklearn`,`matplotlib`,`japanize-matplotlib`をインストールする。  
5. コマンドプロンプトを起動し、`cd`コマンドを使って、1.で解凍したフォルダをカレントパスとして指定する。  
6. 学習モデル作成プログラム`RF_train_05.py`を実行するために、コマンドプロンプトから次のコマンドを打つ。  `python RF_train_05.py`  
7. モデルから推測を行う`RF_predict_05.py`を実行するために、コマンドプロンプトから次のコマンドを打つ。  `python RF_predict_05.py`  

6を実行すると1.で解凍したフォルダ内に出力フォルダ`output_file`が生成され、結果が保存される。  
また、7.を実行すると6.で作成されたモデル`status.pkl`が読み込まれ、予測結果が出力される。

# 以下のライブラリをインストール(4.参考)
python -m pip install pandas  
python -m pip install sklearn  
python -m pip install matplotlib  
python -m pip install japanize-matplotlib
