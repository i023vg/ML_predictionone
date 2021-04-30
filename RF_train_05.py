##05_機器の故障予測による故障の未然防止
##RandomForestを使用
# 以下のライブラリをインストール
#pip install pandas
#pip install sklearn
#pip install matplotlib
#pip install japanize-matplotlib

# ***** ライブラリのインポート *****
print("*** RF_train_05.pyの実行 ***")
print("Step1. ライブラリのインポート")

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import pickle
import os
from include.module1 import Seido #自作モジュール

# ***** 入力ファイルの読み込み *****
print("Step2. 入力ファイルの読み込み")
INPUT_FILE=os.path.join(os.getcwd(),"input_file")
OUTPUT_FILE=os.path.join(os.getcwd(),"output_file")
os.makedirs(OUTPUT_FILE,exist_ok=True)

THRESH=0.3  # 故障確率をいくつ以上を故障と判定するか指定

df=pd.read_csv(os.path.join(INPUT_FILE,'train.csv'))
df['購入からの経過月数']=df['購入からの経過月数'].str.replace('month','').astype(float)

#製品故障の対応表ファイル読み込み
status = pd.read_csv(os.path.join(INPUT_FILE,'status.csv'),encoding='SHIFT-JIS')
#保守担当チームの対応表読み込み
team=pd.read_csv(os.path.join(INPUT_FILE,'team.csv'),encoding='SHIFT-JIS')
#機器タイプの対応表読み込み
m_type=pd.read_csv(os.path.join(INPUT_FILE,'m_type.csv'),encoding='SHIFT-JIS')

print("Step3. 入力データ前処理")
#表の統合
df2=df.merge(status,how='inner',on='状態(予測対象)')
df3=df2.merge(team,how='inner',on='保守担当チーム')
df4=df3.merge(m_type,how='inner',on='機器タイプ')


# ***** 欠損値穴埋めなど *****
#稼働時平均温度
#稼働時平均湿度

##機器故障数のカウント
#print()
#print(df4['状態(予測対象)'].value_counts())
##欠損値のカウント
#print()
#print(df4.isnull().sum())

kesson1=df4.groupby('type_id').mean()['稼働時平均温度']
kesson2=df4.groupby('type_id').mean()['稼働時平均湿度']
kesson=pd.concat([kesson1,kesson2],axis=1)

# 稼働時平均温度の欠損行を抜き出して埋める
is_null1=df4['稼働時平均温度'].isnull()
# type_id=0 つまり,機器タイプAのとき
df4.loc[(df4['type_id']==0) & (is_null1), '稼働時平均温度']=kesson1[0]
# type_id=1 つまり,機器タイプBのとき
df4.loc[(df4['type_id']==1) & (is_null1), '稼働時平均温度']=kesson1[1]
# type_id=2 つまり,機器タイプCのとき
df4.loc[(df4['type_id']==2) & (is_null1), '稼働時平均温度']=kesson1[2]

# 稼働時平均湿度の欠損行を抜き出して埋める
is_null2=df4['稼働時平均湿度'].isnull()
# type_id=0 つまり,機器タイプAのとき
df4.loc[(df4['type_id']==0) & (is_null2), '稼働時平均湿度']=kesson2[0]
# type_id=1 つまり,機器タイプBのとき
df4.loc[(df4['type_id']==1) & (is_null2), '稼働時平均湿度']=kesson2[1]
# type_id=2 つまり,機器タイプCのとき
df4.loc[(df4['type_id']==2) & (is_null2), '稼働時平均湿度']=kesson2[2]

col=['type_id','team_id','購入からの経過月数','稼働時平均温度',
     '稼働時平均湿度','油圧メーター値']
x=df4[col]
t=df4['status_id']


# ***** 学習 *****
print("Step4. モデル学習")
x_train,x_test,y_train,y_test=train_test_split(x,t,
                                                test_size=0.2,random_state=0)
#class_weight='balanced'で不均衡データに対処できる
model=RandomForestClassifier(n_estimators=30,random_state=0,class_weight='balanced')
model.fit(x_train,y_train)
print('訓練データ件数{} 検証データ件数{}'.format(len(y_train),len(y_test)))

# ***** 精度評価 *****
print("Step5. 精度評価")

print("故障確率{}%以上を故障と判定する".format(THRESH*100,".1f"))
#TrainデータのPrecision,Recall,f1,AUCを計算
acc,Precision, Recall, f1, y_auc = Seido(model,THRESH,x_train,y_train,"Train")
#TestデータのPrecision,Recall,f1,AUCを計算
acc2,Precision2, Recall2, f1_2, y_auc2 = Seido(model,THRESH,x_test,y_test,"Test")

# ***** モデル保存 *****
print("Step6. モデル保存")
with open(os.path.join(OUTPUT_FILE,'status.pkl'),'wb') as f:
  pickle.dump(model,f)


# ***** 精度など参考情報についてcsv出力 *****
print("Step7. csvファイル出力")

print(" accuracy.csvを出力")
with open(os.path.join(OUTPUT_FILE,"accracy.csv"),"w",newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['対象データ','件数', '精度','適合率','再現率','f1値','AUC'])
    writer.writerow(['訓練用', len(x_train),
                     format(acc*100,".2f")+"%",
                     format(Precision*100,".2f")+"%",
                     format(Recall*100,".2f")+"%",
                     format(f1*100,".2f")+"%",
                     format(y_auc*100,".2f")+"%"])
    writer.writerow(['検証用', len(x_test),
                     format(acc2*100,".2f")+"%",
                     format(Precision2*100,".2f")+"%",
                     format(Recall2*100,".2f")+"%",
                     format(f1_2*100,".2f")+"%",
                     format(y_auc2*100,".2f")+"%"])

print(" kesson.csvを出力")
kesson.to_csv(os.path.join(OUTPUT_FILE,"kesson.csv"),
              index=False,encoding="shift-jis")

# 特徴量重要度を表示
print(" feature.csvを出力")
feature=pd.DataFrame(model.feature_importances_, index=x.columns)
print(feature)

feature.to_csv(os.path.join(OUTPUT_FILE,"feature.csv"),
               header=False, index=True,encoding="shift-jis")
