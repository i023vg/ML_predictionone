##05_機器の故障予測による故障の未然防止
##RandomForestを使用

# ライブラリのインポート
print("RF_predict_05.pyを実行")
print("Step1. ライブラリインポート")

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import os

#データ読み込み
print("Step2. データ読み込み")
INPUT_FILE=os.path.join(os.getcwd(),"input_file")
OUTPUT_FILE=os.path.join(os.getcwd(),"output_file")

df=pd.read_csv(os.path.join(INPUT_FILE,"test.csv"))
df['購入からの経過月数']=df['購入からの経過月数'].str.replace('month','').astype(float)

#保守担当チームの対応表読み込み
team=pd.read_csv(os.path.join(INPUT_FILE,'team.csv'),encoding='SHIFT-JIS')
#機器タイプの対応表読み込み
m_type=pd.read_csv(os.path.join(INPUT_FILE,'m_type.csv'),encoding='SHIFT-JIS')
#RandomForest.pyで求めた欠損値を読み込む
df_kesson=pd.read_csv(os.path.join(OUTPUT_FILE,"kesson.csv"),encoding='SHIFT-JIS')

print("Step3. 入力データ前処理")
#表の統合
list_id=list(range(df['機器ID'].size))
df_index=pd.DataFrame(list_id,columns=['cnt'])
df1=pd.concat([df_index,df],axis=1)
df2=df1.merge(team,how='inner',on='保守担当チーム')
df3=df2.merge(m_type,how='inner',on='機器タイプ')
df3=df3.sort_values('cnt').reset_index(drop=True)   #Dataframeを順番通りに並べ直す

# 稼働時平均温度の欠損行を抜き出して埋める
is_null1=df3['稼働時平均温度'].isnull()
# type_id=0 つまり,機器タイプAのとき
df3.loc[(df3['type_id']==0) & (is_null1), '稼働時平均温度']=df_kesson['稼働時平均温度'][0]
# type_id=1 つまり,機器タイプBのとき
df3.loc[(df3['type_id']==1) & (is_null1), '稼働時平均温度']=df_kesson['稼働時平均温度'][1]
# type_id=2 つまり,機器タイプCのとき
df3.loc[(df3['type_id']==2) & (is_null1), '稼働時平均温度']=df_kesson['稼働時平均温度'][2]

# 稼働時平均湿度の欠損行を抜き出して埋める
is_null2=df3['稼働時平均湿度'].isnull()
# type_id=0 つまり,機器タイプAのとき
df3.loc[(df3['type_id']==0) & (is_null2), '稼働時平均湿度']=df_kesson['稼働時平均湿度'][0]
# type_id=1 つまり,機器タイプBのとき
df3.loc[(df3['type_id']==1) & (is_null2), '稼働時平均湿度']=df_kesson['稼働時平均湿度'][1]
# type_id=2 つまり,機器タイプCのとき
df3.loc[(df3['type_id']==2) & (is_null2), '稼働時平均湿度']=df_kesson['稼働時平均湿度'][2]

col=['type_id','team_id','購入からの経過月数','稼働時平均温度',
     '稼働時平均湿度','油圧メーター値']
x=df3[col]

#学習済みモデルを読み込み
print("Step4. 学習したモデル読み込み")
with open(os.path.join(OUTPUT_FILE,'status.pkl'),'rb') as f:
  model=pickle.load(f)

#推測結果を出力
print('Step5. 推測結果を出力')
#y_pred=model.predict(x)
# 故障確率0.3以上を故障と判定
# 故障:0, 正常:1とするため、不等号が逆になっている
y_pred=(model.predict_proba(x)[:,0]<0.3).astype(int)
y_pred_proba=model.predict_proba(x)


df4=df3['機器ID']
df5=pd.DataFrame(y_pred,columns=['状態'])
df5.loc[(df5['状態']==0),'状態']='故障'
df5.loc[(df5['状態']==1),'状態']='正常'
df6=pd.DataFrame(y_pred_proba,columns=['故障','正常'])
submission=pd.concat([df4,df5,df6],axis=1)
submission.to_csv(os.path.join(OUTPUT_FILE,"予測結果_RandomForest.csv"),
                  index=False,encoding='shift-jis')
