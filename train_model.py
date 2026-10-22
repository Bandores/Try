from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODEL=ROOT/'model'
MODEL.mkdir(exist_ok=True)
WATER_MAP={'moderate':'Умеренный','regular':'Регулярный','regularly':'Регулярный','water regularly':'Регулярный','infrequent':'Редкий','infrequently':'Редкий','water sparingly':'Редкий','water moderately':'Умеренный'}
LIGHT_MAP={'full sun':'Полное солнце','full sun to partial shade':'Солнце / полутень','partial shade to full sun':'Солнце / полутень','partial shade':'Полутень','bright indirect light':'Яркий рассеянный','low light':'Слабое освещение','shade':'Тень'}
FEATURES=['base_need','light','soil_moisture','air_temperature','air_humidity']

def clean(s):
    return s.fillna('').astype(str).str.strip().str.strip('\"').str.strip().str.lower().str.replace(r'\s+',' ',regex=True)

def clean_original(raw):
    d=pd.DataFrame({'plant':raw['Plant'].fillna('').astype(str).str.strip(),'light':clean(raw['Light']).map(LIGHT_MAP),'base_need':clean(raw['Watering']).map(WATER_MAP)})
    return d.loc[(d.plant!='') & d.light.notna() & d.base_need.notna()].drop_duplicates().reset_index(drop=True)

def rule_label(base_need,soil,temp,humidity):
    # Экспертная ОБУЧАЮЩАЯ эвристика, а НЕ реальные наблюдения или официальные агрономические нормы.
    threshold={'Редкий':22,'Умеренный':34,'Регулярный':45}[base_need]
    threshold+=5 if temp>=30 else (-5 if temp<15 else 0)
    threshold+=4 if humidity<40 else (-3 if humidity>75 else 0)
    return int(soil<threshold)

def make_synthetic(cleaned,seed=42,examples_per_plant=24):
    rng=np.random.default_rng(seed)
    blocks=[]
    for _,rec in cleaned.iterrows():
        n=examples_per_plant
        b=pd.DataFrame({'plant':[rec.plant]*n,'light':[rec.light]*n,'base_need':[rec.base_need]*n,
          'soil_moisture':rng.integers(5,96,n),'air_temperature':rng.integers(5,43,n),'air_humidity':rng.integers(15,96,n)})
        b['needs_watering']=[rule_label(rec.base_need,int(s),int(t),int(h)) for s,t,h in zip(b.soil_moisture,b.air_temperature,b.air_humidity)]
        blocks.append(b)
    return pd.concat(blocks,ignore_index=True)

def main():
    raw=pd.read_csv(DATA/'plant_dataset.csv',low_memory=False)
    cleaned=clean_original(raw)
    synthetic=make_synthetic(cleaned)
    # Разделение на примеры сценариев. Это не независимая проверка на реальных поливах.
    X_train,X_test,y_train,y_test=train_test_split(synthetic[FEATURES],synthetic['needs_watering'],test_size=.25,random_state=42,stratify=synthetic['needs_watering'])
    prep=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'),['base_need','light'])],remainder='passthrough')
    pipe=Pipeline([('prep',prep),('tree',DecisionTreeClassifier(max_depth=9,min_samples_leaf=8,random_state=42))])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    metrics={'source':'jibrand/Plant-dataset ONLY','original_rows':len(raw),'clean_rows':len(cleaned),'synthetic_rows':len(synthetic),'plants':cleaned.plant.nunique(),'train_rows':len(X_train),'test_rows':len(X_test),'accuracy':round(float(accuracy_score(y_test,pred)),4),'balanced_accuracy':round(float(balanced_accuracy_score(y_test,pred)),4),'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0),'warning':'TEST LABELS SYNTHETIC: scores measure agreement with the user-defined heuristic, NOT real-world irrigation accuracy. Numeric sensor readings are generated for demonstration.'}
    joblib.dump(pipe,MODEL/'irrigation_tree.joblib')
    cleaned.to_csv(DATA/'plant_catalog.csv',index=False)
    synthetic.to_csv(DATA/'synthetic_training.csv',index=False)
    (MODEL/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
