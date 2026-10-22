"""Train a Decision Tree on the ONLY project source: jibrand/Plant-dataset.
Uses the bundled exact CSV for reproducibility; remote HF URI is optional.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'plant_dataset.csv'
MODEL=ROOT/'model'
MODEL.mkdir(exist_ok=True)
TARGET_MAP={'moderate':'Умеренный','regular':'Регулярный','regularly':'Регулярный','water regularly':'Регулярный','infrequent':'Редкий','infrequently':'Редкий','water sparingly':'Редкий','water moderately':'Умеренный'}
LIGHT_MAP={'full sun':'Полное солнце','full sun to partial shade':'Солнце / полутень','partial shade to full sun':'Солнце / полутень','partial shade':'Полутень','bright indirect light':'Яркий рассеянный','low light':'Слабое освещение','shade':'Тень'}

def clean_col(s):
    return s.fillna('').astype(str).str.strip().str.strip('"').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

def load_clean():
    raw=pd.read_csv(DATA,low_memory=False)
    d=pd.DataFrame({'plant':raw['Plant'].fillna('').astype(str).str.strip(), 'light_raw':clean_col(raw['Light']), 'water_raw':clean_col(raw['Watering'])})
    d['light']=d['light_raw'].map(LIGHT_MAP)
    d['watering']=d['water_raw'].map(TARGET_MAP)
    d=d.loc[(d.plant!='') & d.light.notna() & d.watering.notna(), ['plant','light','watering']].copy()
    # Repeated entries are common; deduplicate exact records to reduce leakage.
    d=d.drop_duplicates().reset_index(drop=True)
    return d,raw

def main():
    d,raw=load_clean()
    X=d[['plant','light']]
    y=d['watering']
    train_idx,test_idx=train_test_split(d.index,test_size=.25,random_state=42,stratify=y)
    pipeline=Pipeline([('encode',ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['plant','light'])])),('tree',DecisionTreeClassifier(max_depth=12,min_samples_leaf=3,class_weight='balanced',random_state=42))])
    pipeline.fit(X.loc[train_idx],y.loc[train_idx]); pred=pipeline.predict(X.loc[test_idx])
    metrics={'original_rows':len(raw),'usable_unique_rows':len(d),'unique_plants':d.plant.nunique(),'train_rows':len(train_idx),'test_rows':len(test_idx),'accuracy':round(float(accuracy_score(y.loc[test_idx],pred)),4),'balanced_accuracy':round(float(balanced_accuracy_score(y.loc[test_idx],pred)),4),'class_counts':d.watering.value_counts().to_dict(),'report':classification_report(y.loc[test_idx],pred,zero_division=0,output_dict=True),'caveats':'Watering descriptions are categorical and CSV contains shifted/misaligned fields. This is NOT a sensor-trained watering-time predictor. Random split over plant names is optimistic; many repeats may relate.'}
    joblib.dump(pipeline,MODEL/'watering_tree.joblib')
    d.to_csv(DATA.parent/'cleaned_for_training.csv',index=False)
    (MODEL/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['report','caveats']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
