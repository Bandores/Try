"""Reproduce training on the included, cleaned sensor dataset."""
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent
p=pd.read_csv(ROOT/'data/irrigation_clean.csv');p['Time']=pd.to_datetime(p.Time,errors='coerce');p=p.sort_values('Time')
f=['crop type','Temperature','Humidity','Soil Moisture','Soil Tempertuer'];p=p[p['irrigation'].isin([0,1])]
i=int(.8*len(p));tr,te=p.iloc[:i],p.iloc[i:];split='Chronological last 20%'
if tr.irrigation.nunique()<2 or te.irrigation.nunique()<2:
 tr,te=train_test_split(p,test_size=.2,stratify=p.irrigation,random_state=42);split='Stratified random fallback'
m=Pipeline([('imputer',SimpleImputer(strategy='median')),('tree',DecisionTreeClassifier(max_depth=5,min_samples_leaf=30,class_weight='balanced',random_state=42))]);m.fit(tr[f],tr.irrigation.astype(int));pred=m.predict(te[f]);joblib.dump(m,ROOT/'models/irrigation_tree.joblib')
res={'records':len(p),'training_records':len(tr),'testing_records':len(te),'split':split,'accuracy':float(accuracy_score(te.irrigation,pred)),'balanced_accuracy':float(balanced_accuracy_score(te.irrigation,pred)),'class_counts':{str(k):int(v) for k,v in p.irrigation.value_counts().items()},'features':f,'warning':'Historic irrigation state is not verified correct watering advice.'}
(ROOT/'models/metrics.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
print(res)
