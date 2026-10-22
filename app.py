from pathlib import Path
import json
import pandas as pd
import joblib
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Plant Watering AI',page_icon='🌱',layout='wide')
st.title('🌱 Plant Watering AI')
st.caption('Decision Tree · только jibrand/Plant-dataset · числовые признаки и синтетические сценарии')
st.warning('Учебный прототип: числовых датчиков и фактов полива в исходном датасете нет. Модель обучена на СИНТЕТИЧЕСКИХ примерах по явно заданным правилам. Не используйте как единственный источник решений о поливе реальных растений.')
@st.cache_resource
def model(): return joblib.load(ROOT/'model'/'irrigation_tree.joblib')
@st.cache_data
def data():
    d=pd.read_csv(ROOT/'data'/'plant_catalog.csv')
    m=json.loads((ROOT/'model'/'metrics.json').read_text(encoding='utf-8'))
    return d,m
catalog,metrics=data(); clf=model()
plants=sorted(catalog.plant.unique())
plant=st.selectbox('Выберите растение',plants)
choices=catalog.loc[catalog.plant==plant].copy()
choice=st.selectbox('Освещение / известная потребность',list(range(len(choices))),format_func=lambda i:f"{choices.iloc[i]['light']} — {choices.iloc[i]['base_need']}")
selected=choices.iloc[choice]
st.subheader('Изменяйте текущие показатели')
c1,c2,c3=st.columns(3)
with c1: soil=st.slider('Влажность почвы, %',0,100,35,1)
with c2: temp=st.slider('Температура воздуха, °C',0,50,25,1)
with c3: humid=st.slider('Влажность воздуха, %',0,100,50,1)
features=pd.DataFrame([{'base_need':selected.base_need,'light':selected.light,'soil_moisture':soil,'air_temperature':temp,'air_humidity':humid}])
pred=int(clf.predict(features)[0]); p=float(clf.predict_proba(features)[0,list(clf.classes_).index(1)])
st.divider()
if pred:
    st.error('💧 РЕКОМЕНДАЦИЯ МОДЕЛИ: полив требуется (для учебного сценария)')
else:
    st.success('✅ РЕКОМЕНДАЦИЯ МОДЕЛИ: сейчас полив не требуется (для учебного сценария)')
st.metric('Оценка модели: вероятность класса «полить»',f'{p:.1%}')
st.progress(p)
base={'Редкий':22,'Умеренный':34,'Регулярный':45}[selected.base_need]
threshold=base+(5 if temp>=30 else (-5 if temp<15 else 0))+(4 if humid<40 else (-3 if humid>75 else 0))
st.caption(f'Для прозрачности: искусственная обучающая метка создавалась правилом «полить, если влажность почвы < {threshold}%». Реальное дерево может приближать это правило неточно.')
with st.expander('О модели и качестве обучения'):
    st.write('Вход модели:',features)
    st.write('Обучение:',metrics['train_rows'],'сценариев; тест:',metrics['test_rows'],'сценариев.')
    st.write('Accuracy на СИНТЕТИЧЕСКОМ тесте:',f"{metrics['accuracy']:.1%}")
    st.write('Balanced accuracy:',f"{metrics['balanced_accuracy']:.1%}")
    st.error(metrics['warning'])
with st.expander('Каталог растений'):
    st.dataframe(catalog,use_container_width=True)
    st.download_button('Скачать справочник CSV',catalog.to_csv(index=False).encode('utf-8'),'plant_catalog.csv','text/csv')
