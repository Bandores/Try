import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Plant Watering AI', page_icon='🌿',layout='wide')
st.title('🌿 Plant Watering AI')
st.caption('Decision Tree • исходный датасет: jibrand/Plant-dataset • обучение в Google Colab')
@st.cache_resource
def load_model(): return joblib.load(ROOT/'model'/'watering_tree.joblib')
@st.cache_data
def load_data():
    return pd.read_csv(ROOT/'data'/'cleaned_for_training.csv'),json.loads((ROOT/'model'/'metrics.json').read_text(encoding='utf-8'))
model=load_model(); df,metrics=load_data()
st.sidebar.header('Выберите растение')
plants=sorted(df.plant.dropna().unique().tolist())
plant=st.sidebar.selectbox('Название',plants)
lights=sorted(df.loc[df.plant==plant,'light'].dropna().unique().tolist())
light=st.sidebar.selectbox('Освещение',lights)
example=pd.DataFrame([{'plant':plant,'light':light}])
pred=model.predict(example)[0]
probas=pd.Series(model.predict_proba(example)[0],index=model.classes_).sort_values(ascending=False)
left,right=st.columns([3,2])
with left:
 st.subheader('Результат ML-модели')
 st.metric('Прогнозируемый режим полива',pred)
 st.progress(float(probas.iloc[0]))
 st.write(f'Вероятность ведущего класса: **{probas.iloc[0]*100:.1f}%**')
 st.bar_chart(probas)
 st.info('Это прогноз КАТЕГОРИИ полива по растению и освещению. Модель не обучалась на измерениях датчиков.')
with right:
 st.subheader('Текущие условия (демонстрационные правила)')
 soil=st.slider('Влажность почвы, %',0,100,35)
 air_temp=st.slider('Температура воздуха, °C',-10,50,24)
 humidity=st.slider('Влажность воздуха, %',0,100,50)
 st.caption('Эти ползунки НЕ являются входами обученного дерева: в исходном датасете нет соответствующих числовых измерений.')
 # Clearly labeled non-ML educational rule. Not species-specific thresholds.
 if soil<25: advice='Почва выглядит сухой по введённому значению: проверьте субстрат и возможность полива.'
 elif soil<45: advice='Умеренная влажность: проверьте почву перед поливом.'
 else: advice='Влажность достаточная по демонстрационному правилу: пока не поливайте.'
 st.warning('Правило, а не ML-прогноз: '+advice)
 st.caption('Пороговые значения условны, не являются агрономической рекомендацией для выбранного вида. Температура и влажность воздуха показаны для демонстрации будущего подключения датчиков.')
st.divider()
st.subheader('Обучение и оценка')
a,b,c=st.columns(3)
a.metric('Записей после очистки',metrics['usable_unique_rows'])
b.metric('Accuracy на тесте',f"{metrics['accuracy']:.1%}")
c.metric('Balanced accuracy',f"{metrics['balanced_accuracy']:.1%}")
with st.expander('Как обучена модель и где смотреть код'):
 st.markdown('1. Откройте `Plant_Watering_Training_Colab.ipynb` в Google Colab. 2. Запустите все ячейки. 3. Скачайте архив с обученной моделью. 4. Обновите файлы `model/` и `data/` в GitHub. 5. Streamlit автоматически обновит сайт после нового деплоя.')
 st.write('Вход модели: `plant`, `light` → OneHotEncoder → DecisionTreeClassifier → категория `watering`.')
 st.write('Оригинальная таблица содержит смещённые значения в некоторых колонках. Для обучения оставлены только однозначные категории. Точность не подтверждает переносимость на новые виды.')
with st.expander('Очищенные данные'):
 st.dataframe(df,use_container_width=True)
 st.download_button('Скачать CSV',df.to_csv(index=False).encode('utf-8'),'cleaned_for_training.csv','text/csv')
