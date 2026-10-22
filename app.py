import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='PlantWater AI | Умный полив',page_icon='🌿',layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(125deg,#f2fff6,#f7fbff 75%);}
.block-container {max-width:1180px;padding-top:2rem;}
h1,h2,h3 {color:#173a32;}
[data-testid="stMetric"] {background:white;padding:15px;border-radius:13px;border:1px solid #d9efe2;}
</style>''',unsafe_allow_html=True)

@st.cache_data
def plants():return pd.read_csv(ROOT/'data/plant_catalog.csv',low_memory=False).fillna('')
@st.cache_resource
def model():return joblib.load(ROOT/'models/irrigation_tree.joblib')
@st.cache_data
def irrigation():return pd.read_csv(ROOT/'data/irrigation_clean.csv')
@st.cache_data
def metrics():return json.loads((ROOT/'models/metrics.json').read_text())

cat=plants();clf=model();met=metrics()
st.title('🌿 PlantWater AI')
st.caption('Каталог растений WCVP + WaterWise • прогноз дерева решений по историческим записям Irrigation Dataset')
st.warning('⚠️ Учебный прототип, не автоматический контроллер полива. Дерево обучено предсказывать записанное состояние полива для двух кодов культур, а не доказанную потребность любого выбранного вида. Название вида из WCVP не является входом этой модели.')
view=st.tabs(['💧 Прогноз полива','🌱 Каталог растений','🌳 Анализ модели','📘 О проекте'])
with view[0]:
 l,r=st.columns([1.08,1])
 with l:
  st.subheader('Растение и датчики')
  query=st.text_input('Поиск по научному или обычному названию',placeholder='Например, Rosa или Abelia')
  if query.strip():
   m=cat['botanical_name'].str.contains(query,case=False,regex=False,na=False)|cat['common_name'].str.contains(query,case=False,regex=False,na=False)
   opts=cat.loc[m,'botanical_name'].head(150).tolist()
  else:opts=cat.loc[cat['water_needs'].ne(''),'botanical_name'].head(150).tolist()
  if not opts: st.info('Совпадений нет. Измените поисковый запрос.');selected=None
  else: selected=st.selectbox('Научное название растения',opts)
  if selected:
   p=cat[cat.botanical_name==selected].iloc[0]
   st.caption(f"Семейство: {p['family'] or 'нет данных'} · Климат: {p['climate'] or 'нет данных'}")
   if p['water_needs']:
    st.info(f"📚 WaterWise: потребность в воде — {p['water_needs']}. Эти данные справочные, а не прогноз дерева.")
   else:st.info('Для этого растения нет индивидуальной характеристики полива в WaterWise.')
  crop=st.selectbox('Код культуры для ML-модели (виды не расшифрованы)',[1,2],help='Коды 1 и 2 присутствуют в обучающем датасете; нельзя приравнивать к выбранному виду WCVP.')
  temp=st.slider('🌡️ Температура воздуха, °C',0.0,50.0,24.0,0.5)
  air=st.slider('💨 Влажность воздуха, %',0.0,100.0,55.0,1.0)
  soil=st.slider('💧 Влажность почвы, единицы датчика',0.0,100.0,35.0,1.0,help='Из исходных данных неизвестно, откалибрована ли шкала как объёмная влажность %.')
  soiltemp=st.slider('🌱 Температура почвы, °C',0.0,50.0,20.0,0.5)
 with r:
  st.subheader('Результат модели')
  row=pd.DataFrame([{'crop type':crop,'Temperature':temp,'Humidity':air,'Soil Moisture':soil,'Soil Tempertuer':soiltemp}])
  predict=int(clf.predict(row)[0]);prob=clf.predict_proba(row)[0];labels=[int(x) for x in clf.classes_];p_on=float(prob[labels.index(1)])
  if predict:st.error('🚿 Прогноз дерева: исторически полив ВКЛЮЧЁН (класс 1)')
  else:st.success('✅ Прогноз дерева: исторически полив ВЫКЛЮЧЕН (класс 0)')
  a,b=st.columns(2)
  a.metric('Вероятность класса 1',f'{p_on:.0%}')
  b.metric('Прогнозируемый класс',str(predict))
  fig=go.Figure(go.Indicator(mode='gauge+number',value=p_on*100,number={'suffix':'%'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#13795b'}}))
  fig.update_layout(height=255,margin=dict(l=15,r=15,t=10,b=15),paper_bgcolor='rgba(0,0,0,0)')
  st.plotly_chart(fig,use_container_width=True)
  st.markdown('#### Практическая подсказка (отдельно от модели)')
  if soil<25:st.info('Почва по введённой шкале выглядит сухой. Проверьте корневую зону и допустимость полива именно для выбранного вида.')
  elif soil>70:st.info('Показания влажности высокие. Проверьте, нет ли переувлажнения; без необходимости полив лучше отложить.')
  else:st.info('Промежуточное показание. Проверьте реальную влажность субстрата и требования выбранного вида.')
  st.caption('Пороги 25/70 — лишь демонстрационные, не обучены на данных и не являются универсальной агрономической нормой. Точное время до следующего полива не рассчитывается.')
with view[1]:
 st.subheader('Поиск в ботаническом каталоге')
 st.metric('Количество видов в компактной сборке',f'{len(cat):,}'.replace(',',' '))
 st.caption('Оригинальный WCVP содержит больше видов; ради быстрого развёртывания включена репрезентативная часть принятых названий и названия, сопоставленные с WaterWise. Сопоставление не всегда означает идентичность таксонов.')
 q=st.text_input('Найти вид, род, семейство',key='catalog_q')
 filtered=cat
 if q:filtered=cat[cat['botanical_name'].str.contains(q,case=False,regex=False)|cat['family'].str.contains(q,case=False,regex=False)|cat['common_name'].str.contains(q,case=False,regex=False)]
 st.dataframe(filtered[['botanical_name','common_name','family','lifeform','climate','water_needs','light_needs']].head(400),use_container_width=True,hide_index=True)
 st.download_button('Скачать найденные записи CSV',filtered.to_csv(index=False).encode('utf-8-sig'),'plants_filtered.csv','text/csv')
with view[2]:
 st.subheader('Обучение Decision Tree')
 a,b,c=st.columns(3);a.metric('Записей',met['records']);b.metric('Accuracy на тесте',f"{met['accuracy']:.1%}");c.metric('Balanced accuracy',f"{met['balanced_accuracy']:.1%}")
 st.write('Разбиение данных:',met['split'])
 st.warning('Идеальные показатели могут быть следствием временной зависимости и дублирующихся состояний оборудования. Это не гарантия качества на будущих данных и новых видах растений.')
 st.write('Признаки модели:',', '.join(met['features']))
 st.write('Число записей по меткам:',met['class_counts'])
 st.subheader('Визуализация дерева (верхние уровни)')
 fig,ax=plt.subplots(figsize=(14,6));plot_tree(clf.named_steps['tree'],feature_names=met['features'],class_names=['0','1'],filled=True,max_depth=3,fontsize=8,ax=ax);st.pyplot(fig);plt.close(fig)
 d=irrigation();st.subheader('Пример измерений');st.dataframe(d.head(25),hide_index=True,use_container_width=True)
with view[3]:
 st.subheader('Как устроен проект')
 st.markdown('''**1. WCVP:** справочник видов и климатических характеристик. **2. WaterWise:** сведения о потребности в воде для сопоставленных ботанических названий. **3. Irrigation Dataset:** исторические показания датчиков, по которым обучено дерево решений.\n\n**Ключевое ограничение:** между кодами культур 1/2 и конкретными видами WCVP отсутствует установленное соответствие. Поэтому выбор названия растения не влияет на прогноз ML.\n\n**Чтобы действительно прогнозировать «через сколько часов поливать конкретное растение»,** соберите данные с идентификатором вида, калиброванной влажностью, событиями полива, их временем и результатами. Затем обучите модель на целевом времени до следующего необходимого полива и проверьте на независимых участках/датах.\n\n**Источники:** пользовательские `wcvp.zip`, `waterwise-plants.csv`, `Irrigation-Dataset.zip`. Ботанический справочник: Royal Botanic Gardens, Kew — WCVP, https://doi.org/10.34885/egs6-cp24. Соблюдайте лицензию первоисточника при распространении данных.''')
