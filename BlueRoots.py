import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ===== إعداد الصفحة =====
st.set_page_config(page_title="Mangrove Dashboard Pro", layout="wide")
st.title("🌿 مشروع MangroveGuard KSA ")

# ===== مقدمة تعريفية =====
st.markdown("""
المانغروف من أهم البيئات الساحلية التي تحمي الشواطئ، تدعم التنوع البيولوجي، وتقلل تأثير التغير المناخي.  
في هذا المشروع نستعرض مواقع المانغروف في السعودية، معلوماتها البيئية، حالة الحماية، والتنوع البيولوجي لكل موقع.
""")

# ===== قراءة CSV =====
df = pd.read_csv("points.csv", encoding='utf-8')

# ===== دالة لاستخراج الأرقام من النصوص =====
def extract_number(val):
    if isinstance(val,str):
        val = val.replace('°C','').replace('%','').replace('PSU','').replace('m','').strip()
        if '-' in val:
            parts = val.split('-')
            try:
                nums = [float(p) for p in parts]
                return sum(nums)/len(nums)
            except:
                return 0.0
        else:
            try:
                return float(val)
            except:
                return 0.0
    return float(val)

# ===== تجهيز البيانات الرقمية للنموذج =====
features = ['temperature','summer_temperature','winter_temperature','salinity','humidity','tidal_range']
for col in features:
    df[col+'_num'] = df[col].apply(extract_number)

# ===== تدريب نموذج RandomForest =====
X = df[[col+'_num' for col in features]]
y = df['status']
le = LabelEncoder()
y_enc = le.fit_transform(y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X,y_enc)

# ===== Sidebar: فلترة =====
st.sidebar.header("فلترة البيانات")
types = df['type'].unique()
statuses = df['status'].unique()
selected_type = st.sidebar.selectbox("اختر نوع المانغروف:", ["الكل"] + list(types))
selected_status = st.sidebar.selectbox("اختر الحالة البيئية:", ["الكل"] + list(statuses))

filtered_df = df.copy()
if selected_type != "الكل":
    filtered_df = filtered_df[filtered_df['type'] == selected_type]
if selected_status != "الكل":
    filtered_df = filtered_df[filtered_df['status'] == selected_status]

# ===== اختيار موقع محدد =====
site = st.selectbox("اختر موقع المانغروف:", ["كل المواقع"] + list(filtered_df["name"]))
if site != "كل المواقع":
    filtered_df = filtered_df[filtered_df["name"] == site]

# ===== AI: توقع الحالة البيئية لكل موقع وعمل لون ديناميكي =====
def predict_status_color(row):
    data = pd.DataFrame([[row[col+'_num'] for col in features]], columns=[col+'_num' for col in features])
    pred = model.predict(data)[0]
    pred_status = le.inverse_transform([pred])[0]
    color_map = {"healthy":[0,200,0,160], "vulnerable":[255,165,0,160], "critical":[255,0,0,160]}
    return pred_status, color_map.get(pred_status,[100,100,100,160])

# إضافة عمود للنقاط ولونها حسب توقع AI
pred_status_list = []
color_list = []
radius_list = []

def species_to_int(val):
    try:
        return int(''.join(filter(str.isdigit, str(val))))
    except:
        return 1

for idx,row in filtered_df.iterrows():
    pred_status, color = predict_status_color(row)
    pred_status_list.append(pred_status)
    color_list.append(color)
    radius_list.append(species_to_int(row['species_diversity'])*10000)

filtered_df['pred_status'] = pred_status_list
filtered_df['color'] = color_list
filtered_df['radius'] = radius_list

# ===== الخريطة التفاعلية =====
st.subheader("🗺️ الخريطة التفاعلية")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    pickable=True,
    get_position='[lon, lat]',
    get_color='color',
    get_radius='radius'
)

view_state = pdk.ViewState(
    latitude=filtered_df["lat"].mean() if len(filtered_df) > 0 else 24,
    longitude=filtered_df["lon"].mean() if len(filtered_df) > 0 else 45,
    zoom=5,
    pitch=0
)

tooltip = {
    "html": "<b>{name}</b><br>نوع: {type}<br>درجة الحرارة: {temperature}<br>الرطوبة: {humidity}<br>الحالة المتوقعة: {pred_status}<br>التنوع: {species_diversity}",
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "14px", "padding": "10px"}
}

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip
))

# ===== بطاقات لكل موقع =====
st.subheader("📋 معلومات المواقع")
for idx, row in filtered_df.iterrows():
    color_card = {"healthy":"#d4f5d4", "vulnerable":"#ffe0b3", "critical":"#ffb3b3"}.get(row["pred_status"], "#f0f0f0")
    with st.expander(row["name"]):
        st.markdown(f"<div style='background-color:{color_card};padding:10px;border-radius:5px;'>"
                    f"**نوع المانغروف:** {row['type']}  \n"
                    f"**درجة الحرارة:** {row['temperature']}  \n"
                    f"**درجة حرارة الصيف:** {row['summer_temperature']}  \n"
                    f"**درجة حرارة الشتاء:** {row['winter_temperature']}  \n"
                    f"**الملوحة:** {row['salinity']}  \n"
                    f"**الرطوبة:** {row['humidity']}  \n"
                    f"**نطاق المد والجزر:** {row['tidal_range']}  \n"
                    f"**نوع التربة:** {row['soil_type']}  \n"
                    f"**تنوع الأنواع:** {row['species_diversity']}  \n"
                    f"**حالة الحماية:** {row['protected_status']}  \n"
                    f"**الحالة البيئية المتوقعة:** {row['pred_status']}</div>", unsafe_allow_html=True)

# ===== جدول البيانات =====
st.subheader("🗂️ جدول المواقع")
st.dataframe(filtered_df.drop(columns=['color','radius']))

# ===== إحصائيات متغيرة حسب الفلترة =====
st.subheader("📊 إحصائيات المواقع بعد الفلترة")
status_counts = filtered_df['pred_status'].value_counts()
type_counts = filtered_df['type'].value_counts()
st.markdown("**توزيع المواقع حسب الحالة البيئية المتوقعة:**")
st.bar_chart(status_counts)
st.markdown("**توزيع المواقع حسب نوع المانغروف:**")
st.bar_chart(type_counts)

# ===== AI لتوقع حالة موقع جديد =====
st.subheader("🤖 توقع الحالة البيئية لموقع جديد")
st.markdown("أدخل بيانات الموقع الجديد لتوقع حالته البيئية:")

new_temp = st.number_input("درجة الحرارة العامة (°C)", value=28)
new_summer_temp = st.number_input("درجة حرارة الصيف (°C)", value=32)
new_winter_temp = st.number_input("درجة حرارة الشتاء (°C)", value=24)
new_salinity = st.number_input("الملوحة (PSU)", value=30)
new_humidity = st.number_input("الرطوبة (%)", value=70)
new_tidal_range = st.number_input("نطاق المد والجزر (m)", value=2.0)

new_data = pd.DataFrame([[new_temp,new_summer_temp,new_winter_temp,new_salinity,new_humidity,new_tidal_range]],
                        columns=[col+'_num' for col in features])
prediction = model.predict(new_data)
pred_status = le.inverse_transform(prediction)[0]
st.success(f"الحالة البيئية المتوقعة للموقع الجديد: **{pred_status}**")
