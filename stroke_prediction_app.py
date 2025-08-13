
import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# 1. 页面基本设置
# ---------------------------
st.set_page_config(
    page_title="Dynamic Prediction Platform for the Prognosis of Acute Ischemic Stroke",
    layout="wide"
)

st.title("Dynamic Prediction Platform for the Prognosis of Acute Ischemic Stroke")

# ---------------------------
# 2. 读取模型
# ---------------------------
with open("RF.pkl", "rb") as f:
    results = pickle.load(f)

# ---------------------------
# 3. 变量信息
# ---------------------------
model_vars = [
    'hsCRP','HGF','GDF.15','CCL21','Galectin3','S100A8.A9','Thrombomodulin','Osteoprotegerin','Netrin.1',
    'BDNF','CCL19','NT.proBNP','White.Blood.Cell','Triglycerides','LDL.C','Red.Blood.Cell','HDL.C',
    'Baseline.NIH','BMI','Age','SBP','DBP','Ischemic.stroke.subtype','History.of.diabetes.mellitus'
]

display_names = [
    'hsCRP, mg/L','HGF, pg/mL','GDF-15, ng/L','CCL21, pg/mL','Galectin3, ng/mL',
    'S100A8/A9, ng/mL','Thrombomodulin, μmol/L','Osteoprotegerin, pg/mL',
    'Netrin-1, pg /mL','BDNF, pg/mL','CCL19, pg/mL','NT-proBNP, pg/mL',
    'White.Blood.Cell, 10^9/L','Triglycerides , mmol/L','LDL-C, mmol/L',
    'Red.Blood.Cell, 10^12/L','HDL-C, mmol/L','Baseline NIH Stroke Scale score',
    'BMI','Age, y','Systolic blood pressure, mm Hg',
    'Diastolic blood pressure, mm Hg','Ischemic stroke subtype',
    'History of diabetes mellitus'
]

# ---------------------------
# 4. 变量输入（3行×8列）
# ---------------------------
st.subheader("Variable input")
user_input = {}
cols = st.columns(8)
for i, (var, disp) in enumerate(zip(model_vars, display_names)):
    with cols[i % 8]:
        if var == 'Ischemic.stroke.subtype':
            user_input[var] = st.selectbox(
                disp,
                options=[1, 2, 3],
                format_func=lambda x: {1: 'Thrombotic', 2: 'Embolic', 3: 'Lacunar'}[x]
            )
        elif var == 'History.of.diabetes.mellitus':
            user_input[var] = st.selectbox(
                disp,
                options=[0, 1],
                format_func=lambda x: {0: 'No', 1: 'Yes'}[x]
            )
        else:
            user_input[var] = st.number_input(disp, value=0.0)

# 提交按钮
submit = st.button("Submit")

# ---------------------------
# 5. 如果点击 Submit 执行预测
# ---------------------------
if submit:
    input_df = pd.DataFrame([user_input])

    # 中部仪表盘布局
    st.subheader("Prediction Gauges")
    gauge_cols = st.columns(3)

    for i, outcome in enumerate(results):
        clf = outcome['Classifier']
        prob = clf.predict_proba(input_df)[:, 1][0] * 100

        # 绘制仪表盘
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': f"<b>{outcome['Outcome']}</b>", 'font': {'size': 24, 'color': 'black'}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "lightcoral"}
                   ]}
        ))
        gauge_cols[i].plotly_chart(fig, use_container_width=True)

    # 底部 SHAP 瀑布图
    st.subheader("SHAP Waterfall Plots")
    for i, outcome in enumerate(results):
        clf = outcome['Classifier']

        # 用 TreeExplainer 保证和官网一致的颜色
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(input_df)

        # 如果是二分类，shap_values 形状是 (n_samples, n_features, 2)
        if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            shap_value_for_class1 = shap_values[:,:,1]  # 取类别=1的SHAP值
        else:
            shap_value_for_class1 = shap_values

        # 在图上加标题
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_value_for_class1[0], show=False)
        plt.title(outcome['Outcome'], fontsize=20, fontweight='bold', color='black', pad=20)
        st.pyplot(plt.gcf(), clear_figure=True)

