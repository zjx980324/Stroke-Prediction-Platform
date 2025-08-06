
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import tempfile
import streamlit.components.v1 as components
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="Dynamic Prediction Platform", layout="wide")
st.title("Dynamic Prediction Platform for the Prognosis of Acute Ischemic Stroke")

# 加载模型预测结果（KNN模型）
with open('knn_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# 变量标签
input_vars = {
    'age': 'Age',
    'Sex': 'Sex',
    'smoking': 'Current cigarette smoking',
    'drinking': 'Current alcohol drinking',
    'HTNhistory': 'History of hypertension',
    'hyperlipidemia': 'History of hyperlipidemia',
    'heartdisease': 'History of coronary heart disease',
    'diabetes': 'History of diabetes mellitus',
    'subtype': 'Ischemic stroke subtype',
    'nihssru': 'Baseline NIH Stroke Scale score',
    'BMI': 'BMI',
    'D1': 'Family history of stroke',
    'HGF': 'HGF, pg/mL',
    'NT-proBNP': 'NT-proBNP, pg/mL',
    'Galectin3': 'Galectin3, ng/mL',
    'Cystatin C': 'Cystatin C, mg/L',
    'GDF-15': 'GDF-15, ng/L',
    'hsCRP': 'hsCRP, mg/L',
    'Osteoprotegerin': 'Osteoprotegerin, pg/mL',
    'Osteopontin': 'Osteopontin, pg/mL',
    'CCL21': 'CCL21, pg/mL',
    'Netrin-1': 'Netrin-1, pg /mL',
    'S100A8/A9': 'S100A8/A9, ng/mL',
    'Thrombomodulin': 'Thrombomodulin, μmol/L',
    'VD': 'VD, ng/mL'
}

# 左侧输入表单
st.sidebar.header("Variable input")
user_input = {}
for var in input_vars.keys():
    if var == 'Sex':
        user_input[var] = st.sidebar.selectbox(
            input_vars[var], [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
    elif var == 'subtype':
        user_input[var] = st.sidebar.selectbox(
            input_vars[var], [1, 2, 3], format_func=lambda x: {1: 'Thrombotic', 2: 'Embolic', 3: 'Lacunar'}[x])
    elif var in ['smoking', 'drinking', 'HTNhistory', 'hyperlipidemia', 'heartdisease', 'diabetes', 'D1']:
        user_input[var] = st.sidebar.selectbox(
            input_vars[var], [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    else:
        user_input[var] = st.sidebar.number_input(input_vars[var], value=0.0)

# 提交按钮
if st.sidebar.button("Submit"):
    input_df = pd.DataFrame([user_input])
    outcome_order = [
        'Month 3 Recurrent stroke', 'Month 3 Vascular events', 'Month 3 Death', 'Month 3 Primary outcome',
        'Yr1 Recurrent stroke', 'Yr1 Vascular events', 'Yr1 Death', 'Yr1 Primary outcome',
        'Yr2 Recurrent stroke', 'Yr2 Vascular events', 'Yr2 Death', 'Yr2 Primary outcome'
    ]

    # 仪表盘展示（每行4列，间距最小）
    st.markdown("### Prediction Probability (KNN)")
    for row_idx in range(0, 12, 4):
        cols = st.columns(4)
        for i in range(4):
            idx = row_idx + i
            if idx >= len(outcome_order):
                break
            outcome = outcome_order[idx]
            model = model_results[idx]['Classifier']
            input_for_model = input_df[model.feature_names_in_]
            prob = model.predict_proba(input_for_model)[0][1] * 100
            with cols[i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={'text': f"<b>{outcome}</b>", 'font': {'size': 14, 'color': 'black'}},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "tomato"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

    # SHAP 力图显示函数（紧凑排布）
    def render_shap_force_plot(base_value, shap_values, features, feature_names):
        force_plot_obj = shap.plots.force(
            base_value,
            shap_values,
            features=features,
            feature_names=feature_names,
            matplotlib=False
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            shap.save_html(tmpfile.name, force_plot_obj)
            html_str = open(tmpfile.name, 'r', encoding='utf-8').read()
        components.html(
            f"""
            <div style="margin: 0px 0px 4px 0px; padding: 5px 10px; border: 1px solid #ccc; overflow-x:auto;">
                {html_str}
            </div>
            """,
            height=400,
            width=1400,
            scrolling=True
        )

    # SHAP 力图区域标题
    st.markdown("### Local Explanation (SHAP Force Plot)")

    # 循环生成12张 SHAP 力图
    if all('X_train' in res for res in model_results):
        for i, res in enumerate(model_results):
            clf = res['Classifier']
            X_train_sample = res['X_train']
            feature_names = list(input_vars.keys())

            # 解释器构建
            explainer = shap.Explainer(clf.predict_proba, X_train_sample, feature_names=feature_names)
            input_for_model = input_df[clf.feature_names_in_]
            shap_values = explainer(input_for_model)

            # 图标题加粗+黑色
            st.markdown(
                f"<h4 style='margin-bottom:2px; margin-top:10px; font-weight: bold; color: black;'>SHAP Force Plot for: {outcome_order[i]}</h4>",
                unsafe_allow_html=True
            )

            render_shap_force_plot(
                shap_values.base_values[0][1],
                shap_values.values[0, :, 1],
                input_for_model.iloc[0],
                shap_values.feature_names
            )
    else:
        st.warning("模型缺少训练数据 (X_train)，无法生成 SHAP 力图。")
else:
    st.info('Please enter the variables on the left side and then click the "Submit" button to display the prediction results and explanations.')

