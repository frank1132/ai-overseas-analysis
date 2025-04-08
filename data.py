import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import io
import matplotlib as mpl
import matplotlib.font_manager
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import base64

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS', 'DejaVu Sans']  # 添加更多字体选项
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置图表样式
plt.style.use('seaborn')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 设置页面
st.set_page_config(
    page_title="AI海外应用分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加页面标题和说明
st.title("📊 AI海外应用数据分析报告")
st.markdown("""
本应用用于分析AI海外应用的发展状况，包括：
- 企业营收分析
- 融资情况分析
- 发展阶段判断
- 异常值检测
""")

# 文件上传功能
uploaded_file = st.file_uploader("请上传数据文件 (AI海外应用 20250408.xlsx)", type=['xlsx'])

# 定义字体回退函数
def plot_with_fallback_font(fig, ax):
    try:
        # 尝试使用中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
        st.pyplot(fig)
    except:
        # 如果中文字体失败，使用英文显示
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        # 将中文标签转换为英文
        if hasattr(ax, 'set_title'):
            ax.set_title(ax.get_title().replace('变量相关性热力图', 'Correlation Heatmap'))
        if hasattr(ax, 'set_xlabel'):
            ax.set_xlabel(ax.get_xlabel().replace('第一主成分', 'First Principal Component'))
        if hasattr(ax, 'set_ylabel'):
            ax.set_ylabel(ax.get_ylabel().replace('第二主成分', 'Second Principal Component'))
        st.pyplot(fig)

# 读取数据
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file is None:
            st.warning("请上传数据文件")
            st.stop()
            
        # 读取数据，保留所有列
        df = pd.read_excel(uploaded_file, sheet_name="source")
        
        # 显示原始数据信息
        st.write("原始数据信息:")
        st.write("列名:", df.columns.tolist())
        st.write("数据类型:", df.dtypes)
        st.write("数据预览:")
        st.dataframe(df.head().astype(str))
        st.write("数据形状:", df.shape)
        
        # 分离文本列和数值列
        text_cols = ['公司名', 'AI行业分类']  # 前两列是公司名和行业
        numeric_cols = ['年营收估测\n（百万美元）', '上轮融资额\n（百万美元）', '累计融资额\n（百万美元）']  # 后三列是数值数据
        
        # 创建新的DataFrame以避免修改原始数据
        processed_df = df.copy()
        
        # 确保文本列是字符串类型
        for col in text_cols:
            processed_df[col] = processed_df[col].fillna('').astype(str)
        
        # 清理数值数据
        for col in numeric_cols:
            # 先转换为字符串，去除可能的空格和特殊字符
            processed_df[col] = processed_df[col].astype(str).str.strip()
            # 将空字符串和'nan'转换为NaN
            processed_df[col] = processed_df[col].replace(['', 'nan', 'NaN', 'None'], np.nan)
            # 转换为数值
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # 删除包含NaN的行
        processed_df = processed_df.dropna()
        
        # 确保数值列都是float类型
        for col in numeric_cols:
            processed_df[col] = processed_df[col].astype(float)
        
        # 按年营收排序
        processed_df = processed_df.sort_values(by='年营收估测\n（百万美元）', ascending=True)
        
        # 显示清理后的数据信息
        st.write("清理后的数据信息:")
        st.write("列名:", processed_df.columns.tolist())
        st.write("数据类型:", processed_df.dtypes)
        st.write("数据预览:")
        st.dataframe(processed_df.head().astype(str))
        st.write("数据形状:", processed_df.shape)
        
        return processed_df, text_cols, numeric_cols
    except Exception as e:
        st.error(f"数据加载错误: {str(e)}")
        st.write("请确保上传正确的数据文件。")
        st.stop()

# 加载数据
if uploaded_file is not None:
    try:
        df, text_cols, numeric_cols = load_data(uploaded_file)
        
        # 显示包含公司名和行业的数据预览
        st.subheader("🔍 数据预览")
        display_df = df[text_cols + ['增长率', '发展阶段']].copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df.head())

        # 描述统计（仅数值列）
        st.subheader("📈 描述性统计")
        st.dataframe(df[numeric_cols].describe().astype(str))

        # 阶段分布统计
        st.subheader("📊 发展阶段分布")
        stage_counts = df['发展阶段'].value_counts()
        st.bar_chart(stage_counts)

        # 显示各阶段公司详情
        st.subheader("📋 各阶段公司详情")
        for stage in ['正常期', '扩张期', '泡沫期']:
            st.write(f"### {stage}公司列表")
            stage_companies = df[df['发展阶段'] == stage][text_cols + ['增长率']]
            display_df = stage_companies.copy()
            for col in display_df.columns:
                display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df)

        # 相关性分析
        st.subheader("📎 相关性分析（皮尔逊）")
        corr_matrix = df[numeric_cols].corr()
        st.write("📊 相关矩阵：")
        st.dataframe(corr_matrix)

        # 相关性热图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        plt.title('Correlation Heatmap', fontsize=12)
        plot_with_fallback_font(fig, ax)

        # 显著性检验
        st.write("🧪 显著性（p < 0.05）")
        cols = numeric_cols
        sig_result = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, p = pearsonr(df[cols[i]], df[cols[j]])
                sig_result.append((cols[i], cols[j], round(r, 3), "{:.3e}".format(p), "✅" if p < 0.05 else "❌"))
        sig_df = pd.DataFrame(sig_result, columns=["Variable 1", "Variable 2", "Correlation", "p-value", "Significant"])
        st.dataframe(sig_df)

        # 异常值检测
        st.subheader("🚨 异常值检测")
        numeric_df = df[numeric_cols]

        # 方法1：Z-score
        z_scores = numeric_df.apply(zscore)
        z_outliers = (z_scores.abs() > 3).any(axis=1)

        # 方法2：IQR
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)

        # 方法3：Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso_outliers = iso.fit_predict(numeric_df)
        iso_mask = iso_outliers == -1

        # 汇总
        combined_outliers = z_outliers | iqr_outliers | iso_mask
        outlier_df = df[text_cols + numeric_cols + ['增长率', '发展阶段']][combined_outliers]

        st.write(f"🔺 Z-score 异常值数量: {z_outliers.sum()}")
        st.write(f"🔺 IQR 异常值数量: {iqr_outliers.sum()}")
        st.write(f"🔺 Isolation Forest 异常值数量: {np.sum(iso_mask)}")
        st.write(f"✅ 合并异常值数量（去重后）: {combined_outliers.sum()}")

        st.dataframe(outlier_df)

        # 下载按钮
        to_download = outlier_df.copy()
        csv = to_download.to_csv(index=False).encode()
        st.download_button("📥 下载异常值数据", csv, "异常值结果.csv", "text/csv")

        # PCA 可视化
        st.subheader("🧬 PCA Dimensionality Reduction Visualization")
        pca = PCA(n_components=2)
        components = pca.fit_transform(numeric_df)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        scatter = ax2.scatter(components[:, 0], components[:, 1], 
                            c=df['增长率'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Growth Rate (%)')

        # 添加阶段标签
        stage_mapping = {
            '正常期': 'Normal Period',
            '扩张期': 'Expansion Period',
            '泡沫期': 'Bubble Period'
        }

        for stage in df['发展阶段'].unique():
            mask = df['发展阶段'] == stage
            ax2.scatter(components[mask, 0], components[mask, 1], 
                       label=stage_mapping[stage], alpha=0.7)

        ax2.set_xlabel(f"First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=10)
        ax2.set_ylabel(f"Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=10)
        ax2.set_title("PCA Results and Development Stage Distribution", fontsize=12)
        ax2.legend(fontsize=10)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        st.write("请检查数据文件是否正确上传。")
else:
    st.warning("请先上传数据文件") 
