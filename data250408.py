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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']  # 添加更多中文字体选项
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

# 初始化变量
df = None
text_cols = []
numeric_cols = []

# 加载数据
if uploaded_file is not None:
    df, text_cols, numeric_cols = load_data(uploaded_file)

# 只有在成功加载数据后才执行后续操作
if df is not None and len(df) > 0:
    # 添加增长率计算
    try:
        # 使用年营收估测计算增长率
        revenue_col = '年营收估测\n（百万美元）'
        funding_col = '上轮融资额\n（百万美元）'
        total_funding_col = '累计融资额\n（百万美元）'
        
        # 计算增长率
        df['增长率'] = df[revenue_col].pct_change() * 100
        
        # 按营收区间分组
        # 定义营收区间
        # 说明：将企业按营收规模分组，便于在同规模企业间进行比较
        # 区间划分依据：
        # 1. 0-10M：初创期企业
        # 2. 10-50M：成长期企业
        # 3. 50-100M：稳定期企业
        # 4. 100-500M：扩张期企业
        # 5. 500M+：成熟期企业
        revenue_bins = [0, 10, 50, 100, 500, float('inf')]
        revenue_labels = ['0-10M', '10-50M', '50-100M', '100-500M', '500M+']
        df['营收区间'] = pd.cut(df[revenue_col], bins=revenue_bins, labels=revenue_labels)
        
        # 计算每个营收区间的融资中位数
        # 说明：使用中位数而不是平均值，避免异常值的影响
        # 分别计算上轮融资和累计融资的中位数，作为该规模企业的"正常"融资水平
        median_funding = df.groupby('营收区间')[funding_col].median()
        median_total_funding = df.groupby('营收区间')[total_funding_col].median()
        
        # 计算融资泡沫指数
        # 说明：泡沫指数反映企业在相同营收规模下的融资水平
        # 计算方法：
        # 1. 上轮融资比值 = 企业上轮融资额 / 所在区间上轮融资中位数
        # 2. 累计融资比值 = 企业累计融资额 / 所在区间累计融资中位数
        # 3. 综合泡沫指数 = (上轮融资比值 + 累计融资比值) / 2
        # 指数含义：
        # - 指数=1：融资水平与同规模企业相当
        # - 指数>1：融资水平高于同规模企业
        # - 指数<1：融资水平低于同规模企业
        df['融资泡沫指数'] = 0.0
        for interval in revenue_labels:
            mask = df['营收区间'] == interval
            if mask.any():
                # 计算上轮融资额与区间中位数的比值
                funding_ratio = df.loc[mask, funding_col] / median_funding[interval]
                # 计算累计融资额与区间中位数的比值
                total_funding_ratio = df.loc[mask, total_funding_col] / median_total_funding[interval]
                # 综合泡沫指数 = (上轮融资比值 + 累计融资比值) / 2
                df.loc[mask, '融资泡沫指数'] = (funding_ratio + total_funding_ratio) / 2
        
        # 根据融资泡沫指数判断发展阶段
        # 说明：发展阶段判断标准基于融资泡沫指数
        # 判断依据：
        # 1. 正常期（指数≤1.5）：
        #    - 融资水平在合理范围内
        #    - 与同规模企业融资水平相近
        #    - 发展较为稳健
        # 2. 扩张期（1.5<指数≤3.0）：
        #    - 融资水平显著高于同规模企业
        #    - 可能存在市场扩张或技术突破
        #    - 需要关注其发展持续性
        # 3. 泡沫期（指数>3.0）：
        #    - 融资水平远高于同规模企业
        #    - 可能存在估值泡沫
        #    - 需要警惕投资风险
        df['发展阶段'] = '正常期'
        df.loc[df['融资泡沫指数'] > 1.5, '发展阶段'] = '扩张期'  # 融资泡沫指数>1.5为扩张期
        df.loc[df['融资泡沫指数'] > 3.0, '发展阶段'] = '泡沫期'  # 融资泡沫指数>3.0为泡沫期
        
        # 显示各营收区间的融资中位数
        # 说明：展示不同规模企业的"正常"融资水平
        # 用途：
        # 1. 了解各规模企业的融资基准
        # 2. 判断企业融资是否异常
        # 3. 为投资决策提供参考
        st.write("📊 各营收区间的融资中位数：")
        median_df = pd.DataFrame({
            '营收区间': revenue_labels,
            '上轮融资中位数': median_funding,
            '累计融资中位数': median_total_funding
        })
        st.dataframe(median_df)
        
        # 显示融资泡沫指数分布
        # 说明：展示所有企业的融资泡沫指数分布情况
        # 分析要点：
        # 1. 均值：反映整体融资水平
        # 2. 标准差：反映融资水平的离散程度
        # 3. 分位数：帮助确定判断阈值
        st.write("📈 融资泡沫指数分布：")
        st.write(df['融资泡沫指数'].describe())
        
        # 显示发展阶段判断结果
        # 说明：展示每个企业的发展阶段判断结果
        # 包含信息：
        # 1. 企业基本信息（公司名、行业）
        # 2. 财务数据（营收、融资额）
        # 3. 分析结果（营收区间、泡沫指数、发展阶段）
        st.write("📋 发展阶段判断结果：")
        display_df = df[text_cols + [revenue_col, '营收区间', funding_col, total_funding_col, '融资泡沫指数', '发展阶段']].copy()
        # 确保所有数值列都是float类型
        for col in [revenue_col, funding_col, total_funding_col, '融资泡沫指数']:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
        # 确保文本列是字符串类型
        for col in text_cols + ['营收区间', '发展阶段']:
            display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df)
        
        # 发展阶段说明
        st.write("""
        📝 发展阶段判断说明：
        1. 判断方法：
           - 按营收区间分组（0-10M, 10-50M, 50-100M, 100-500M, 500M+）
           - 计算每个营收区间的融资中位数
           - 计算各公司的融资泡沫指数 = (上轮融资/区间中位数 + 累计融资/区间中位数) / 2
        
        2. 判断标准：
           - 正常期：融资泡沫指数 ≤ 1.5
           - 扩张期：1.5 < 融资泡沫指数 ≤ 3.0
           - 泡沫期：融资泡沫指数 > 3.0
        
        3. 判断意义：
           - 反映了企业在相同营收规模下的融资水平
           - 可以识别出融资异常的企业
           - 考虑了不同规模企业的特点
        
        4. 分析局限性：
           - 未考虑行业差异
           - 未考虑企业发展阶段
           - 未考虑市场环境因素
        """)
    except Exception as e:
        st.error(f"发展阶段判断错误: {str(e)}")
        st.write("错误时的数据:")
        st.dataframe(df.head().astype(str))
        st.stop()

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

    # ------------------------
    # 相关性分析 + 显著性标注
    # ------------------------
    st.subheader("📎 相关性分析（皮尔逊）")

    # 只选择数值列进行相关性分析
    corr_matrix = df[numeric_cols].corr()
    st.write("📊 相关矩阵：")
    st.dataframe(corr_matrix)

    # 相关性热图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    plt.title('Correlation Heatmap', fontsize=12)

    # 定义列名映射
    column_mapping = {
        '年营收估测\n（百万美元）': 'Annual Revenue (Million USD)',
        '上轮融资额\n（百万美元）': 'Last Round Funding (Million USD)',
        '累计融资额\n（百万美元）': 'Total Funding (Million USD)'
    }

    # 设置英文标签
    ax.set_xticklabels([column_mapping.get(col, col) for col in numeric_cols], rotation=45, ha='right')
    ax.set_yticklabels([column_mapping.get(col, col) for col in numeric_cols], rotation=0)
    st.pyplot(fig)

    # 显著性检验
    st.write("🧪 显著性（p < 0.05）")
    cols = numeric_cols
    sig_result = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = pearsonr(df[cols[i]], df[cols[j]])
            sig_result.append((cols[i], cols[j], round(r, 3), "{:.3e}".format(p), "✅" if p < 0.05 else "❌"))
    sig_df = pd.DataFrame(sig_result, columns=["变量1", "变量2", "相关系数", "p值", "显著性"])
    st.dataframe(sig_df)

    # ------------------------
    # 异常值检测
    # ------------------------
    st.subheader("🚨 异常值检测")

    # 只选择数值列进行异常值检测
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

    # ------------------------
    # PCA 可视化
    # ------------------------
    st.subheader("🧬 PCA Dimensionality Reduction Visualization")

    # 只使用数值列进行PCA分析
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

    # 使用英文标签
    for stage in df['发展阶段'].unique():
        mask = df['发展阶段'] == stage
        ax2.scatter(components[mask, 0], components[mask, 1], 
                   label=stage_mapping.get(stage, stage), alpha=0.7)

    ax2.set_xlabel(f"First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=10)
    ax2.set_ylabel(f"Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=10)
    ax2.set_title("PCA Results and Development Stage Distribution", fontsize=12)
    ax2.legend(fontsize=10)
    st.pyplot(fig2)

    # 添加阶段分析总结
    st.subheader("📋 阶段分析总结")
    st.write("""
    1. 扩张期特征：
       - 增长率 > 20%
       - 通常伴随较高的市场关注度
       - 可能存在估值泡沫风险

    2. 泡沫期特征：
       - 增长率 > 50%
       - 可能存在过度投机
       - 需要警惕市场回调风险

    3. 正常期特征：
       - 增长率相对稳定
       - 市场表现较为理性
       - 投资风险相对较低
    """)

    # 查看系统中已安装的中文字体
    fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist if 'SimHei' in f.name or 'YaHei' in f.name or 'KaiTi' in f.name]
    print(fonts)