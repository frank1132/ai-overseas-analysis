# AI海外应用分析平台

这是一个用于分析AI海外应用公司营收、融资和发展阶段的数据分析平台。

## 功能特点

- 企业营收分析
- 融资情况分析
- 发展阶段判断
- 异常值检测
- 相关性分析
- PCA降维可视化

## 在线访问

您可以通过以下链接访问此应用：

[AI海外应用分析平台](https://ai-overseas-analysis.streamlit.app)

## 本地运行

1. 克隆仓库
```bash
git clone https://github.com/yourusername/ai-overseas-analysis.git
cd ai-overseas-analysis
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
streamlit run data250408.py
```

## 数据说明

应用使用`AI海外应用 20250408.xlsx`文件作为数据源，包含以下字段：
- 公司名
- AI行业分类
- 年营收估测（百万美元）
- 上轮融资额（百万美元）
- 累计融资额（百万美元）

## 技术栈

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 许可证

MIT 