# AI海外应用分析平台

这是一个基于Streamlit的AI海外应用数据分析平台，用于分析AI企业的营收、融资情况和发展阶段。

## 功能特点

- 📊 企业营收分析
- 💰 融资情况分析
- 📈 发展阶段判断
- 🔍 异常值检测
- 📉 相关性分析
- 🎯 PCA降维可视化

## 部署说明

本项目已部署在Streamlit Cloud上，可以通过以下链接访问：
[AI海外应用分析平台](https://your-app-url.streamlit.app)

## 本地运行

1. 克隆仓库：
```bash
git clone https://github.com/your-username/ai-overseas-analysis.git
cd ai-overseas-analysis
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
streamlit run data.py
```

## 数据说明

数据文件 `AI海外应用 20250408.xlsx` 包含以下字段：
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

MIT License 