# Streamlit Cloud 部署指南

本指南将帮助您在Streamlit Cloud上部署AI海外应用分析平台，使其可以被所有人访问。

## 部署步骤

### 1. 准备GitHub仓库

1. 在GitHub上创建一个新仓库，例如 `ai-overseas-analysis`
2. 将以下文件上传到仓库：
   - `data250408.py`（主应用程序文件）
   - `requirements.txt`（依赖项列表）
   - `README.md`（项目说明）
   - `.gitignore`（Git忽略文件）
   - `AI海外应用 20250408.xlsx`（数据文件）

### 2. 部署到Streamlit Cloud

1. 访问 [Streamlit Community Cloud](https://streamlit.io/cloud)
2. 使用GitHub账号登录
3. 点击"New app"按钮
4. 选择您的GitHub仓库 `ai-overseas-analysis`
5. 在"Main file path"中输入 `data250408.py`
6. 点击"Deploy"按钮

### 3. 配置应用

1. 部署完成后，您将获得一个公共URL，例如 `https://ai-overseas-analysis.streamlit.app`
2. 在Streamlit Cloud的"Settings"中，您可以：
   - 设置应用名称
   - 配置环境变量（如需要）
   - 设置资源限制

## 注意事项

- 确保您的数据文件 `AI海外应用 20250408.xlsx` 已上传到GitHub仓库
- 如果您的应用需要额外的环境变量或配置，请在Streamlit Cloud的"Settings"中设置
- 免费版本的Streamlit Cloud有一些资源限制，如果您的应用需要更多资源，可能需要升级到付费版本

## 故障排除

如果部署后应用无法正常运行，请检查：

1. 控制台日志中是否有错误信息
2. 数据文件是否正确上传
3. 所有依赖项是否在`requirements.txt`中列出
4. 主文件路径是否正确

## 技术支持

如果您在部署过程中遇到问题，可以：

1. 查看 [Streamlit Cloud文档](https://docs.streamlit.io/streamlit-community-cloud)
2. 在 [Streamlit论坛](https://discuss.streamlit.io/) 寻求帮助
3. 提交GitHub issue到您的仓库 