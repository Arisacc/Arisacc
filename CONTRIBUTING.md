# 开发说明

## 代码边界

- `audio_dataset_agent`：核心流水线、Agent、工具适配器和 Web API。
- `frontend`：React/Vite 本地工作台。
- `knowledge_base`：本地音频工具与数据规范资料。
- `tests`：规则、工具适配器、流水线和 Web API 测试。
- `local_tools`：可选工具下载脚本与说明，不提交模型和缓存。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
cd frontend
npm ci
npm run build
```

提交前请确认没有加入 `datasets/`、`local_tools/cache/`、`local_tools/models/`、`.venv/`、`frontend/node_modules/`、密钥或个人文件路径。
