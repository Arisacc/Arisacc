# MSST 模型目录

当前支持模型：

`deverb_bs_roformer_8_256dim_8depth`

用途：去混响。

运行该模型需要同时存在：

- `deverb_bs_roformer_8_256dim_8depth.ckpt`
- `deverb_bs_roformer_8_256dim_8depth.ckpt.yaml`
- `pymss/MSST` 推理运行时

正式版已在 `local_tools/venv` 中安装 `pymss==2.1.8`。轻量发布版不携带虚拟环境，安装依赖时执行：

```powershell
local_tools\venv\python.exe -m pip install -r requirements-optional.txt
```

项目默认不启用 MSST。启用后，如果运行时或模型文件缺失：

- `error`：阻断任务并返回明确错误。
- `skip`：跳过 MSST，继续执行 FFmpeg、VAD、Inspector、Strategy 和 Curator。

MSST 输出只写入当前 Run 的 `work/preprocessed_msst`，不会覆盖原始音频。
