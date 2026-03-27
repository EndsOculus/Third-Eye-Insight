# CLAUDE.md（项目：觉之瞳）

## uv 已知问题
- `uv add` 需要 pyproject.toml，本项目无此文件，装包用 `uv pip install`
- MSYS2 Python 排在 PATH 前时 `uv pip install` 报 `Unknown operating system: mingw_x86_64_ucrt_gnu`；需显式指定：
  `uv pip install --python "C:/Users/caiju/AppData/Local/Programs/Python/Python313/python.exe" <包名>`

## 环境
- SQLCipher DLL: `C:/msys64/mingw64/bin/libsqlcipher-0.dll`（mingw64，非 ucrt64）
- 加密配置: `config.json`，`encrypted: true` 启用 SQLCipher 路径
- 运行: `uv run python train.py`
