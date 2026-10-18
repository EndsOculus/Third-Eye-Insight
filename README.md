# 恋之瞳 · Third Eye Insight

本地聊天互动分析工具。群聊使用语义、时间窗口内的行为共现和关系网络；私聊使用行为对称、词汇上下文和时间黏性。分数是互动特征，不是对真实情感或个人关系的判断。

## 安装与启动

Windows、Python 3.13，在项目目录执行 `uv sync --locked`。双击 **Start Insight.vbs** 打开桌面应用，启动器和后台分析使用 pythonw，不打开命令行窗口。若 Windows 禁用 VBScript，可直接使用 `.venv\Scripts\pythonw.exe desktop.py` 或为该命令创建快捷方式。

调试入口：`debug_desktop.ps1`。原有完整 CLI：`run_train.ps1` 或 `.venv\Scripts\python.exe train.py`。

## 桌面功能

- 数据库选择、群聊/私聊、账号校验、起止日期、聚焦与排除账号。
- SQLCipher 可选读取本地 config.json；默认不读取配置且只打开明文 SQLite。
- 独立后台进程、阶段进度、取消、关闭清理、重复启动保护。
- 匿名成员关系表、前三名关系柱图、搜索防抖、每页 100 行。
- 合成演示不依赖模型下载/API；随机向量仅用于体验界面。

桌面不调用 AI API，不输出正文或真实姓名/账号，不导出私人数据。日志只含固定事件名，位于 output/logs/desktop.log，自动轮转。输入框中的路径和账号仍是用户输入，截图前请清空。

## 数据与模型

支持原项目 NTQQ SQLite 架构（group_msg_table、c2c_msg_table），不是任意聊天导出格式。数据库只读打开，不自动裁剪文件头、解密生成文件或删除原库；被占用或损坏的数据请使用用户自行取得的完整副本，不强制关闭客户端。

群聊和私聊都需要可信的本地 paraphrase-multilingual-MiniLM-L12-v2 缓存。存在项目 .cache/huggingface 时桌面设置该目录为默认 HF_HOME；不自动联网下载。私聊的上下文指标也使用相邻回复向量相似度；仅合成演示不需要本地模型。旧 pickle 缓存不会加载或删除，新缓存使用原子替换 JSON，仍属于私人派生数据，不上传。

桌面保守上限为提取 50,000 条消息、300 名成员，超限拒绝分析，不显示截断结果。日期在提取后筛选，因此大库需先提供较小的只读副本。匿名编号仅在本次分析内有效。

CLI 保留 PostgreSQL、图片/CSV、聚焦精简模式和可选 AI 报告。远程模式需自行安装适合服务端的 SQLAlchemy/PostgreSQL 驱动。AI 报告仅由用户明确选择启用，会向第三方发送分析资料，本次未测试。CLI 输出可能含私人信息，留在忽略的 output 目录。

依赖锁不再引用本机 CUDA wheel，默认从 PyPI 安装，已验证 CPU 路径。GPU 开关仍保留，需要受支持、与驱动匹配的 CUDA PyTorch；未测量 GPU 吞吐。不要使用旧 wheel 辅助脚本降级到 2.6。

## 测试与回滚

`uv sync --locked --group dev` 后运行 `.venv\Scripts\python.exe -m pytest -q`。GUI 测试需要可用 Windows/Tk 会话，使用隐藏窗口和合成数据。

详见 [VALIDATION.md](VALIDATION.md) 和 [SECURITY.md](SECURITY.md)。本机已备份升级前源码和 Git bundle，未纳入仓库。基线为 2d44d85；需要回滚时在新分支恢复基线源码，保留私人配置/数据库，不用 hard reset 覆盖未提交工作。旧依赖存在已知告警，恢复前应评估。

语义归一化只对实际观察到窗口内互动的成员对进行确定性中位排名；无观察的成员对保持零分，相同相似度共享排名。仅一对可观察时使用中性分位 0.5（离散化后约 0.5005）。因此与旧版随机抖动排名相比，零信息/并列样本的分数会有意改变；CLI 和桌面采用同一修正规则。
