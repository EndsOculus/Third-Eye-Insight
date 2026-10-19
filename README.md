# 恋之瞳 · Third Eye Insight

恋之瞳是一款本地运行的 QQ 聊天记录分析工具，用于统计群聊和私聊中的发言量、回应轮次、双向会话及回应延迟。

[快速开始](#快速开始) · [使用方法](#使用方法) · [详细文档](#详细文档) · [致谢](#致谢) · [许可证](#许可证)

## 主要功能

- **筛选与浏览**：按日期筛选记录，搜索匿名成员，分页查看结果。
- **互动统计**：查看发言量、回应轮次、双向会话和回应延迟，打开详情查看对应记录。
- **桌面操作**：后台分析、进度提示、取消和关闭；正常启动不弹命令行窗口。
- **本地 AI 辅助**：可选向量话题候选和生成实验，默认关闭。

![恋之瞳桌面：演示数据的互动统计](docs/images/desktop-demo.png)

*界面示例使用程序生成的演示数据。*

## 快速开始

准备 **Windows、Python 3.13 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)**。有 Git 时，在 PowerShell 中执行：

```powershell
git clone https://github.com/EndsOculus/Third-Eye-Insight.git
cd Third-Eye-Insight
uv sync --locked
```

也可从 [Releases](https://github.com/EndsOculus/Third-Eye-Insight/releases) 下载源码包，解压后在项目目录运行 `uv sync --locked`。源码包不是免安装程序；安装依赖可能联网，模型不随源码提供。

随后双击 **Start Insight.vbs**，点击“体验合成演示”即可先了解界面，无需数据库或模型。系统禁用 VBScript 时，使用：

```powershell
.venv\Scripts\pythonw.exe desktop.py
```

升级时先保留自己的数据库和私有配置，再将新版源码解压到新目录并重新安装依赖，不覆盖原始数据。命令行、SQLCipher 与可选模型配置见[使用指南](USAGE.md)。

## 使用方法

1. 选择数据库与输入格式：原生 NT 或结构导入。
2. 选择群聊或私聊，填写群号或自己的 QQ，按需设置日期和排除项。
3. 点击“开始分析”，用搜索和分页浏览结果；双击记录查看详情。
4. 如需本地 AI 辅助，按[使用指南](USAGE.md)配置模型，再选择“向量候选”或“生成实验”。

## 注意事项

- 原生 NT 的引用目标和 @ 收件人解析仍有限制；“未知”表示缺少可用信息，不等于零。加密私有数据库尚未经过真实数据验证。
- AI 辅助可能误判，目前仅用合成数据评测；候选不改变直接统计。互动次数和模型结果都不能用来判断真实感情。
- 分析在本机进行，原始库只读，不向外部 AI 上传聊天。分享结果时请留意日期等信息，并清除截图中的路径和账号；不要上传数据库、私有配置、日志或凭据。
- 数据量与模型选区设有上限，超限时请缩小范围，具体限制见[使用指南](USAGE.md)。

## 详细文档

| 内容 | 文档 |
|---|---|
| 安装、升级、CLI、SQLCipher 与本地模型配置 | [使用指南](USAGE.md) |
| 互动指标、数据覆盖与结构导入 | [互动口径](INTERACTION_V2.md) |
| 本地 AI 配置与评测 | [向量辅助](AI_ASSISTANCE.md) / [生成实验](GENERATION.md) |
| 测试与性能 | [验证记录](VALIDATION.md) / [性能口径](BENCHMARK_V2.md) |
| 隐私与安全 | [安全说明](SECURITY.md) |
| 版本变化 | [更新记录](CHANGELOG.md) |

## 作者与名称

作者：[EndsOculus](https://github.com/EndsOculus)。[查看贡献记录](https://github.com/EndsOculus/Third-Eye-Insight/commits/main/)。

“恋之瞳”的名称灵感来自《东方 Project》中的古明地恋（Komeiji Koishi）与“第三只眼”的意象。

## 致谢

感谢以下开源项目和服务：

[Pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/) · [Matplotlib](https://matplotlib.org/) · [Seaborn](https://seaborn.pydata.org/) · [NetworkX](https://networkx.org/) · [PyTorch](https://pytorch.org/) · [SentenceTransformers](https://www.sbert.net/) · [SciPy](https://www.scipy.org/) · [DeepSeek API](https://api.deepseek.com/)

DeepSeek API 曾用于旧版报告功能，当前桌面与默认 CLI 不调用该服务。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。第三方组件的许可见[第三方说明](THIRD_PARTY.md)。
