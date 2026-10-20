# NT 前缀数据库的隔离只读兼容

这项功能兼容原始 NT 数据库的 1024 字节前缀及 1 GiB 锁字节页布局，不进行数据库修复。普通 SQLite 和普通 SQLCipher 继续走标准读取路径。不要根据文件名、一次 malformed 或索引计数启用兼容。

## 启用条件与边界

原始文件必须为不透明加密文件，按现有密文参数识别出 1024 字节前缀；页大小必须为 4096，文件跨过 1 GiB 边界；偏移后的第 262144 页全零，第 262145 页包含数据。子进程还必须成功解密并验证 `c2c_msg_table` / `group_msg_table` 的 NT 字段及实际页大小。去前缀副本没有原始偏移来源，不自动调整，建议选择原始前缀文件。

只有 Windows、SQLite **3.46.1** / SQLCipher **4.6.1 community** 已验证。其他版本、页大小或平台明确拒绝兼容模式，不猜测修复。没有新增依赖、下载或自编二进制。

现有依赖没有每连接的保留页配置。专用 `tei-nt-readonly` 子进程在默认映射下验证结构和版本，关闭全部连接后，经现有 `sqlite3_test_control(SQLITE_TESTCTRL_PENDING_BYTE)` 将锁字节改为 `0x3ffffc00`，再以 `mode=ro&immutable=1` 重开。该接口属于 SQLite **测试接口，不保证跨版本稳定**，所以使用精确版本白名单且遇到未知状态立即停止。未修改 DLL 代码、符号地址或文件；不在 GUI、分析进程或其他 SQLite 连接上调整映射。后续版本支持必须重新验证，不能直接放宽白名单。

每个读取器只服务一个快照。Windows Job 在启动门打开前拥有子进程；GUI 的外层 Job 同时拥有分析进程及全部后代。取消、关闭、错误、提前停止迭代会回收读取器。IPC 使用有界 JSON、批次背压和固定错误码。正文只通过本机受控管道进入分析内存，不写入日志。

沿用现有最多 4 GiB、120 秒预算的临时去前缀副本流程；复制同时计算原始 SHA-256 与载荷 SHA-256。复制只写工作目录，结束后清理，不新增持久聊天缓存。非空 WAL/journal、来源变化、空间不足、超时和取消继续拒绝返回部分统计。意外终止命令行宿主可能留下其已加密临时副本，GUI 会按自己持有的 scratch 目录清理；不要对陌生目录批量删除。

## 本地运行

先激活现有 Python 3.13 环境，不运行 `uv sync` 或下载依赖。在此工作树启动 GUI：

```powershell
$env:TEI_CONFIG_FILE = 'D:\test\code3\config.json'
& 'D:\test\code3\.venv\Scripts\python.exe' desktop.py
```

配置只通过正常本地加载器读取，不复制到工作树，密钥值不会出现在日志或命令行。GUI 选择原始 `nt_msg.db`、群号和需要的日期范围，关闭 AI 辅助即可运行。成功后上下文显示“NT 前缀兼容 / 隔离只读”；普通库显示“标准只读”。原始来源和载荷 SHA-256、前缀长度、兼容模式、引擎版本及实际选中行数保留在结果 `source` 元数据中，来源哈希亦在详情窗口可见。哈希是文件溯源，不代表历史消息完整或身份语义已验证。

只输出计数和溯源元数据的演示：

```powershell
& 'D:\test\code3\.venv\Scripts\python.exe' nt_readonly_demo.py --db 'D:\test\code3\nt_msg.db' --config 'D:\test\code3\config.json'
```

它会临时复制一次加密载荷，执行群聊表计数并清理，不输出正文、导入训练、下载模型或上传信息。

## 本次验证

本实现保留导入字段边界、会话冲突证据及去重别名等既有修复。安全依赖更新由主线维护；读取兼容不能替代依赖安全修复。本功能不启动训练或自动下载模型。

本地一致快照已通过 `open_database` 完整路径的表计数、关键页正文读取、来源哈希前后不变和父进程映射不变验证，临时副本已清理。实际消息计数、来源指纹及私有路径只保存在使用者的独立本地验证目录，不随仓库上传。

验证分为表计数、关键页正文和已有全量证据，不因重复测试无必要地复制大文件或重复长扫描。当前软件的有界原生解析器不会把未知类型或引用缓存猜作正文/接收者；不能把其他解析器的统计冒充当前软件的全量解析结果。

测试覆盖普通库、前缀库、错误文件/偏移/页大小、真正的加密页破坏、未知引擎拒绝、取消/重复/并行、部分迭代、写操作拒绝、父进程映射不变、外层 Job 回收内层读取器，以及 python/pythonw 和 Windows GUI 生命周期。Linux/macOS 兼容模式、其他 SQLCipher 版本及历史 WAL 完整性未经验证。

诊断和验证证据应保存在独立本地目录，不提交数据库、配置、密钥、模型或聊天正文。不要自动把读取结果加入训练或盲测数据。

## 给其他本地工具复用

稳定入口是安装目录内 `database_access.py` 的 `open_database`，依赖同目录的 `isolated_database.py`、`local_process.py`。调用者使用非 daemon 主进程或独立助手进程，并保留 Windows spawn 主入口保护。先把安装目录加入模块路径，确认没有同名模块被其他项目提前加载；有命名冲突时改用专用助手进程，不替换现有 `sys.modules`。

```python
import multiprocessing as mp
import sys
from pathlib import Path

APP = Path(r'D:\test\code3')
sys.path.insert(0, str(APP))
import database_access as reader_api

def main():
    if Path(reader_api.__file__).resolve() != (APP / 'database_access.py').resolve():
        raise RuntimeError('Reader module namespace conflict')
    with reader_api.open_database(
        r'D:\private\nt_msg.db',
        config_file=APP / 'config.json',
        scratch_dir=r'D:\private\reader-scratch',
    ) as (reader, encrypted, receipt):
        count = reader.query('SELECT count(*) AS n FROM group_msg_table NOT INDEXED')[0]['n']
        # Only publish authorized local metadata; never log message bodies.
        print(receipt['compatibility_mode'], count)

if __name__ == '__main__':
    mp.freeze_support()
    main()
```

`config_file` 为每次调用的配置路径，通过既有加载器读取，不修改进程环境；不要同时传 `config` 和 `config_file`。省略时仍支持 `TEI_CONFIG_FILE` 或安装目录默认配置。配置值不放命令行或日志。调用者自己的 scratch 目录须事先存在、有足够空间且仅用于该任务。

`query` 返回有界字典列表，`iter_query` 逐批返回字典；使用者必须在 `with` 内消费完迭代器。兼容 SQLCipher 路径不接受绑定参数；使用固定表/列名，对群号和时间等只允许先转为整数的受控输入，不拼接任意字符串。BLOB 应以 `hex` 在接口中读取，不猜测 UTF-8 正文或归属。`cancelled` 回调可终止读取，错误不代表可使用此前收到的部分结果。

`receipt` 包含原始/载荷 SHA-256、前缀长度、只读/隔离模式和引擎版本；实际选取行数由调用者一并记录。接入其他工具不会授权修改其数据集、训练集或冻结评测，来源、身份、时间和去重约束仍须独立验证。
