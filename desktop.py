"""Responsive Tk desktop, with an isolated cancellable analysis process."""
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk, filedialog

from desktop_core import Request

EVIDENCE = {"coverage_unknown": "结构覆盖不足", "no_observed_response": "未观测到回应", "insufficient": "样本不足", "observed": "已观测（未验证效度）"}

PAGE_SIZE = 100

class Desktop(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=14)
        self.root = root
        root.title("Third Eye Insight · 恋之瞳")
        root.geometry("1120x780")
        root.minsize(900, 700)
        self.pack(fill="both", expand=True)
        self.process = None
        self.events = None
        self.closing = False
        self.cancelling = False
        self.finished = False
        self.rows = []
        self.result = {}
        self.filtered = []
        self.page = 0
        self.search_timer = None
        style = ttk.Style(root)
        style.theme_use("clam")
        surface, ink, muted, accent = "#f4f5f7", "#303744", "#667085", "#635bff"
        root.configure(background=surface)
        style.configure(".", font=("Microsoft YaHei UI", 10), background=surface, foreground=ink)
        style.configure("TFrame", background=surface)
        style.configure("TLabel", background=surface, foreground=ink)
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Card.TLabel", background="#ffffff", foreground=ink)
        style.configure("Section.TLabel", background="#ffffff", foreground="#202632", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Title.TLabel", foreground="#202632", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Sub.TLabel", foreground=muted)
        style.configure("Badge.TLabel", background="#eeecff", foreground="#554ad4", padding=(10, 5), font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Metric.TLabel", background="#f6f7fa", foreground="#475467", padding=(10, 7), font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", foreground=muted)
        style.configure("Error.TLabel", foreground="#b42318")
        style.configure("Success.TLabel", foreground="#067647")
        style.configure("TEntry", foreground=ink, fieldbackground="#f8fafc", padding=(8, 5), relief="flat", borderwidth=1, bordercolor="#e4e7ec", lightcolor="#e4e7ec", darkcolor="#e4e7ec", insertcolor=ink)
        style.map("TEntry", bordercolor=[("focus", accent)], lightcolor=[("focus", accent)], darkcolor=[("focus", accent)])
        style.configure("TButton", foreground="#475467", background="#eaedf2", padding=(14, 7), relief="flat", borderwidth=0, bordercolor="#eaedf2", lightcolor="#eaedf2", darkcolor="#eaedf2", focusthickness=0)
        style.map("TButton", background=[("disabled", "#f0f2f5"), ("active", "#e0e5ec")], foreground=[("disabled", "#98a2b3"), ("active", "#303744")], bordercolor=[("focus", accent)], lightcolor=[("disabled", "#f0f2f5"), ("active", "#e0e5ec")], darkcolor=[("disabled", "#f0f2f5"), ("active", "#e0e5ec")], relief=[("pressed", "flat"), ("!pressed", "flat")])
        style.configure("Accent.TButton", foreground="#ffffff", background=accent, bordercolor=accent, lightcolor=accent, darkcolor=accent)
        style.map("Accent.TButton", foreground=[("disabled", "#ffffff"), ("!disabled", "#ffffff")], background=[("disabled", "#c8c4ee"), ("active", "#5148e5")], bordercolor=[("disabled", "#c8c4ee"), ("active", "#5148e5")], lightcolor=[("disabled", "#c8c4ee"), ("active", "#5148e5")], darkcolor=[("disabled", "#c8c4ee"), ("active", "#5148e5")])
        for control in ("TRadiobutton", "TCheckbutton"):
            style.configure(control, background="#ffffff", foreground=ink, indicatorbackground="#ffffff", indicatorforeground=accent, bordercolor="#d0d5dd", lightcolor="#ffffff", darkcolor="#ffffff")
            style.map(control, background=[("active", "#ffffff")], foreground=[("disabled", "#98a2b3"), ("!disabled", ink)])
        style.configure("Horizontal.TProgressbar", background=accent, troughcolor="#e9e7fc", bordercolor=surface, lightcolor=accent, darkcolor=accent, borderwidth=0, thickness=5)
        style.configure("Treeview", rowheight=30, fieldbackground="#ffffff", background="#ffffff", foreground=ink, borderwidth=0, relief="flat")
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.map("Treeview", background=[("selected", "#eeecff")], foreground=[("selected", "#4438a8")])
        style.configure("Treeview.Heading", background="#f3f5f8", foreground="#667085", font=("Microsoft YaHei UI", 9, "bold"), padding=(8, 7), relief="flat", borderwidth=0, bordercolor="#f3f5f8", lightcolor="#f3f5f8", darkcolor="#f3f5f8")
        style.layout("Treeview.Heading", [("Treeheading.cell", {"sticky": "nswe"}), ("Treeheading.padding", {"sticky": "nswe", "children": [("Treeheading.text", {"sticky": "we"})]})])
        style.map("Treeview.Heading", background=[("active", "#f3f5f8")], relief=[("active", "flat")])
        style.configure("Vertical.TScrollbar", background="#d5dae3", troughcolor="#ffffff", bordercolor="#ffffff", lightcolor="#d5dae3", darkcolor="#d5dae3", arrowcolor="#667085", borderwidth=0, relief="flat", arrowsize=12)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="恋之瞳  /  Third Eye Insight", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="LOCAL  /  OFFLINE", style="Badge.TLabel").grid(row=0, column=1, sticky="e")
        self.context = tk.StringVar(value="互动 v2 · 只读数据 · 不推断情感或亲密度")
        ttk.Label(self, textvariable=self.context, style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 8))
        form = ttk.Frame(self, style="Card.TFrame", padding=(14, 10))
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        ttk.Label(form, text="分析设置", style="Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        self.values = {k: tk.StringVar() for k in ("db", "identifier", "start", "end", "focus", "exclude")}
        ttk.Label(form, text="数据库", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.values["db"]).grid(row=1, column=1, columnspan=3, sticky="ew", padx=10)
        ttk.Button(form, text="选择文件", command=self.browse).grid(row=1, column=4)
        self.mode = tk.StringVar(value="group")
        modes = ttk.Frame(form, style="Card.TFrame")
        modes.grid(row=2, column=0, columnspan=5, sticky="w", pady=4)
        ttk.Radiobutton(modes, text="群聊", variable=self.mode, value="group").pack(side="left")
        ttk.Radiobutton(modes, text="私聊（填我的 QQ）", variable=self.mode, value="c2c").pack(side="left", padx=12)
        for index, (key, title) in enumerate((("identifier", "群号 / 我的 QQ"), ("focus", "聚焦账号（可选）"), ("start", "开始 YYYY-MM-DD"), ("end", "结束 YYYY-MM-DD"))):
            r, c = 3 + index // 2, (index % 2) * 2
            ttk.Label(form, text=title, style="Card.TLabel").grid(row=r, column=c, sticky="w", pady=2)
            ttk.Entry(form, textvariable=self.values[key]).grid(row=r, column=c+1, sticky="ew", padx=10, pady=2)
        self.flags = {k: tk.BooleanVar() for k in ("encrypted", "ai_assist")}
        self.source_format = tk.StringVar(value="原生 NT")
        options = ttk.Frame(modes, style="Card.TFrame")
        options.pack(side="left", padx=(12, 0))
        for key, label in (("encrypted", "SQLCipher"), ("ai_assist", "本地话题候选")):
            ttk.Checkbutton(options, text=label, variable=self.flags[key]).pack(side="left", padx=(0, 12))
        ttk.Combobox(options, textvariable=self.source_format, values=("原生 NT", "结构导入"), state="readonly", width=8).pack(side="left", padx=(0,8))
        ttk.Label(modes, text="排除（逗号）", style="Card.TLabel").pack(side="left", padx=(2, 4))
        ttk.Entry(modes, textvariable=self.values["exclude"], width=16).pack(side="left")
        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, sticky="ew", pady=8)
        self.run_button = ttk.Button(controls, text="开始分析", style="Accent.TButton", command=self.start)
        self.run_button.pack(side="left")
        self.demo_button = ttk.Button(controls, text="体验合成演示", command=lambda: self.start(demo=True))
        self.demo_button.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(controls, text="取消", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left")
        self.progress = ttk.Progressbar(controls, maximum=100, length=180)
        self.status = tk.StringVar(value="准备就绪。先体验演示，或选择只读数据库。")
        self.status_label = ttk.Label(self, textvariable=self.status, style="Status.TLabel", wraplength=840)
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        panel = ttk.Frame(self, style="Card.TFrame", padding=(12, 10))
        panel.grid(row=5, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1, minsize=120)
        ttk.Label(panel, text="互动证据 · 按回应轮次排序", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.summary = tk.StringVar(value="暂无结果 · 日期按北京时间；结束日完整包含")
        ttk.Label(panel, textvariable=self.summary, style="Metric.TLabel", wraplength=790).grid(row=1, column=0, sticky="ew")
        self.chart = tk.Canvas(panel, height=44, background="white", highlightthickness=0)
        self.chart.grid(row=2, column=0, sticky="ew", pady=5)
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        table_frame = ttk.Frame(panel, style="Card.TFrame")
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("a", "b", "responses", "sessions", "messages", "days", "oneway", "evidence")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=4)
        for col, label, width in zip(columns,
                ("成员 A", "成员 B", "回应轮次 ↓", "双向会话", "发言 A / B", "共同活跃日", "单向程度", "证据状态"),
                (85, 85, 90, 80, 100, 90, 85, 145)):
            self.table.heading(col, text=label)
            self.table.column(col, width=width, minwidth=65, anchor="center")
        self.table.bind("<<TreeviewSelect>>", lambda _: self.draw_chart())
        self.table.bind("<Double-1>", lambda _: self.show_details())
        self.table.bind("<Return>", lambda _: self.show_details())
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.tag_configure("stripe", background="#fafbfc")
        footer = ttk.Frame(panel, style="Card.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(footer, text="查找匿名成员", style="Card.TLabel").pack(side="left")
        self.search = tk.StringVar()
        self.search.trace_add("write", self.schedule_filter)
        ttk.Entry(footer, textvariable=self.search, width=18).pack(side="left", padx=8)
        ttk.Button(footer, text="查看证据", command=self.show_details).pack(side="left")
        self.next_button = ttk.Button(footer, text="下一页", command=lambda: self.change_page(1))
        self.next_button.pack(side="right")
        self.page_text = tk.StringVar(value="0 / 0")
        ttk.Label(footer, textvariable=self.page_text, style="Card.TLabel").pack(side="right", padx=12)
        self.prev_button = ttk.Button(footer, text="上一页", command=lambda: self.change_page(-1))
        self.prev_button.pack(side="right")
        self.render_page()
        self.poll_id = self.after(80, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def browse(self):
        path = filedialog.askopenfilename(title="选择数据库（只读）", filetypes=[("数据库", "*.db *.sqlite *.sqlite3"), ("所有文件", "*")])
        if path:
            self.values["db"].set(path)

    def start(self, demo=False):
        if self.process is not None or self.closing:
            return
        request = Request(demo=True, mode=self.mode.get(), ai_assist=self.flags["ai_assist"].get()) if demo else Request(mode=self.mode.get(), source_format="nt" if self.source_format.get()=="原生 NT" else "tei", **{k:v.get().strip() for k,v in self.values.items()}, **{k:v.get() for k,v in self.flags.items()})
        try:
            request.validate()
        except ValueError as exc:
            self.status_label.configure(style="Error.TLabel")
            self.status.set(str(exc))
            return
        from desktop_core import worker
        context = mp.get_context("spawn")
        self.events = context.Queue()
        self.process = context.Process(target=worker, args=(request, self.events), daemon=True)
        try:
            self.process.start()
        except Exception:
            self.process = None
            self.events.close()
            self.events = None
            self.status.set("无法启动后台进程，请检查 Python 环境。")
            return
        self.finished = False
        self.cancelling = False
        self.context.set("合成演示 · 已标注回复 / 未解析内容 · 无真实聊天" if demo else "互动 v2 · 只读数据 · 不推断情感或亲密度")
        self.rows = []
        self.filter_rows()
        self.summary.set("分析进行中…")
        self.status_label.configure(style="Status.TLabel")
        self.status.set("准备后台分析…")
        self.progress.pack(side="right", padx=8)
        self.progress["value"] = 0
        self.run_button.configure(state="disabled")
        self.demo_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        logging.getLogger("desktop").info("analysis_started")

    def cancel(self):
        if self.process is not None and not self.cancelling:
            self.cancelling = True
            self.cancel_button.configure(state="disabled")
            self.status.set("正在取消…")
            if self.process.is_alive():
                self.process.terminate()

    def drain_events(self):
        if self.events is not None and not self.cancelling:
            for _ in range(20):
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    self.progress["value"], text = payload
                    self.status.set(text)
                elif kind == "result":
                    self.finished = True
                    self.context.set(("合成演示 · " if payload.get("demo") else "只读分析 · ") + ("私聊" if payload.get("mode")=="c2c" else "群聊") + " · 互动 v2 · 不推断亲密度")
                    self.result = payload
                    self.rows = payload["pairs"]
                    self.filter_rows()
                    obs = payload.get("observation", {})
                    cov = payload.get("coverage", {})
                    source = payload.get("source", {})
                    self.summary.set(f"{payload['rows']:,} 条去重记录 · {payload['users']} 名成员 · {len(self.rows):,} 对 · {payload['seconds']:.2f} 秒\n"
                                     f"实际覆盖 {obs.get('actual_start') or '无'} 至 {obs.get('actual_end') or '无'} · 结构已知 {cov.get('structured_rows', 0)}/{cov.get('total_rows', 0)}")
                    assist = payload.get("assistance", {})
                    ai_note = {"disabled":"", "completed":f" 本地话题候选 {assist.get('candidate_count',0)} 条（非回应事实）。", "model_unavailable":" 本地模型不可用，已保留规则统计。", "limit_exceeded":" AI 选区超过 2000 条；未运行辅助。"}.get(assist.get("status"), "")
                    issues = cov.get('unresolved_references', 0) + cov.get('conflicting_ids', 0) + source.get('invalid_time', 0) + source.get('invalid_sender', 0)
                    self.status.set("完成 · 真实数据测量效度尚未验证。双击记录或按 Enter 查看统计口径。" + (" 存在未解析/冲突/无效记录，见证据详情。" if issues else "") + ai_note)
                    self.status_label.configure(style="Success.TLabel")
                    self.progress["value"] = 100
                    self.progress.pack_forget()
                    logging.getLogger("desktop").info("analysis_completed")
                elif kind == "error":
                    self.finished = True
                    self.status_label.configure(style="Error.TLabel")
                    self.progress.pack_forget()
                    self.status.set(payload)
                    self.summary.set("没有生成结果 · 可调整输入后重试")
                    logging.getLogger("desktop").info("analysis_failed")

    def poll(self):
        self.drain_events()
        if self.process is not None and not self.process.is_alive():
            self.process.join(timeout=0)
            # Normal worker exit joins its queue feeder; drain final events before closing.
            # Cancellation can interrupt a queue write, so drain_events skips it.
            self.drain_events()
            self.process.close()
            self.progress.pack_forget()
            self.process = None
            self.events.close()
            self.events = None
            if self.cancelling:
                self.status.set("已取消。原始数据未修改。")
                self.summary.set("已取消 · 可重新开始")
                self.progress["value"] = 0
            elif not self.finished:
                self.status.set("后台进程意外退出，请检查本地依赖。")
            self.run_button.configure(state="normal")
            self.demo_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
        if self.closing and self.process is None:
            self.root.destroy()
            return
        self.poll_id = self.after(80, self.poll)

    def schedule_filter(self, *_):
        if self.search_timer:
            self.after_cancel(self.search_timer)
        self.search_timer = self.after(180, self.filter_rows)

    def filter_rows(self):
        self.search_timer = None
        needle = self.search.get().strip().casefold()
        self.filtered = [r for r in self.rows if not needle or needle in (r["a"]+" "+r["b"]).casefold()]
        self.page = 0
        self.render_page()
        self.draw_chart()

    def change_page(self, delta):
        last = max(0, (len(self.filtered)-1)//PAGE_SIZE)
        self.page = min(last, max(0, self.page + delta))
        self.render_page()

    def render_page(self):
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        for index, row in enumerate(self.filtered[self.page*PAGE_SIZE:(self.page+1)*PAGE_SIZE]):
            self.table.insert("", "end", iid=str(self.page*PAGE_SIZE+index), values=self.row_values(row), tags=("stripe",) if index % 2 else ())
        pages = (len(self.filtered)+PAGE_SIZE-1)//PAGE_SIZE
        self.page_text.set(f"{self.page+1 if pages else 0} / {pages}")
        self.prev_button.configure(state="normal" if self.page else "disabled")
        self.next_button.configure(state="normal" if self.page+1 < pages else "disabled")

    @staticmethod
    def row_values(row):
        unknown = row['evidence'] == 'coverage_unknown'
        def count(key):
            value = row[key]
            return ('已观测 '+str(value) if value else '未知') if unknown else str(value)
        imbalance = row.get('one_way_fraction')
        return (row['a'], row['b'], count('response_rounds'), count('bidirectional_sessions'),
                f"{row['source_messages_a']} / {row['source_messages_b']}", count('common_active_days'),
                '未知' if unknown else ('未定义' if imbalance is None else f"{imbalance:.0%}"), EVIDENCE[row['evidence']])

    def selected_row(self):
        selection = self.table.selection()
        if selection:
            index = int(selection[0])
            if index < len(self.filtered):
                return self.filtered[index]
        return self.filtered[0] if self.filtered else None

    def draw_chart(self):
        self.chart.delete("all")
        row = self.selected_row()
        if not row:
            self.chart.create_text(12, 21, anchor="w", text="没有匹配记录 · 无数据不等于没有互动", fill="#667085")
            return
        self.chart.create_text(12, 11, anchor="w", text=f"{row['a']} ↔ {row['b']}   ·   {EVIDENCE[row['evidence']]}   ·   共同圈层 / 时间共现不计作直接回应", fill="#475467")
        self.chart.create_text(12, 32, anchor="w", text="发言为选区内去重条数；群聊不代表彼此回应。私聊回应为会话内换人。单向程度按定向轮次计算。", fill="#667085")

    def show_details(self):
        row = self.selected_row()
        if not row:
            return
        win = tk.Toplevel(self.root)
        win.title('互动证据 · v2')
        win.geometry('740x580')
        body = ttk.Frame(win)
        body.pack(fill='both', expand=True)
        view = tk.Text(body, wrap='word', font=('Microsoft YaHei UI', 10), padx=16, pady=14)
        scrollbar=ttk.Scrollbar(body,command=view.yview)
        scrollbar.pack(side='right',fill='y')
        view.pack(side='left',fill='both', expand=True)
        view.configure(yscrollcommand=scrollbar.set)
        unknown = row['evidence'] == 'coverage_unknown'
        def count(key):
            number = row.get(key, 0)
            return ('已观测 '+str(number) if number else '未知') if unknown else str(number)
        def value(key, unit=''):
            if row.get(key) is None:
                return '未知 / 无已观测样本' if unknown else '未定义 / 无有效样本'
            prefix = '仅已观测样本 ' if unknown and key != 'coactive_bins' else ''
            return prefix+str(round(row[key], 4))+unit
        obs = self.result.get('observation', {})
        cov = self.result.get('coverage', {})
        source = self.result.get('source', {})
        text = (f"{row['a']} ↔ {row['b']} · {EVIDENCE[row['evidence']]}\n\n"
                f"回应轮次 {count('response_rounds')}；定向轮次 {count('directed_rounds')}（A {count('rounds_a')} / B {count('rounds_b')}）\n"
                f"双向会话 {count('bidirectional_sessions')} / 会话 {count('sessions')}；消息 A {count('messages_a')} / B {count('messages_b')}\n"
                f"去重发言 A {row.get('source_messages_a', 0)} / B {row.get('source_messages_b', 0)}（群聊为全选区，私聊仅该对象）\n"
                f"共同活跃日 {count('common_active_days')}；单向程度 {value('one_way_fraction')}（|A轮次−B轮次| / 总轮次）\n"
                f"明确引用链接 {count('reply_links')}；@ 轮次 {count('mention_rounds')}（@ 不等于已回应）\n"
                f"方向性回应 A {count('response_rounds_a')} / B {count('response_rounds_b')}；引用链接 A {count('reply_links_a')} / B {count('reply_links_b')}\n"
                f"@ 轮次 A {count('mention_rounds_a')} / B {count('mention_rounds_b')}；不算已回应\n"
                f"延迟类别：{'群聊明确引用' if self.result.get('mode')=='group' else '私聊换人候选'}\n"
                f"响应间隔中位数 {value('median_response_seconds', ' 秒')}；有效样本 {count('latency_samples')}\n"
                f"文本词面重合 {value('lexical_overlap_mean')}；有效样本 {count('lexical_samples')}（非语义/情感判定）\n"
                f"低置信共现时间格 {value('coactive_bins')}；共同定向邻居 {value('common_neighbors')}（均不增加回应数）\n"
                f"会话内同人重复文本 {count('repeated_messages')}；转发标记 {count('forwarded_messages')}（原始条数保留）\n\n"
                f"请求日期：{obs.get('requested_start') or '未指定'} 至 {obs.get('requested_end') or '未指定'}（结束日包含）\n"
                f"实际覆盖：{obs.get('actual_start') or '无'} 至 {obs.get('actual_end') or '无'}，北京时间\n"
                f"会话间隔阈值 {obs.get('session_gap_seconds', 21600)} 秒；共现格 {obs.get('coactive_bin_seconds', 300)} 秒\n"
                f"原始选区记录 {source.get('selected_rows', 0)}；排除 {source.get('excluded_rows', 0)}；重复 ID {cov.get('duplicate_rows', 0)}；冲突 ID {cov.get('conflicting_ids', 0)}\n"
                f"未知结构 {cov.get('unknown_rows', 0)}；未解析引用 {cov.get('unresolved_references', 0)}；同刻邻接 {cov.get('simultaneous_rows', 0)}\n"
                f"原生正文已解析 {source.get('native_text_rows', 0)}；引用缓存标记 {source.get('native_quote_rows', 0)}（目标未知）\n"
                f"无效时间 {source.get('invalid_time', 0)}；无效发送方向 {source.get('invalid_sender', 0)}\n\n"
                "群聊只认已解析的结构引用；普通 @文字不能证明目标。原生 NT 仅有限正文/引用缓存标记可读，引用目标与 @ 尚未知。\n"
                "私聊换人仅是观察到的回应候选，不代表语义回应。跨会话或同刻记录不计算回复延迟。\n"
                "样本不足阈值：少于 3 次回应或少于 2 个双向会话；这是描述提示，不是统计置信保证。\n"
                "无亲密度总分、夜间奖励或自动权重。真实数据测量效度尚未验证。")
        text += '\n\n规则证据样例（每对最多 20 条；R 为选区输入行编号，非原生 ID；不显示正文）：\n'
        text += '结构不完整时：未知不等于 0；正数仅为已观测计数，不是下限。补全事件可能合并轮次/会话；比例和延迟也仅针对已观测样本。\n'
        for example in row.get('evidence_examples', []):
            text += f"消息 {example['message_id'] or '无稳定ID'} → 来源 {example['source_message_id'] or '无'}；{example['sender']} → {example['target']}；{example['kind']}\n"
        assistance = self.result.get('assistance', {})
        text += "\n\n本地 AI 辅助：" + assistance.get('status','disabled') + "（可关闭；只提供话题候选）\n"
        text += "向量模型不具备已验证的问答、安慰、争论、讽刺或情感分类能力。余弦不是置信概率。\n"
        candidates = [p for p in assistance.get('predictions',[]) if p.get('source_sender') in (row['a'],row['b']) and p.get('candidate_target') in (row['a'],row['b'])]
        for candidate in candidates[:30]:
            text += (f"候选消息 {candidate['message_id']} → 来源消息 {candidate['source_message_id']}；"
                     f"{candidate['source_sender']} → {candidate['candidate_target']}；余弦 {candidate['cosine']:.3f}；"
                     f"依据：前 5 分钟同话题，未确认回应（输入行 {candidate['row']}）\n")
        text += f"本对子候选共 {len(candidates)} 条，此处最多显示 30 条；CLI 可导出全部有界证据及弃权原因。"
        view.insert('1.0',text)
        view.configure(state='disabled')
        ttk.Button(win, text='关闭', command=win.destroy).pack(pady=8)
        win.bind('<Escape>', lambda _:win.destroy())
        view.focus_set()

    def close(self):
        self.closing = True
        if self.search_timer:
            self.after_cancel(self.search_timer)
            self.search_timer = None
        self.cancel()

def main():
    import os
    os.chdir(Path(__file__).resolve().parent)
    mp.freeze_support()
    # Ensure spawned workers also use pythonw, even from a debug console.
    pythonw = Path(__import__("sys").executable).with_name("pythonw.exe")
    if pythonw.exists():
        mp.set_executable(str(pythonw))
    log_dir = Path("output") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "desktop.log", maxBytes=100000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log = logging.getLogger("desktop")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False
    root = tk.Tk()
    root.report_callback_exception = lambda *_: log.error("ui_callback_failed")
    Desktop(root)
    root.mainloop()

if __name__ == "__main__":
    main()
