"""Responsive Tk desktop, with an isolated cancellable analysis process."""
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk, filedialog

from desktop_core import Request

PAGE_SIZE = 100

class Desktop(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=24)
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
        self.filtered = []
        self.page = 0
        self.search_timer = None
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#f3f5fa", foreground="#202a44")
        style.configure("TEntry", fieldbackground="white", padding=7)
        style.configure("TButton", padding=(14, 8))
        style.configure("Accent.TButton", background="#5557d9", foreground="white")
        style.map("Accent.TButton", background=[("active", "#4345b5"), ("disabled", "#b6b9ce")])
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 23, "bold"))
        style.configure("Sub.TLabel", foreground="#64708a")
        style.configure("Treeview", rowheight=30, fieldbackground="white", background="white")
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"), padding=8)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        ttk.Label(self, text="恋之瞳  /  Third Eye Insight", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self, text="本地分析 · 只读数据 · 匿名关系视图", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 18))
        form = ttk.LabelFrame(self, text="  01  数据与筛选  ", padding=14)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        self.values = {k: tk.StringVar() for k in ("db", "identifier", "start", "end", "focus", "exclude")}
        ttk.Label(form, text="数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.values["db"]).grid(row=0, column=1, columnspan=3, sticky="ew", padx=10)
        ttk.Button(form, text="选择文件", command=self.browse).grid(row=0, column=4)
        self.mode = tk.StringVar(value="group")
        modes = ttk.Frame(form)
        modes.grid(row=1, column=0, columnspan=5, sticky="w", pady=10)
        ttk.Radiobutton(modes, text="群聊", variable=self.mode, value="group").pack(side="left")
        ttk.Radiobutton(modes, text="私聊（填我的 QQ）", variable=self.mode, value="c2c").pack(side="left", padx=20)
        for index, (key, title) in enumerate((("identifier", "群号 / 我的 QQ"), ("focus", "聚焦账号（可选）"), ("start", "开始 YYYY-MM-DD"), ("end", "结束 YYYY-MM-DD"), ("exclude", "排除账号（逗号分隔）"))):
            r, c = 2 + index // 2, (index % 2) * 2
            ttk.Label(form, text=title).grid(row=r, column=c, sticky="w", pady=4)
            ttk.Entry(form, textvariable=self.values[key]).grid(row=r, column=c+1, sticky="ew", padx=10, pady=4)
        self.flags = {k: tk.BooleanVar() for k in ("encrypted", "gpu", "auto_weight")}
        options = ttk.Frame(form)
        options.grid(row=5, column=0, columnspan=5, sticky="w", pady=(8, 0))
        for key, label in (("encrypted", "SQLCipher（使用本地 config.json）"), ("gpu", "GPU"), ("auto_weight", "自动权重")):
            ttk.Checkbutton(options, text=label, variable=self.flags[key]).pack(side="left", padx=(0, 16))
        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, sticky="ew", pady=12)
        self.run_button = ttk.Button(controls, text="开始分析", style="Accent.TButton", command=self.start)
        self.run_button.pack(side="left")
        self.demo_button = ttk.Button(controls, text="体验合成演示", command=lambda: self.start(demo=True))
        self.demo_button.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(controls, text="取消", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left")
        self.progress = ttk.Progressbar(controls, maximum=100, length=180)
        self.progress.pack(side="right", padx=8)
        self.status = tk.StringVar(value="准备就绪。先体验演示，或选择只读数据库。")
        ttk.Label(self, textvariable=self.status, wraplength=1000).grid(row=4, column=0, sticky="ew", pady=(0, 10))
        panel = ttk.LabelFrame(self, text="  02  关系洞察  ", padding=12)
        panel.grid(row=5, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        self.summary = tk.StringVar(value="暂无结果 · 分析后显示消息数、成员数与阶段耗时")
        ttk.Label(panel, textvariable=self.summary, style="Sub.TLabel").grid(row=0, column=0, sticky="w")
        self.chart = tk.Canvas(panel, height=100, background="white", highlightthickness=0)
        self.chart.grid(row=1, column=0, sticky="ew", pady=10)
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        table_frame = ttk.Frame(panel)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("a", "b", "score", "semantic", "behavior", "network")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        for col, label in zip(columns, ("成员 A", "成员 B", "综合得分 ↓", "语义 / 上下文", "行为", "网络 / 时间")):
            self.table.heading(col, text=label)
            self.table.column(col, width=130, minwidth=85, anchor="center")
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        footer = ttk.Frame(panel)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(footer, text="查找匿名成员").pack(side="left")
        self.search = tk.StringVar()
        self.search.trace_add("write", self.schedule_filter)
        ttk.Entry(footer, textvariable=self.search, width=18).pack(side="left", padx=8)
        self.next_button = ttk.Button(footer, text="下一页", command=lambda: self.change_page(1))
        self.next_button.pack(side="right")
        self.page_text = tk.StringVar(value="0 / 0")
        ttk.Label(footer, textvariable=self.page_text).pack(side="right", padx=12)
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
        request = Request(demo=True) if demo else Request(mode=self.mode.get(), **{k:v.get().strip() for k,v in self.values.items()}, **{k:v.get() for k,v in self.flags.items()})
        try:
            request.validate()
        except ValueError as exc:
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
        self.rows = []
        self.filter_rows()
        self.summary.set("分析进行中…")
        self.status.set("准备后台分析…")
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

    def poll(self):
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
                    self.rows = payload["pairs"]
                    self.filter_rows()
                    self.summary.set(f"{payload['rows']:,} 条消息  ·  {payload['users']} 名成员  ·  {len(self.rows):,} 对关系  ·  后台总耗时 {payload['seconds']:.2f} 秒")
                    self.status.set("合成演示完成（随机向量，仅用于体验界面）。" if payload["demo"] else "分析完成。得分仅为互动指标，不代表真实情感或关系判断。")
                    self.progress["value"] = 100
                    logging.getLogger("desktop").info("analysis_completed")
                elif kind == "error":
                    self.finished = True
                    self.status.set(payload)
                    self.summary.set("没有生成结果 · 可调整输入后重试")
                    logging.getLogger("desktop").info("analysis_failed")
        if self.process is not None and not self.process.is_alive():
            self.process.join(timeout=0)
            self.process.close()
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
        self.filtered = [r for r in self.rows if not needle or needle in (r[0]+" "+r[1]).casefold()]
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
        for row in self.filtered[self.page*PAGE_SIZE:(self.page+1)*PAGE_SIZE]:
            self.table.insert("", "end", values=(*row[:2], *(f"{n:.3f}" for n in row[2:])))
        pages = (len(self.filtered)+PAGE_SIZE-1)//PAGE_SIZE
        self.page_text.set(f"{self.page+1 if pages else 0} / {pages}")
        self.prev_button.configure(state="normal" if self.page else "disabled")
        self.next_button.configure(state="normal" if self.page+1 < pages else "disabled")

    def draw_chart(self):
        self.chart.delete("all")
        top = self.filtered[:3]
        if not top:
            self.chart.create_text(20, 45, anchor="w", text="暂无匹配关系 · 结果将在这里呈现", fill="#64708a")
            return
        maximum = max(max(r[2] for r in top), 0.001)
        width = max(100, self.chart.winfo_width()-240)
        for i, row in enumerate(top):
            y = 18 + 30*i
            self.chart.create_text(12, y, anchor="w", text=f"{row[0]} — {row[1]}", fill="#202a44")
            self.chart.create_rectangle(190, y-7, 190 + width*row[2]/maximum, y+7, fill="#6667df", outline="")
            self.chart.create_text(200+width*row[2]/maximum, y, anchor="w", text=f"{row[2]:.2f}", fill="#64708a")

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
