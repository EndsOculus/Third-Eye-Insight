import time
import tkinter as tk
import pytest
from desktop import Desktop, PAGE_SIZE
from desktop_core import Request, InputError, analyze

def pump(root, condition, timeout=60):
    end = time.monotonic()+timeout
    while not condition() and time.monotonic() < end:
        root.update()
        time.sleep(0.01)
    assert condition()

def test_validation(tmp_path):
    with pytest.raises(InputError):
        Request().validate()
    path = tmp_path / "中文 space.db"
    path.touch()
    Request(db=str(path), identifier="12345").validate()
    for kwargs in ({"identifier":"invalid"}, {"start":"bad"}, {"start":"2026-02-02", "end":"2026-01-01"}, {"focus":"bad"}):
        values = dict(db=str(path), identifier="12345")
        values.update(kwargs)
        with pytest.raises(InputError):
            Request(**values).validate()


def test_reader_mode_is_visible_and_resets_for_normal_selection():
    import queue
    root = tk.Tk()
    root.withdraw()
    app = Desktop(root)
    app.after_cancel(app.poll_id)
    class Events(queue.Queue):
        def close(self):
            pass
    class Process:
        def is_alive(self):
            return False
        def join(self, **_):
            pass
        def close(self):
            pass
    try:
        payload = analyze(Request(demo=True))
        payload['demo'] = False
        for isolated in (True, False):
            payload['source']['reader_isolated'] = isolated
            payload['source']['source_sha256'] = '0'*64
            app.events = Events()
            app.events.put(('result', payload))
            app.process = Process()
            app.poll()
            assert ('隔离只读' if isolated else '标准只读') in app.context.get()
            app.show_details()
            def walk(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from walk(child)
            assert any('0'*64 in w.get('1.0','end') for w in walk(root) if isinstance(w,tk.Text))
            for window in app.detail_windows:
                window.destroy()
            app.detail_windows.clear()
            app.after_cancel(app.poll_id)
    finally:
        root.destroy()

def test_demo_metrics_and_empty():
    result = analyze(Request(demo=True))
    assert result["rows"] == 1200
    assert len(result["pairs"]) == 66
    assert all(r["response_rounds"] >= 0 and "score" not in r for r in result["pairs"])
    assert analyze(Request(demo=True, start="2027-01-01"))["pairs"] == []
    private = analyze(Request(demo=True, mode="c2c"))
    assert len(private["pairs"]) == 11


def _details_text(payload):
    root = tk.Tk()
    root.withdraw()
    app = Desktop(root)
    try:
        app.result = payload
        app.filtered = payload['pairs']
        assert app.filtered
        app.show_details()
        def walk(widget):
            for child in widget.winfo_children():
                yield child
                yield from walk(child)
        return '\n'.join(w.get('1.0', 'end') for w in walk(root) if isinstance(w, tk.Text))
    finally:
        root.destroy()


def test_standard_database_details_do_not_mislabel_null_hash_as_demo(tmp_path):
    from tests.interaction_fixtures import msg, make_db
    path = tmp_path / 'ordinary.sqlite'
    make_db(path, [msg(1 + i % 2, i * 30, str(i)) for i in range(12)], source_format='nt')
    payload = analyze(Request(db=str(path), identifier='99'))
    assert not payload['demo']
    assert payload['source']['source_sha256'] is None
    assert payload['source']['source_hash_status'] == 'not_computed_standard'
    text = _details_text(payload)
    assert '来源 SHA-256：未计算（标准只读）' in text
    assert '合成演示' not in text


@pytest.mark.parametrize(('demo', 'expected'), [
    (True, '合成演示（无来源文件）'),
    (False, '未提供（来源信息缺失）'),
])
def test_details_distinguish_demo_from_missing_provenance(demo, expected):
    payload = analyze(Request(demo=True))
    payload['demo'] = demo
    payload['source'] = {}
    assert '来源 SHA-256：' + expected in _details_text(payload)

def test_real_gui_lifecycle_and_pagination():
    root = tk.Tk()
    root.withdraw()
    app = Desktop(root)
    try:
        app.start()
        assert app.process is None
        app.start(demo=True)
        pid = app.process.pid
        app.start(demo=True)
        assert app.process.pid == pid
        app.cancel()
        app.cancel()
        pump(root, lambda: app.process is None)
        assert "已取消" in app.status.get()
        app.start(demo=True)
        pump(root, lambda: app.process is None)
        assert len(app.rows) == 66, app.status.get()
        # Maximum 300-member pair count, without inserting all rows into Tk.
        app.rows = [dict(app.rows[0], a=f"成员 {i:03d}", b=f"成员 {j:03d}") for i in range(300) for j in range(i+1,300)]
        app.filter_rows()
        assert len(app.filtered) == 44850
        assert len(app.table.get_children()) == PAGE_SIZE
        app.change_page(1)
        assert app.page == 1
        app.search.set("成员 299")
        pump(root, lambda: len(app.filtered) == 299)
        app.search.set("no match")
        pump(root, lambda: len(app.filtered) == 0)
        assert not app.table.get_children()
        app.start(demo=True)
        app.close()
        pump(root, lambda: app.process is None)
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass

@pytest.mark.parametrize("terminal", ["result", "error"])
def test_terminal_event_arriving_between_drain_and_exit(terminal):
    import queue
    root = tk.Tk()
    root.withdraw()
    app = Desktop(root)
    app.after_cancel(app.poll_id)
    class Events(queue.Queue):
        def close(self):
            pass
    events = Events()
    row = analyze(Request(demo=True))["pairs"][0]
    payload = {"pairs": [row], "rows": 2, "users": 2, "seconds": .1, "demo": True} if terminal == "result" else "synthetic terminal error"
    class Process:
        def is_alive(self):
            # Deterministically model the worker publishing after an empty drain.
            events.put((terminal, payload))
            return False
        def join(self, **kwargs):
            pass
        def close(self):
            pass
    app.events, app.process = events, Process()
    try:
        app.poll()
        assert app.process is None
        assert app.finished
        assert not app.progress.winfo_manager()
        if terminal == "result":
            assert app.rows == [row]
        else:
            assert app.status.get() == payload
    finally:
        app.after_cancel(app.poll_id)
        root.destroy()

@pytest.mark.parametrize("size", ["1120x780", "900x700"])
def test_layout_keeps_table_and_pagination_inside_window(size):
    root = tk.Tk()
    app = Desktop(root)
    root.geometry(size)
    root.update()
    try:
        assert app.table.winfo_height() >= 120
        for widget in (app.table, app.prev_button, app.next_button):
            assert widget.winfo_rooty() >= root.winfo_rooty()
            assert widget.winfo_rooty()+widget.winfo_height() <= root.winfo_rooty()+root.winfo_height()
            assert widget.winfo_rootx()+widget.winfo_width() <= root.winfo_rootx()+root.winfo_width()
        assert app.values["exclude"].get() == ""
    finally:
        app.after_cancel(app.poll_id)
        root.destroy()


def test_unknown_directional_cells_are_not_plain_zero():
    row=next(r for r in analyze(Request(demo=True))['pairs'] if r['evidence']=='coverage_unknown')
    values=Desktop.row_values(row)
    assert values[2]==values[3]==values[5]==values[6]=='未知'
    positive=dict(row,response_rounds=3,bidirectional_sessions=1,common_active_days=1)
    values=Desktop.row_values(positive)
    assert values[2]=='已观测 3' and values[3]==values[5]=='已观测 1'


def test_unknown_details_state_limits_and_direction_categories():
    root=tk.Tk();root.withdraw();app=Desktop(root)
    try:
        app.result=analyze(Request(demo=True))
        app.rows=[r for r in app.result['pairs'] if r['evidence']=='coverage_unknown']
        app.filter_rows();app.show_details();root.update()
        def walk(widget):
            for child in widget.winfo_children():
                yield child
                yield from walk(child)
        view=next(w for w in walk(root) if isinstance(w,tk.Text))
        text=view.get('1.0','end')
        assert '双向会话 未知' in text and '共同活跃日 未知' in text
        assert '方向性回应 A 未知 / B 未知' in text
        assert '延迟类别：群聊明确引用' in text
        assert '正数仅为已观测计数，不是下限' in text
        assert '≥' not in text
    finally:
        app.after_cancel(app.poll_id);root.destroy()


def test_generation_gui_missing_config_and_job_cancel():
    root=tk.Tk();root.withdraw();app=Desktop(root)
    try:
        app.assist_mode.set('\u751f\u6210\u5b9e\u9a8c')
        app.generation_config='missing-synthetic-local-config.json'
        app.start(demo=True)
        pump(root,lambda:app.process is None)
        assert app.result['assistance']['status']=='model_unavailable'
        assert app.worker_job is None
        app.start(demo=True);pid=app.process.pid
        app.start(demo=True);assert app.process.pid==pid
        app.cancel();app.cancel()
        pump(root,lambda:app.process is None)
        assert app.worker_job is None
    finally:
        app.close()
        try:root.destroy()
        except tk.TclError:pass


def test_prefixed_database_gui_and_parent_cleanup(tmp_path):
    from pathlib import Path
    from tests.interaction_fixtures import make_db,msg
    path=tmp_path/'\u4e2d\u6587 space.db'
    make_db(path,[msg(1,0,'a'),msg(2,10,'b')],source_format='nt')
    path.write_bytes(b'N'*1024+path.read_bytes())
    root=tk.Tk();root.withdraw();app=Desktop(root)
    try:
        app.values['db'].set(str(path));app.values['identifier'].set('99')
        app.start();scratch=Path(app.scratch.name)
        pump(root,lambda:app.process is None)
        assert app.result['source']['database_format']=='nt_prefix_sqlite'
        assert app.result['rows']==2 and not scratch.exists()
        app.start();scratch=Path(app.scratch.name)
        # An interrupted worker may leave a partial derived file. Parent owns cleanup.
        (scratch/'partial.db').write_bytes(b'synthetic')
        app.cancel();app.cancel();pump(root,lambda:app.process is None)
        assert not scratch.exists()
        app.start();scratch=Path(app.scratch.name);app.close()
        pump(root,lambda:app.process is None)
        assert not scratch.exists()
    finally:
        try:root.destroy()
        except tk.TclError:pass


def test_cleanup_failure_recovers_ui_and_retries():
    import queue
    root=tk.Tk();root.withdraw();app=Desktop(root);app.after_cancel(app.poll_id)
    class Temp:
        attempts=0
        def cleanup(self):
            self.attempts+=1
            if self.attempts==1:raise PermissionError('private-path-must-not-appear')
    class Process:
        def is_alive(self):return False
        def join(self,**kwargs):pass
        def close(self):pass
    class Events(queue.Queue):
        def close(self):pass
    temporary=Temp();app.scratch=temporary;app.process=Process();app.events=Events()
    try:
        app.poll()
        assert app.process is None and app.pending_scratch==[temporary]
        assert str(app.run_button['state'])=='normal'
        assert 'private-path' not in app.status.get()
        assert app.cleanup_scratch() and not app.pending_scratch
    finally:
        app.after_cancel(app.poll_id);root.destroy()


def test_private_identity_search_sort_details_and_privacy(tmp_path):
    import sqlite3
    from tests.interaction_fixtures import make_db,msg
    path=tmp_path/'private identity.db'
    make_db(path,[msg(900,0,'a',peer='2'),msg(2,10,'b',peer='2'),msg(900,20,'c',peer='10'),msg(10,30,'d',peer='10')],source_format='nt',mode='c2c')
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE contacts(qq TEXT,remark TEXT,nickname TEXT)')
        db.execute("INSERT INTO contacts VALUES ('2','Synthetic Friend','Nick')")
    root=tk.Tk();root.withdraw();app=Desktop(root)
    try:
        app.mode.set('c2c');app.values['db'].set(str(path));app.values['identifier'].set('900');app.start()
        pump(root,lambda:app.process is None)
        assert 'b' not in tuple(app.table['displaycolumns'])
        shown=[app.table.item(i,'values')[0] for i in app.table.get_children()]
        assert 'Synthetic Friend\uff082\uff09' in shown and '10' in shown
        assert not any('900' in value for value in shown)
        assert '\u6635\u79f0\u4e0d\u53ef\u7528' in app.status.get()
        app.search.set('Synthetic Friend');pump(root,lambda:len(app.filtered)==1)
        app.show_details();root.update()
        def walk(w):
            for c in w.winfo_children():yield c;yield from walk(c)
        texts=[w.get('1.0','end') for w in walk(root) if isinstance(w,tk.Text)]
        assert any('Synthetic Friend' in text for text in texts)
        app.sort_rows('a');assert len(app.filtered)==1
        app.privacy.set(True);app.identity_changed();root.update()
        assert not app.detail_windows
        assert all('Synthetic Friend' not in str(app.table.item(i,'values')) for i in app.table.get_children())
        assert all(w.cget('show')=='*' for w in app.identity_inputs)
        app.search.set('Synthetic Friend');pump(root,lambda:len(app.filtered)==0)
        app.privacy.set(False);app.identity_changed()
        assert len(app.filtered)==2
        app.result['mode']='group';app.filter_rows()
        assert 'b' in tuple(app.table['displaycolumns'])
    finally:
        app.after_cancel(app.poll_id);root.destroy()


def test_streaming_gui_cancel_and_close_large_source(tmp_path):
    import hashlib,sqlite3
    from pathlib import Path
    from tests.interaction_fixtures import make_db
    path=tmp_path/'large streaming.db';make_db(path,[],source_format='nt',mode='c2c')
    with sqlite3.connect(path) as db:
        db.executemany('INSERT INTO c2c_msg_table VALUES (?,?,?,?,?,?)',((str(i),str(1+i%2),'2',99,1767225600+i,'synthetic') for i in range(100001)))
    before=hashlib.sha256(path.read_bytes()).digest()
    root=tk.Tk();root.withdraw();app=Desktop(root)
    try:
        app.mode.set('c2c');app.values['db'].set(str(path));app.values['identifier'].set('1')
        app.start();scratch=Path(app.scratch.name);pid=app.process.pid
        app.start();assert app.process.pid==pid
        pump(root,lambda:float(app.progress['value'])>=28)
        app.cancel();pump(root,lambda:app.process is None)
        assert not scratch.exists()
        app.start();scratch=Path(app.scratch.name)
        pump(root,lambda:float(app.progress['value'])>=28)
        app.close();pump(root,lambda:app.process is None)
        assert not scratch.exists() and hashlib.sha256(path.read_bytes()).digest()==before
    finally:
        try:root.destroy()
        except tk.TclError:pass
