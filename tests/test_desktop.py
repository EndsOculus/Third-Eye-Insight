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

def test_demo_metrics_and_empty():
    result = analyze(Request(demo=True))
    assert result["rows"] == 1200
    assert len(result["pairs"]) == 66
    assert all(r["response_rounds"] >= 0 and "score" not in r for r in result["pairs"])
    assert analyze(Request(demo=True, start="2027-01-01"))["pairs"] == []
    private = analyze(Request(demo=True, mode="c2c"))
    assert len(private["pairs"]) == 11

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
