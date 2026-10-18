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
    assert all(0 <= r[2] <= 1 for r in result["pairs"])
    with pytest.raises(InputError):
        analyze(Request(demo=True, start="2027-01-01"))
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
        app.rows = [(f"成员 {i:03d}", f"成员 {j:03d}", .5, .4, .5, .6) for i in range(300) for j in range(i+1,300)]
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
