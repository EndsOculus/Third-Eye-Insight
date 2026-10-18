import os
import sys
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visualization import filter_for_gbk, plot_custom_heatmap


def test_filter_for_gbk_ascii():
    result = filter_for_gbk("hello")
    assert result == "hello"
    assert isinstance(result, str)


def test_filter_for_gbk_chinese():
    result = filter_for_gbk("你好")
    assert result == "你好"
    assert isinstance(result, str)


def test_filter_for_gbk_returns_str():
    result = filter_for_gbk("test")
    assert isinstance(result, str)


def test_plot_custom_heatmap_no_error():
    matrix = np.zeros((3, 3))
    labels = ["A", "B", "C"]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        save_path = f.name
    try:
        plot_custom_heatmap(matrix, labels, "test", save_path)
        assert os.path.exists(save_path)
    finally:
        plt.close("all")
        if os.path.exists(save_path):
            os.remove(save_path)
