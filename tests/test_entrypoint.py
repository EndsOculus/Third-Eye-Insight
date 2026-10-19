import importlib


def test_train_main_calls_run_analysis(monkeypatch):
    train = importlib.import_module("train")
    called = []

    def fake_run_analysis():
        called.append(True)

    monkeypatch.setattr(train, "run_analysis", fake_run_analysis)
    train.main()

    assert called == [True]
