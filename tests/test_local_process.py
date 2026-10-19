import os
import sys
import time
import ctypes
import pytest
from local_process import run_contained,WindowsJob,ProcessFailure


def test_contained_stdin_and_hidden_child():
    result=run_contained([sys.executable,'-c','import sys;print(sys.stdin.readline().strip())'],'synthetic',os.environ.copy())
    assert result.strip()=='synthetic'


@pytest.mark.parametrize('mode',['timeout','cancel','overflow'])
def test_owned_process_tree_is_reaped(mode):
    pids=[];start=time.monotonic()
    code='import time;time.sleep(60)' if mode!='overflow' else 'print("x"*40000)'
    with pytest.raises(ProcessFailure):
        run_contained([sys.executable,'-c',code],'synthetic',os.environ.copy(),timeout=.2 if mode=='timeout' else 5,
                      cancelled=lambda:mode=='cancel' and time.monotonic()-start>.2,on_started=pids.append)
    assert pids
    api=ctypes.WinDLL('kernel32',use_last_error=True);api.OpenProcess.restype=ctypes.c_void_p
    handle=api.OpenProcess(0x1000,False,pids[0])
    if handle:
        status=ctypes.c_ulong();api.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        api.GetExitCodeProcess(handle,ctypes.byref(status))
        api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(handle)
        assert status.value!=259

@pytest.mark.parametrize('mode',['timeout','cancel'])
def test_unread_pipe_does_not_block_cancellation(mode):
    start=time.monotonic()
    with pytest.raises(ProcessFailure,match=mode if mode=='timeout' else 'cancelled'):
        run_contained([sys.executable,'-c','import time;time.sleep(60)'],'x'*10000,
                      os.environ.copy(),timeout=.2 if mode=='timeout' else 5,
                      cancelled=lambda:mode=='cancel' and time.monotonic()-start>.2)
    assert time.monotonic()-start<3


def test_descendants_are_reaped(tmp_path):
    marker=tmp_path/'child.pid'
    # Publish only a completed PID: cancellation may kill the writer as soon as marker exists.
    code='import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);marker=Path(sys.argv[1]);pending=marker.with_suffix(".pending");pending.write_text("");time.sleep(.05);pending.write_text(str(p.pid));pending.replace(marker);time.sleep(60)'
    with pytest.raises(ProcessFailure,match='cancelled'):
        run_contained([sys.executable,'-c',code,str(marker)],'synthetic',os.environ.copy(),timeout=10,
                      cancelled=lambda:marker.exists())
    pid=int(marker.read_text())
    api=ctypes.WinDLL('kernel32',use_last_error=True);api.OpenProcess.restype=ctypes.c_void_p
    handle=api.OpenProcess(0x1000,False,pid)
    if handle:
        status=ctypes.c_ulong();api.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        api.GetExitCodeProcess(handle,ctypes.byref(status))
        api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(handle)
        assert status.value!=259
