"""Windows-owned process jobs: close/timeout/cancel kills descendants, not user apps."""
import ctypes
from ctypes import wintypes
import os
import subprocess
import threading
import time


class ProcessFailure(RuntimeError):
    pass


class _BasicLimits(ctypes.Structure):
    _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),
              ('min_working',ctypes.c_size_t),('max_working',ctypes.c_size_t),('active',wintypes.DWORD),
              ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]


class _Limits(ctypes.Structure):
    _fields_=[('basic',_BasicLimits),('io',ctypes.c_uint64*6),('process_memory',ctypes.c_size_t),
              ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]


class WindowsJob:
    def __init__(self):
        if os.name!='nt':raise ProcessFailure('Local generation requires Windows process containment')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        for name,args,result in [
            ('CreateJobObjectW',[ctypes.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),
            ('SetInformationJobObject',[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD],wintypes.BOOL),
            ('AssignProcessToJobObject',[wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),
            ('OpenProcess',[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
            ('CloseHandle',[wintypes.HANDLE],wintypes.BOOL),
        ]:
            func=getattr(self.api,name);func.argtypes=args;func.restype=result
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:raise ProcessFailure('Unable to contain local process')
        limits=_Limits();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close();raise ProcessFailure('Unable to contain local process')

    def assign(self,pid):
        process=self.api.OpenProcess(0x0101,False,pid)  # SET_QUOTA | TERMINATE
        try:
            if not process or not self.api.AssignProcessToJobObject(self.handle,process):
                raise ProcessFailure('Unable to contain local process')
        finally:
            if process:self.api.CloseHandle(process)

    def close(self):
        if getattr(self,'handle',None):
            self.api.CloseHandle(self.handle);self.handle=None


def run_contained(args, prompt, env, timeout=45, cancelled=lambda:False, on_started=None):
    """No shell, prompt file or command-line body; bounded pipes, hidden child job.

    Desktop workers also belong to an outer job before their start gate opens,
    so terminating the worker cannot orphan a child during creation/assignment.
    """
    payload=(prompt+'\n').encode('utf-8')
    if len(payload)>16384:raise ProcessFailure('input_limit')
    started=time.monotonic()
    job=WindowsJob();process=None;threads=[];output=bytearray();overflow=threading.Event()
    write_failed=threading.Event()
    def write_input():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (OSError,ValueError):write_failed.set()
    def drain(stream,keep,limit):
        count=0
        try:
            while True:
                chunk=stream.read(4096)
                if not chunk:break
                count+=len(chunk)
                if count>limit:overflow.set();break
                if keep:output.extend(chunk)
        finally:stream.close()
    try:
        process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 env=env,creationflags=0x00000004|0x08000000)  # suspended + no window
        job.assign(process.pid)
        resume=ctypes.WinDLL('ntdll').NtResumeProcess
        resume.argtypes=[wintypes.HANDLE];resume.restype=ctypes.c_long
        if resume(int(process._handle))!=0:raise ProcessFailure('Unable to start local process')
        for stream,keep,limit in [(process.stdout,True,32768),(process.stderr,False,262144)]:
            thread=threading.Thread(target=drain,args=(stream,keep,limit),daemon=True);thread.start();threads.append(thread)
        if on_started:on_started(process.pid)
        writer=threading.Thread(target=write_input,daemon=True);writer.start();threads.append(writer)
        while process.poll() is None:
            if cancelled():raise ProcessFailure('cancelled')
            if overflow.is_set():raise ProcessFailure('output_limit')
            if time.monotonic()-started>timeout:raise ProcessFailure('timeout')
            time.sleep(.03)
        # A CLI may leave helper descendants holding pipes; closing its job also
        # cleans those up before joining readers.
        job.close()
        for thread in threads:thread.join(timeout=2)
        if any(t.is_alive() for t in threads) or overflow.is_set():raise ProcessFailure('output_limit')
        if write_failed.is_set() or process.returncode!=0:raise ProcessFailure('runtime_failed')
        return output.decode('utf-8',errors='strict')
    except (OSError,UnicodeError,ValueError) as exc:
        raise ProcessFailure('runtime_failed') from None
    finally:
        job.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for thread in threads:thread.join(timeout=2)
        if process is not None and process.stdin and not process.stdin.closed:
            process.stdin.close()
