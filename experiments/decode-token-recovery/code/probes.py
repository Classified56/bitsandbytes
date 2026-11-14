"""Controlled measurements on experiment-owned storage only."""
import ctypes as ct
import mmap
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from prefill_flush_reload.native import Native
from prefill_flush_reload.measurement import observe
from flush_reload_test import choose_threshold


def exact(s,n):
    b=bytearray()
    while len(b)<n:
        x=s.recv(n-len(b))
        if not x: raise EOFError('observer exited')
        b.extend(x)
    return bytes(b)

class Engine:
    def __init__(self, cross=False, timer='paper', capacity=64*4864*4):
        self.native=Native()
        self.mapping=mmap.mmap(-1,capacity)
        self.bytes=np.frombuffer(self.mapping,dtype=np.uint8)
        self.bytes[:]=1
        self.address=self.bytes.ctypes.data
        self.out64=np.zeros(capacity//64,dtype=np.uint64)
        self.out32=np.zeros(capacity//64,dtype=np.uint32)
        self.timer=timer
        self.cross=cross
        self.legacy=None
        if timer=='legacy':
            lib=ROOT/'code/legacy_native.so'
            if not lib.exists():
                subprocess.run(['cc','-O3','-shared','-fPIC',str(ROOT/'code/legacy_native.c'),'-o',str(lib)],check=True)
            self.legacy=ct.CDLL(str(lib))
            self.legacy.probe_lines.argtypes=[ct.c_void_p]*3+[ct.c_size_t]
        if cross:
            self.sock,child=socket.socketpair()
            self.pid=os.fork()
            if self.pid==0:
                self.sock.close()
                try:
                    os.sched_setaffinity(0,[4])
                    child.sendall(b'R')
                    while True:
                        address,lines=struct.unpack('<QQ',exact(child,16))
                        if not lines: break
                        self.native.flush_lines(address,lines)
                        child.sendall(b'F')
                        n=struct.unpack('<Q',exact(child,8))[0]
                        order=np.frombuffer(exact(child,n*8),dtype=np.uint64).copy()
                        result=self._probe(order)
                        child.sendall(result[order].tobytes())
                    child.close()
                    os._exit(0)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
            child.close()
            assert exact(self.sock,1)==b'R'
    def _probe(self,order):
        out=self.out64
        if self.timer=='paper':
            self.native.probe_lines(self.address,order.ctypes.data,out.ctypes.data,len(order))
        else:
            o=order.astype(np.uint32);t=self.out32
            self.legacy.probe_lines(self.address,o.ctypes.data,t.ctypes.data,len(o))
            out[:]=t
        return out
    def flush_lines(self,address,n):
        if self.cross:
            self.sock.sendall(struct.pack('<QQ',address,n))
            assert exact(self.sock,1)==b'F'
        else: self.native.flush_lines(address,n)
    def probe_lines(self,data,order,times,n):
        if not self.cross and self.timer=='paper':
            self.native.probe_lines(self.address,order,times,n)
            return
        o=np.ctypeslib.as_array(order)[:n].astype(np.uint64)
        if self.cross:
            self.sock.sendall(struct.pack('<Q',n)+o.tobytes())
            t=np.frombuffer(exact(self.sock,8*n),dtype=np.uint64)
        else: t=self._probe(o)[o]
        for i,v in zip(o,t): times[int(i)]=int(v)
    def sample(self,order,lines,action=None):
        order=np.asarray(order,dtype=np.uint64)
        t=(ct.c_uint64*(len(self.bytes)//64))()
        o=(ct.c_size_t*len(order))(*order)
        self.flush_lines(self.address,lines)
        if action: action()
        self.probe_lines(None,o,t,len(order))
        return np.array([t[int(i)] for i in order],dtype=np.uint64)
    def calibrate(self,seed=917,samples=2000):
        rng=np.random.default_rng(seed)
        labels=rng.permutation(np.tile([0,1],samples//2))
        ticks=np.array([self.sample([0],1,lambda:self.native.fr_touch(self.address))[0]
                        if hit else self.sample([0],1)[0] for hit in labels])
        cut=choose_threshold(list(zip(labels[:samples//2],ticks[:samples//2])))
        return int(cut),labels,ticks
    def close(self):
        if self.cross:
            self.sock.sendall(struct.pack('<QQ',0,0));self.sock.close()
            _,status=os.waitpid(self.pid,0)
            assert status==0,status
        # External numpy/torch views can outlive this wrapper until process exit.
        self.native.close()


def collect(engine,rows,width,itemsize,gather,method,repeats,seed,probe_rows=1):
    """Call the UNMODIFIED upstream observation scheduler.

    width is expressed in 16-bit storage units solely for its line arithmetic.
    The native probe always reads one byte per physical 64-byte cache line.
    """
    sim=types.SimpleNamespace(native=engine,tokens=rows,width=width*itemsize//2,
        activations=types.SimpleNamespace(address=engine.address,data=engine.address),gather=gather)
    trace=[]
    timings=observe(sim,True,repeats,seed,probe_stride=5 if method=='stride5' else 1,
        probe_rows=probe_rows,probe_mode='single' if method=='single' else 'sweep',
        flush_scope='observed',trace=trace)
    n=width*itemsize//64
    ticks=np.array([[t[r*n+l] for r in range(probe_rows) for l in range(n)] for t in timings],dtype=np.uint32)
    order=np.zeros_like(ticks)
    for trial,line,sweep,index,prev in trace: order[trial,line]=index
    return ticks.reshape(repeats,probe_rows,n),order.reshape(repeats,probe_rows,n)
