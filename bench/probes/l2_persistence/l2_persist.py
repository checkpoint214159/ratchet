"""ctypes shim for CUDA's L2 persisting-access window (cc >= 8.0).

Two entry points from libcudart, both already loaded in any torch process:

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value)

Reference: NVIDIA CUDA C Programming Guide, "Device Memory L2 Access Management".
"""
from __future__ import annotations

import ctypes
import ctypes.util

# cudaLimit
cudaLimitPersistingL2CacheSize = 0x06
# cudaStreamAttrID (== cudaLaunchAttributeAccessPolicyWindow in CUDA 12)
cudaStreamAttributeAccessPolicyWindow = 1
# cudaDeviceAttr
cudaDevAttrMaxPersistingL2CacheSize = 108
cudaDevAttrMaxAccessPolicyWindowSize = 109
# cudaAccessProperty
cudaAccessPropertyNormal, cudaAccessPropertyStreaming, cudaAccessPropertyPersisting = 0, 1, 2


class cudaAccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class cudaStreamAttrValue(ctypes.Union):
    _fields_ = [
        ("accessPolicyWindow", cudaAccessPolicyWindow),
        ("syncPolicy", ctypes.c_int),
    ]


def _load():
    name = ctypes.util.find_library("cudart") or "libcudart.so.12"
    return ctypes.CDLL(name)


_rt = _load()
_rt.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
_rt.cudaDeviceSetLimit.restype = ctypes.c_int
_rt.cudaDeviceGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
_rt.cudaDeviceGetLimit.restype = ctypes.c_int
_rt.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_rt.cudaDeviceGetAttribute.restype = ctypes.c_int
_rt.cudaStreamSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_rt.cudaStreamSetAttribute.restype = ctypes.c_int
_rt.cudaStreamGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_rt.cudaStreamGetAttribute.restype = ctypes.c_int
_rt.cudaGetErrorString.argtypes = [ctypes.c_int]
_rt.cudaGetErrorString.restype = ctypes.c_char_p


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} -> cudaError {rc}: {_rt.cudaGetErrorString(rc).decode()}")


def device_attr(attr: int, device: int = 0) -> int:
    v = ctypes.c_int()
    _check(_rt.cudaDeviceGetAttribute(ctypes.byref(v), attr, device),
           f"cudaDeviceGetAttribute({attr})")
    return v.value


def max_persisting_l2_bytes(device: int = 0) -> int:
    return device_attr(cudaDevAttrMaxPersistingL2CacheSize, device)


def max_window_bytes(device: int = 0) -> int:
    return device_attr(cudaDevAttrMaxAccessPolicyWindowSize, device)


def set_persisting_set_aside(nbytes: int) -> int:
    """Reserve `nbytes` of L2 for persisting accesses. Returns what the driver accepted."""
    _check(_rt.cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, nbytes),
           "cudaDeviceSetLimit(PersistingL2CacheSize)")
    got = ctypes.c_size_t()
    _check(_rt.cudaDeviceGetLimit(ctypes.byref(got), cudaLimitPersistingL2CacheSize),
           "cudaDeviceGetLimit(PersistingL2CacheSize)")
    return got.value


def set_window(stream_ptr: int, base_ptr: int, num_bytes: int,
               hit_ratio: float = 1.0,
               hit_prop: int = cudaAccessPropertyPersisting,
               miss_prop: int = cudaAccessPropertyStreaming) -> None:
    val = cudaStreamAttrValue()
    val.accessPolicyWindow.base_ptr = ctypes.c_void_p(base_ptr)
    val.accessPolicyWindow.num_bytes = num_bytes
    val.accessPolicyWindow.hitRatio = hit_ratio
    val.accessPolicyWindow.hitProp = hit_prop
    val.accessPolicyWindow.missProp = miss_prop
    _check(_rt.cudaStreamSetAttribute(ctypes.c_void_p(stream_ptr),
                                      cudaStreamAttributeAccessPolicyWindow,
                                      ctypes.byref(val)),
           "cudaStreamSetAttribute(AccessPolicyWindow)")


def get_window(stream_ptr: int) -> dict:
    val = cudaStreamAttrValue()
    _check(_rt.cudaStreamGetAttribute(ctypes.c_void_p(stream_ptr),
                                      cudaStreamAttributeAccessPolicyWindow,
                                      ctypes.byref(val)),
           "cudaStreamGetAttribute(AccessPolicyWindow)")
    w = val.accessPolicyWindow
    return dict(base_ptr=w.base_ptr, num_bytes=w.num_bytes, hitRatio=w.hitRatio,
                hitProp=w.hitProp, missProp=w.missProp)


def clear_window(stream_ptr: int) -> None:
    set_window(stream_ptr, 0, 0, 0.0, cudaAccessPropertyNormal, cudaAccessPropertyNormal)
