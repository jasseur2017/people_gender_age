from pathlib import Path
import ctypes


ctypes.CDLL(
    Path(Path(__file__).resolve().parent, "DCNv2", "build", "dcn_plugin.so"),
    mode=ctypes.RTLD_GLOBAL
)
