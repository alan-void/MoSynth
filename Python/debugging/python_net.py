import importlib
import sys
import time

_DEBUGGER_RETRY_SECONDS = 1.0


def _debugger_is_healthy():
    pydevd = sys.modules.get("pydevd")
    if pydevd is None:
        return False

    debugger = pydevd.get_global_debugger()
    return debugger is not None and not debugger.pydb_disposed


def _reset_debugger():
    pydevd_pycharm = sys.modules.get("pydevd_pycharm")
    if pydevd_pycharm is not None:
        # stoptrace() leaves the stdout/stderr redirectors installed. Restore
        # them before unloading pydevd so the new session owns the redirectors.
        try:
            pydevd_pycharm.stoptrace()
        except Exception:
            pass

    for stream_name, buffer_name in (("stdout", "_pydevd_out_buffer_"),
                                     ("stderr", "_pydevd_err_buffer_")):
        original_name = f"{stream_name}_original"
        if hasattr(sys, original_name):
            setattr(sys, stream_name, getattr(sys, original_name))
            delattr(sys, original_name)
        if hasattr(sys, buffer_name):
            delattr(sys, buffer_name)

    # pydevd stores the socket connection in module globals, so a new server
    # needs a fresh debugger module graph.
    for name in tuple(sys.modules):
        if name.startswith("pydevd") or name.startswith("_pydev"):
            del sys.modules[name]
    importlib.invalidate_caches()


def connect_debugger():
    pycharm_egg = "C:\\Program Files\\JetBrains\\PyCharm 2026.1.2\\debug-eggs\\pydevd-pycharm.egg"
    if pycharm_egg not in sys.path:
        sys.path.insert(0, pycharm_egg)

    # PyCharm clears pydevd's global debugger when its debug-server socket
    # closes. Leave a healthy connection alone, including during Unity reloads.
    if _debugger_is_healthy():
        return True

    now = time.monotonic()
    retry_at = getattr(sys, "_mosynth_debugger_retry_at", 0.0)
    if now < retry_at:
        return False

    try:
        if "pydevd" in sys.modules:
            _reset_debugger()

        pydevd_pycharm = importlib.import_module("pydevd_pycharm")
        pydevd_pycharm.settrace('localhost', port=9229, stdout_to_server=True,
                                stderr_to_server=True, suspend=False)
    except Exception:
        # A debug server is normally absent outside a debugging session. Retry
        # later without flooding Unity's console with connection-refused errors.
        sys._mosynth_debugger_retry_at = now + _DEBUGGER_RETRY_SECONDS
        return False

    sys._mosynth_debugger_retry_at = 0.0
    return True
