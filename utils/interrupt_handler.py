import signal
import sys
import threading
import time


class GracefulInterruptHandler:
    def __init__(self):
        self.interrupted = False
        self.released = False
        self._lock = threading.Lock()
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        self.interrupt_received = threading.Event()

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.interrupt_received.clear()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        with self._lock:
            if self.released:
                return False
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            self.released = True
            return True

    def _signal_handler(self, signum, frame):
        with self._lock:
            if self.interrupted:
                if signum == signal.SIGINT and self.original_sigint:
                    self.release()
                    self.original_sigint(signum, frame)
                return

            print("\n\n[⚠️ Interrupt] Caught interrupt signal...")
            self.interrupted = True
            self.interrupt_received.set()  # Signal the interrupt

    def check(self):
        return self.interrupt_received.is_set()

    @property
    def should_stop(self):
        return self.interrupted
