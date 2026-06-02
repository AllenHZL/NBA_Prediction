import sys
from contextlib import contextmanager
from datetime import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(data.encode(encoding, errors="replace").decode(encoding))
            stream.flush()

    def flush(self):
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()


@contextmanager
def tee_output_to_markdown(log_file):
    log_file.parent.mkdir(exist_ok=True)
    log_handle = log_file.open("w", encoding="utf-8")
    log_handle.write("# NBA Prediction CLI Output\n\n")
    log_handle.write(f"- Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_handle.write("```text\n")
    log_handle.flush()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    try:
        yield
    finally:
        if not log_handle.closed:
            log_handle.write("\n```\n\n")
            log_handle.write(f"- Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if not log_handle.closed:
            log_handle.close()

