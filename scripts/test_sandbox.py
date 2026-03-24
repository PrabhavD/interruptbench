import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra import DockerSandbox

sandbox = DockerSandbox(image="interruptbench:base", cpus=1.0, memory="512m", timeout_sec=10)

print("=== basic execution ===")
res = sandbox.run_python("""
print('sandbox works')
""")
print(res.stdout.strip())
print("ok=", res.ok, "rc=", res.returncode)

print("\n=== network blocked check ===")
res = sandbox.run_python("""
import socket
s = socket.socket()
try:
    s.connect(('example.com', 80))
    print('network-open')
except Exception as e:
    print(type(e).__name__)
""")
print(res.stdout.strip())

print("\n=== timeout check ===")
slow = DockerSandbox(image="interruptbench:base", timeout_sec=1)
res = slow.run_python("""
import time
time.sleep(3)
print('finished')
""")
print("timed_out=", res.timed_out, "rc=", res.returncode)