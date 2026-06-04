import sys
import platform

print("=== Python Environment Check ===")
print(f"Python version : {sys.version}")
print(f"Executable     : {sys.executable}")
print(f"Platform       : {platform.platform()}")
print(f"Machine        : {platform.machine()}")
print(f"Processor      : {platform.processor()}")