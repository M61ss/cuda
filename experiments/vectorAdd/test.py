#!/bin/python3

import platform, re, subprocess, os

def get_processor():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
            
print(f"CPU: {get_processor()} - {os.cpu_count()} cores")
print("")
print("########### CPU ###########")
print("")
print(subprocess.Popen(["./build/vectorAddCPU", ""], stdout=subprocess.PIPE).communicate()[0].decode())
print("########### GPU ###########")
print("")
print(subprocess.Popen(["./build/vectorAddGPU", ""], stdout=subprocess.PIPE).communicate()[0].decode())
