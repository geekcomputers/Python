import platform  # built in lib

print(f"System : {platform.system()}")  # Prints type of Operating System
print(f"System name : {platform.node()}")  # Prints System Name
print(f"version : {platform.release()}")  # Prints System Version
# TO get the detailed version number
print(
    f"detailed version number : {platform.version()}"
)  # Prints detailed version number
print(
    f"System architecture : {platform.machine()}"
)  # Prints whether the system is 32-bit ot 64-bit
print(f"System processor : {platform.processor()}")  # Prints CPU model
