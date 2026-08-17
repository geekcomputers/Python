from plyer import notification  # pip install plyer
import psutil  # pip install psutil
from datetime import datetime

def TestFeature_Memory():
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024**3):.2f} GB")
    print(f"Memory Usage Percent: {memory.percent}%")

def TestFeature_CPU():
    # Get system-wide CPU usage over 1 second
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage Percent: {cpu_percent}%")

    # Get per-core CPU usage over 1 second
    for i, percentage in enumerate(psutil.cpu_percent(interval=1, percpu=True)):
        print(f"Core {i}: {percentage}%")

def TestFeature_Battery():
    # psutil.sensors_battery() will return the information related to battery
    battery = psutil.sensors_battery()
    if battery is None:
        print("Battery not found")
        return

    # battery percent will return the current battery prcentage
    percent = battery.percent
    charging = battery.power_plugged

    # Notification(title, description, duration)--to send
    # notification to desktop
    # help(Notification)
    if charging:
        if percent == 100:
            charging_message = "Unplug your Charger"
        else:
            charging_message = "Charging"
    else:
        charging_message = "Not Charging"
    message = str(percent) + "% Charged\n" + charging_message

    notification.notify("Battery Information", message, timeout=10)

def TestFeature_Network():
    net_io = psutil.net_io_counters()
    print(f"Bytes Sent: {net_io.bytes_sent}")
    print(f"Bytes Received: {net_io.bytes_recv}")

def TestFeature_BootTime():
    boot_time_timestamp = psutil.boot_time()
    print(f"System booted at: {datetime.fromtimestamp(boot_time_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")

def TestFeature_Processes():
    for proc in psutil.process_iter(['pid', 'name']):
        print(f"PID: {proc.info['pid']}, Name: {proc.info['name']}")


if __name__ == "__main__":
    TestFeature_CPU()
    TestFeature_Memory()
    TestFeature_Battery()
    TestFeature_Network()
    TestFeature_BootTime()
    TestFeature_Processes()