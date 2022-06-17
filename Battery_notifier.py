from plyer import notification  # pip install plyer
import psutil  # pip install psutil

# psutil.sensors_battery() will return the information related to battery
battery = psutil.sensors_battery()

# battery percent will return the current battery prcentage
percent = battery.percent
charging = battery.power_plugged

# Notification(title, description, duration)--to send
# notification to desktop
# help(Notification)
if charging:
    charging_message = "Unplug your Charger" if percent == 100 else "Charging"
else:
    charging_message = "Not Charging"
message = str(percent) + "% Charged\n" + charging_message

notification.notify("Battery Information", message, timeout=10)
