# Website Blocker without timer, funtion call oriented

import platform

os_name = platform.system()

sites = ["youtube.com", "www.youtube.com",
         "facebook.com", "www.facebook.com", "reddit.com"]  # add the sites you want to block over here


hostfile_path = "C:/Windows/System32/drivers/etc/hosts" if os_name == "Windows" else "/etc/hosts"

local = "127.0.0.1"


def block():
    print("Blocked sites")
    with open(hostfile_path, 'r+') as hostfile:
        file_content = hostfile.read()
        for site in sites:
            if site not in file_content:
                hostfile.write(local+" "+site+"\n")


def unBlock():
    print("sites unblocked")
    with open(hostfile_path, 'r+') as hostfile:
        lines = hostfile.readline()
        hostfile.seek(0)
        for line in lines:
            if not any(site in line for site in sites):
                hostfile.write(line)
        hostfile.truncate()


if __name__ == "__main__":

    block()  # To block
    unBlock()  # To Unblock
