import re

from .text_utils import normalize_text


DANGEROUS_WORDS = {
    "install",
    "uninstall",
    "delete",
    "remove",
    "erase",
    "format",
    "move",
    "rename",
    "edit",
    "modify",
    "write",
    "download",
    "update",
    "upgrade",
    "registry",
    "script",
    "powershell",
    "terminal",
    "shell",
}

BLOCKED_APPS = {
    "powershell",
    "terminal",
    "windows terminal",
    "regedit",
    "registry editor",
    "control panel",
    "task manager",
    "services",
    "computer management",
    "disk management",
    "device manager",
    "administrative tools",
    "windows tools",
    "windows powershell",
    "system tools",
}


def is_dangerous_request(text):
    words = set(re.findall(r"[a-z0-9]+", normalize_text(text)))
    return bool(words & DANGEROUS_WORDS)

