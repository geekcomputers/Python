from typing import List

def crc_check(data: str, div: str) -> List[int]:
    """
    Perform CRC (Cyclic Redundancy Check) calculation.
    
    Args:
        data: Input data string (binary digits)
        div: Divisor string (binary digits, typically a polynomial)
    
    Returns:
        List of integers representing the CRC remainder
    """
    divisor_length: int = len(div)  # Renamed from 'l' to 'divisor_length'
    data_list: List[int] = [int(i) for i in data]
    div_list: List[int] = [int(i) for i in div]
    zero: List[int] = [0] * divisor_length
    temp_data: List[int] = data_list[:divisor_length]
    result: List[int] = []
    
    for j in range(len(data_list) - divisor_length + 1):
        print("Temp_dividend", temp_data)
        msb: int = temp_data[0]
        
        if msb == 0:
            result.append(0)
            temp_data = [temp_data[i] ^ zero[i] for i in range(divisor_length-1, -1, -1)]
        else:
            result.append(1)
            temp_data = [temp_data[i] ^ div_list[i] for i in range(divisor_length-1, -1, -1)]
        
        temp_data.pop(0)
        if divisor_length + j < len(data_list):
            temp_data.append(data_list[divisor_length + j])
    
    crc: List[int] = temp_data
    print("Quotient: ", result, "remainder", crc)
    return crc

def validate_binary_input(input_str: str) -> bool:
    """Check if input string contains only 0s and 1s"""
    return all(c in {'0', '1'} for c in input_str)

def main() -> None:
    """Main program loop for CRC calculation and verification"""
    while True:
        try:
            # Get input from user
            data: str = input("Enter data: ").strip()
            if not validate_binary_input(data):
                raise ValueError("Data must be a binary string (0s and 1s)")
            
            div: str = input("Enter divisor: ").strip()
            if not validate_binary_input(div):
                raise ValueError("Divisor must be a binary string (0s and 1s)")
            if len(div) < 2:
                raise ValueError("Divisor length must be at least 2")
            
            original_data: str = data
            padded_data: str = data + "0" * (len(div) - 1)
            
            # Calculate CRC
            crc: List[int] = crc_check(padded_data, div)
            crc_str: str = ''.join(str(c) for c in crc)
            
            # Display sent data
            sent_data: str = original_data + crc_str
            print("Sent data: ", sent_data)
            
            # Verify CRC at receiver side
            print("If again applying CRC algorithm, the remainder/CRC must be zero if errorless.")
            receiver_crc: List[int] = crc_check(sent_data, div)
            print("Receiver side remainder: ", receiver_crc)
            
            # Check if remainder is zero
            if all(bit == 0 for bit in receiver_crc):
                print("CRC verification successful - no detected errors.")
            else:
                print("CRC verification failed - errors detected.")
                
        except ValueError as ve:
            print(f"Input Error: {ve}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue
        
        # Ask user to continue
        ch: str = input("Continue [Y/N]: ").strip().upper()
        if ch == "N":
            break

if __name__ == "__main__":
    main()