def crc_check(data, div):
    l = len(div)
    ct = 0
    data = [int(i) for i in data]
    div = [int(i) for i in div]
    zero = [0 for i in range(l)]
    temp_data = [data[i] for i in range(l)]
    result = []
    for j in range(len(data) - len(div) + 1):
        print("Temp_dividend", temp_data)
        msb = temp_data[0]
        if msb == 0:
            result.append(0)
            for i in range(l - 1, -1, -1):
                temp_data[i] = temp_data[i] ^ zero[i]
        else:
            result.append(1)
            for i in range(l - 1, -1, -1):
                temp_data[i] = temp_data[i] ^ div[i]
        temp_data.pop(0)
        if l + j < len(data):
            temp_data.append(data[l + j])
    crc = temp_data
    print("Quotient: ", result, "remainder", crc)
    return crc


# returning crc value


while 1 > 0:
    print("Enter data: ")
    data = input()  # can use it like int(input())
    print("Enter divisor")
    div = input()  # can use it like int(input())
    original_data = data
    data = data + ("0" * (len(div) - 1))
    crc = crc_check(data, div)
    crc_str = ""
    for c in crc:
        crc_str += c
    print("Sent data: ", original_data + crc_str)
    sent_data = original_data + crc_str
    print(
        "If again applying CRC algorithm, the remainder/CRC must be zero if errorless."
    )
    crc = crc_check(sent_data, div)
    remainder = crc
    print("Receiver side remainder: ", remainder)
    print("Continue [Y/N]:")
    ch = input()
    if ch == "N" or ch == "n":
        break
    else:
        continue
