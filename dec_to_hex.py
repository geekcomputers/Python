mydict={
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10':'A','11':'B',
    '12':'C','13':'D','14':'E','15':'F'
        }
dec_num=input('Enter the decimal number\n');
dec_num=int(dec_num)
value=""
while dec_num>0:
    value+=mydict[str(dec_num%16)]
    dec_num=dec_num//16
hex_value=value[::-1]
print(hex_value)
