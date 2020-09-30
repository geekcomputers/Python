# ImageDownloader - Muhammed Shokr

def ImageDownloader(url):
    import os, re, requests

    response = requests.get(url)
    text = response.text

    p = r'<img.*?src="(.*?)"[^\>]+>'
    img_addrs = re.findall(p, text)

    for i in img_addrs:
        os.system("wget {}".format(i))
    
    return 'DONE'

# USAGE
ImageDownloader("https://www.123rf.com/stock-photo/spring_color.html?oriSearch=spring&ch=spring&sti=oazo8ueuz074cdpc48|")
