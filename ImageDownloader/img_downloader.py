# ImageDownloader - Muhammed Shokr its amazing


def ImageDownloader(url):
    import os, re, requests

    response = requests.get(url)
    text = response.text

    p = r'<img.*?src="(.*?)"[^\>]+>'
    img_addrs = re.findall(p, text)

    for i in img_addrs:
        os.system("wget {}".format(i))

    return "DONE"


# USAGE
print("Hey!! Welcome to the Image downloader...")
link=input("Please enter the url from where you want to download the image..")
# now you can give the input at run time and get download the images.
# https://www.123rf.com/stock-photo/spring_color.html?oriSearch=spring&ch=spring&sti=oazo8ueuz074cdpc48
ImageDownloader(link)
