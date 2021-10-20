import os
import shutil

ext = {
    "web": "css less scss wasm ",
    "audio": "aac aiff ape au flac gsm it m3u m4a mid mod mp3 mpa pls ra s3m sid wav wma xm ",
    "code": "c cc class clj cpp cs cxx el go h java lua m m4 php pl po py rb rs swift vb vcxproj xcodeproj xml diff patch html js ",
    "slide": "ppt odp ",
    "sheet": "ods xls xlsx csv ics vcf ",
    "image": "3dm 3ds max bmp dds gif jpg jpeg png psd xcf tga thm tif tiff ai eps ps svg dwg dxf gpx kml kmz webp ",
    "archiv": "7z a apk ar bz2 cab cpio deb dmg egg gz iso jar lha mar pea rar rpm s7z shar tar tbz2 tgz tlz war whl xpi zip zipx xz pak ",
    "book": "mobi epub azw1 azw3 azw4 azw6 azw cbr cbz ",
    "text": "doc docx ebook log md msg odt org pages pdf rtf rst tex txt wpd wps ",
    "exec": "exe msi bin command sh bat crx ",
    "font": "eot otf ttf woff woff2 ",
    "video": "3g2 3gp aaf asf avchd avi drc flv m2v m4p m4v mkv mng mov mp2 mp4 mpe mpeg mpg mpv mxf nsv ogg ogv ogm qt rm rmvb roq srt svi vob webm wmv yuv ",
}

for key, value in ext.items():
    value = value.split()
    ext[key] = value


def add_to_dir(ex, src_path, path):
    file_with_ex = os.path.basename(src_path)
    file_without_ex = file_with_ex[: file_with_ex.find(ex) - 1]
    for cat, extensions in ext.items():
        if ex in extensions:
            os.chdir(path)
            dest_path = path + "\\" + cat
            if cat in os.listdir():
                try:
                    shutil.move(src_path, dest_path)
                except shutil.Error:
                    renamed_file = rename(file_without_ex, ex, dest_path)
                    os.chdir(path)
                    os.rename(file_with_ex, renamed_file)
                    os.chdir(dest_path)
                    shutil.move(path + "\\" + renamed_file, dest_path)
            else:
                os.mkdir(cat)

                try:
                    shutil.move(src_path, dest_path)
                except Exception as e:
                    print(e)
                if os.path.exists(src_path):
                    os.unlink(src_path)


def rename(search, ex, dest_path):
    count = 0
    os.chdir(dest_path)
    for filename in os.listdir():
        if filename.find(search, 0, len(search) - 1):
            count = count + 1

    return search + str(count) + "." + ex
