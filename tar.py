import gzip
import os
import rarfile
import tarfile
import zipfile

from library.utils.file import get_filetype
from library.utils.path import make_dir


def uncompress(src_file, dest_dir):
    result = get_filetype(src_file)
    if not result[0]:
        return (False, result[1], '')
    filefmt = result[1]

    result = make_dir(dest_dir)
    if not result:
        return (False, '创建解压目录失败', filefmt)

    if filefmt in ('tgz', 'tar'):
        try:
            tar = tarfile.open(src_file)
            names = tar.getnames()
            for name in names:
                tar.extract(name, dest_dir)
            tar.close()
        except Exception as e:
            return (False, e, filefmt)
    elif filefmt == 'zip':
        try:
            zip_file = zipfile.ZipFile(src_file)
            for names in zip_file.namelist():
                zip_file.extract(names, dest_dir)
            zip_file.close()
        except Exception as e:
            return (False, e, filefmt)
    elif filefmt == 'rar':
        try:
            rar = rarfile.RarFile(src_file)
            os.chdir(dest_dir)
            rar.extractall()
            rar.close()
        except Exception as e:
            return (False, e, filefmt)
    elif filefmt == 'gz':
        try:

            f_name = dest_dir + '/' + os.path.basename(src_file)
            # 获取文件的名称，去掉  
            g_file = gzip.GzipFile(src_file)
            # 创建gzip对象  
            open(f_name, "w+").write(g_file.read())
            # gzip对象用read()打开后，写入open()建立的文件中。  
            g_file.close()
            # 关闭gzip对象  

            result = get_filetype(src_file)
            if not result[0]:
                new_filefmt = '未知'
            else:
                new_filefmt = result[1]
            return (True, '解压后的文件格式为：' + new_filefmt, filefmt)
        except Exception as e:
            return (False, e, filefmt)
    else:
        return (False, '文件格式不支持或者不是压缩文件', filefmt)

    return (True, '', filefmt)
