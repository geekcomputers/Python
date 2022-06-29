"""
Following script deals with system varibales 
"""
import os

# this will get all path varibales


def get_all_env():
    for key in os.environ:
        print(key, "-->", os.environ[key])

# this will check for a particular variable


def var_check(var):
    if os.environ[var]:
        return True
    else:
        return False

# it set a path variable


def set_path(path):
    os.environ["PATH"] += ";" + path


if __name__ == "__main__":

    get_all_env()
    var_check("HOMEDRIVE")
    set_path("abc")
    var_check("PATH")
