import os
import json
import os
import shutil

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [e.strip("\n") for e in f.readlines()]
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                return [e.strip("\n") for e in f.readlines()]
        except Exception as e:
            return None


def save_lines(filepath, data):
    with open(filepath, "w") as f:
        f.write("\n".join(data))


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def deletedir(p):
    if os.path.exists(p):
        shutil.rmtree(p)

def fileExist(p):
    if os.path.exists(p):
        return True
    else:
        return False
