def get_filename(filepath : str):
    file = filepath.split("/")[-1]
    return file.split(".")[0]

def get_extension(filepath : str):
    return filepath.split(".")[-1]

def replace_extension(path, new_extension):
    return ".".join(path.split(".")[:-1]) + new_extension

def keyify(*args):
    if len(args)==1:
        key = [x for x in args[0]]
    else :
        key = [x for x in args]
    key.sort()
    return tuple(key)

def replace_in_list(l, x, y):
    return [y if e==x else e for e in l]

class Logger:
    def __init__(self, name = "Logger", verbose=True):
        self.name = name
        self.verbose = verbose

    def log(self, *messages):
        if self.verbose:
            print(*((f"[{self.name}]",) + messages))