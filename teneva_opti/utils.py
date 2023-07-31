import os


class Log:
    def __init__(self, fpath=None, with_log=True, with_log_info=True):
        self.fpath = path(fpath, '.txt')
        self.with_log = with_log
        self.with_log_info = with_log_info
        self.is_file_new = True

    def __call__(self, text):
        if self.with_log:
            print(text)
            if self.fpath:
                with open(self.fpath, 'w' if self.is_file_new else 'a') as f:
                    f.write(text + '\n')
                self.is_file_new = False

    def err(self, content=''):
        raise ValueError(content)

    def info(self, content=''):
        if self.with_log_info:
            self(content)

    def prc(self, content=''):
        self(f'\n.... {content}')

    def res(self, content=''):
        self(f'DONE {content}')

    def wrn(self, content=''):
        self(f'WRN ! {content}')


def path(fpath=None, ext=None):
    if not fpath:
        return

    fold = os.path.dirname(fpath)
    if fold:
        os.makedirs(fold, exist_ok=True)

    if ext and not fpath.endswith('.' + ext):
        fpath += '.' + ext

    return fpath
