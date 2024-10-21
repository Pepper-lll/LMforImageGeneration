import numpy as np
import shutil
# from .common import qd_tqdm as tqdm
import mmap
import time
# from .common import dict_update_path_value, dict_get_path_value, get_all_path, load_from_yaml_str
import logging
from azfuse import File
import os
import os.path as op
import io
import base64
from PIL import Image

def pilimg_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = Image.open(io.BytesIO(jpgbytestring))
    image = image.convert('RGB')
    return image

def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != b'' and s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return b''.join(result)

class TSVFile(object):
    def __init__(self, tsv_file, cache_policy=None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self.lineidx_8b = self.lineidx + '.8b'
        self._fp = None
        self._mfp = None
        self._lineidx = None
        self.fp8b = None
        self.cache_policy= cache_policy
        self.close_fp_after_read = False
        if os.environ.get('QD_TSV_CLOSE_FP_AFTER_READ'):
            self.close_fp_after_read = bool(os.environ['QD_TSV_CLOSE_FP_AFTER_READ'])
        self.use_mmap = False
        if os.environ.get('QD_TSV_MMAP'):
            self.use_mmap = int(os.environ['QD_TSV_MMAP'])
        #self.has_lineidx_8b = int(os.environ.get('QD_USE_LINEIDX_8B', '0'))
        self.has_lineidx_8b = True
        # the process always keeps the process which opens the
        # file. If the pid is not equal to the currrent pid, we will re-open
        # teh file.
        self.pid = None
        self.lineidx_8b_pid = None
        self.open_once = False

        self._len = None
        self._tsv_file_size = None

    @property
    def tsv_file_size(self):
        if self._tsv_file_size is None:
            self._tsv_file_size = File.get_file_size(self.tsv_file)
        return self._tsv_file_size

    def get_row_len(self, i):
        start = self.get_offset(i)
        if i < len(self) - 1:
            end = self.get_offset(i + 1)
        else:
            end = self.tsv_file_size
        return end - start

    def get_row_offsets(self, i):
        start = self.get_offset(i)
        if i < len(self) - 1:
            end = self.get_offset(i + 1)
        else:
            end = self.tsv_file_size
        return start, end

    def close_fp(self):
        if self._fp:
            self._fp.close()
            self._fp = None
        if self._mfp:
            self._mfp.close()
            self._mfp = None
        if self.has_lineidx_8b and self.fp8b:
            self.fp8b.close()
            self.fp8b = None

    def release(self):
        self.close_fp()
        self._lineidx = None

    def close(self):
        #@deprecated('use release to make it more clear not to release lineidx')
        self.close_fp()

    def __del__(self):
        self.release()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        self._ensure_tsv_opened()
        self.fp_seek(0)
        if not self.use_mmap:
            for line in self._fp:
                result = [s.strip() for s in line.decode().split('\t')]
                yield result
        else:
            while True:
                line = self._mfp.readline()
                if line == b'':
                    break
                result = [s.strip() for s in line.decode().split('\t')]
                yield result

    def num_rows(self):
        if self._len is None:
            if self.has_lineidx_8b:
                self._len = File.get_file_size(self.lineidx_8b) // 8
            else:
                self._ensure_lineidx_loaded()
                self._len = len(self._lineidx)
        return self._len

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def get_current_column(self):
        if self.use_mmap:
            result = [s.strip() for s in self._mfp.readline().decode().split('\t')]
        else:
            result = [s.strip() for s in self._fp.readline().split('\t')]
        return result

    def get_current_column2(self, size):
        if self.use_mmap:
            result = [s.strip() for s in self._mfp.read(size).decode().split('\t')]
        else:
            result = [s.strip() for s in self._fp.read(size).decode().split('\t')]
        return result

    def fp_seek(self, pos):
        if self.use_mmap:
            self._mfp.seek(pos)
        else:
            self._fp.seek(pos)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos, end = self.get_row_offsets(idx)
        self.fp_seek(pos)
        result = self.get_current_column2(end - pos)
        if self.close_fp_after_read:
            self.close_fp()
        return result

    def seek3(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)
        self.fp_seek(pos)
        result = self.get_current_column()
        if self.close_fp_after_read:
            self.close_fp()
        return result

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        pos = self.get_offset(idx)
        self._fp.seek(pos)
        return read_to_character(self._fp, b'\t').decode()

    def seek_first_columns(self):
        assert self.has_lineidx_8b
        self._ensure_tsv_opened()
        self.ensure_lineidx_8b_opened()
        result = []
        for idx in range(len(self)):
            self.fp8b.seek(idx * 8)
            pos = int.from_bytes(self.fp8b.read(8), 'little')
            self._fp.seek(pos)
            result.append(read_to_character(self._fp, b'\t').decode())
        return result

    def open(self, fname, mode):
        return File.open(fname, mode)

    def ensure_lineidx_8b_opened(self):
        if self.fp8b is None:
            self.fp8b = self.open(self.lineidx_8b, 'rb')
            self.lineidx_8b_pid = os.getpid()
        if self.lineidx_8b_pid != os.getpid():
            self.fp8b.close()
            logging.info('re-open {} because the process id changed'.format(
                self.lineidx_8b))
            self.fp8b= self.open(self.lineidx_8b, 'rb')
            self.lineidx_8b_pid = os.getpid()

    def get_offset(self, idx):
        # do not use op.isfile() to check whether lineidx_8b exists as it may
        # incur API call for blobfuse, which will be super slow if we enumerate
        # a bunch of data
        if self.has_lineidx_8b:
            self.ensure_lineidx_8b_opened()
            self.fp8b.seek(idx * 8)
            ret = int.from_bytes(self.fp8b.read(8), 'little')
            return ret
        else:
            self._ensure_lineidx_loaded()
            pos = self._lineidx[idx]
            return pos

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            with File.open(self.lineidx, 'r') as fp:
                self._lineidx = tuple([int(i.strip()) for i in fp.readlines()])
            logging.info('loaded {} from {}'.format(
                len(self._lineidx),
                self.lineidx
            ))

    def get_tsv_fp(self):
        start = time.time()
        fp = File.open(self.tsv_file, 'rb')
        if self.use_mmap:
            mfp = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mfp = fp
        end = time.time()
        if (end - start) > 10:
            logging.info('too long ({}) to open {}'.format(
                end - start,
                self.tsv_file))
        return mfp, fp

    def _ensure_tsv_opened(self):
        if self.cache_policy == 'memory':
            assert self._fp is not None
            return

        if self._fp is None:
            self._mfp, self._fp = self.get_tsv_fp()
            self.pid = os.getpid()

        if self.pid != os.getpid():
            self._mfp.close()
            self._fp.close()
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._mfp, self._fp = self.get_tsv_fp()
            self.pid = os.getpid()

