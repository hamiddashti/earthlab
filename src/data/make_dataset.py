# File used in this script are sided on my personal Google Drive
import gdown


def gdown_file(url, outname):
    gdown.download(url, outname, quiet=False)
