# File used in this script are sided on my personal Google Drive
import gdown


def gdown_file(url, outname):
    """Download from Google Drive

    :param str url: the google drive download link
    :param str outname: the output name for downloaded file

    """
    gdown.download(url, outname, quiet=True)
