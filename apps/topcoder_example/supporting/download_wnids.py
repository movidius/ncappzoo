#!/usr/bin/env python

# Original work Copyright (C) <2015-2016> TzuTa Lin and Will Kao
# Modified work Copyright 2018 Jonathan Sculley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import sys
import os
import urllib.request
import urllib.parse


def download_file(url, desc=None, renamed_file=None):
    u = urllib.request.urlopen(url)

    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = 'downloaded.file'

    if not renamed_file is None:
        filename = renamed_file

    if desc:
        filename = os.path.join(desc, filename)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)

    return filename


def getImageURLsOfWnid(wnid):
    url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + str(wnid)
    f = urllib.request.urlopen(url)
    contents = f.read().decode().split('\n')
    imageUrls = []

    for each_line in contents:
        # Remove unnecessary char
        each_line = each_line.replace('\r', '').strip()
        if each_line:
            imageUrls.append(each_line)

    return imageUrls


def mkWnidDir(wnid):
    if not os.path.exists(wnid):
        os.mkdir(wnid)
    return os.path.abspath(wnid)


def downloadImagesByURLs(wnid, imageUrls):
    wnid_urlimages_dir = mkWnidDir(wnid)
    if not os.path.exists(wnid_urlimages_dir):
        os.mkdir(wnid_urlimages_dir)

    for url in imageUrls:
        try:
            download_file(url, wnid_urlimages_dir)
        except Exception as error:
            print('Fail to download : ' + url)
            print(str(error))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Download all images from specific ImageNet wnids')
    p.add_argument('--wnid', nargs='+', help='ImageNet wnids. e.g. : n00007846')
    args = p.parse_args()
    if args.wnid is None:
        print('No wnids')
        sys.exit()

    for id in args.wnid:
        list = getImageURLsOfWnid(id)
        downloadImagesByURLs(id, list)
