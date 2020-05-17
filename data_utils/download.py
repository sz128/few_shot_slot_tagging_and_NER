#!/usr/bin/env python3

'''
@Time   : 2020-05-11 21:31:37
@Author : su.zhu
@Desc   : 
'''

import os
import urllib.request
from io import BytesIO
from zipfile import ZipFile

def loadData():
    data_url = "data/ACL2020data"
    dataset_url = "https://atmahou.github.io/attachments/ACL2020data.zip"
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(data_url):
        print("Downloading and unzipping the ACL2020 dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("data")
        zip_ref.close()

if __name__ == '__main__':
	loadData()
