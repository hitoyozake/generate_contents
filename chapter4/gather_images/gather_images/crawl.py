# -* encoding: utf-8 *-

import os
import sys
import re

files = os.listdir("./spiders/json")

jsonfiles = [ jsons for jsons in files if re.search(".*\.json$", jsons)]

import subprocess

for json in jsonfiles:
    subprocess.call(["scrapy", "runspider", "./spiders/googleimg.py", "-a", "jsonfile="+json])

