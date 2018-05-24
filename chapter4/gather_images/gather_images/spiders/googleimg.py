# -* encoding: utf-8 *-

import scrapy
import json
import os

class GoogleImageSpider(scrapy.Spider):

    name = "googleImageSpider"

    allowed_domains = ["google.com"]

    start_urls = []

    def generate(self, keyword):
        ar = []
        for i in range(0, 8):
            s = "https://www.google.com/search?q={0}&" \
                "biw=1739&bih=1179&gbv=1&tbm=isch&ei=WjAEW7PjGsXM0gSe9awg&start={1}&sa=N".format(keyword, str(int(i*20)))
            ar.append(s)
        return ar

    def __init__(self):
        print('initialize')
        # もしもURLなどを初期化したいのならここで行う
        root = os.path.dirname(os.path.abspath(__file__)) + "/"

        with open(root+"start_urls.json") as f:
            jsdata = json.loads(f.read(), 'shift_jis')

            for i in jsdata:
                if 'start_urls' in i:
                    for url in i['start_urls']:
                        GoogleImageSpider.start_urls.append(url)
                if 'keywords' in i:
                    for keyword in i['keyword']:
                        ar = self.generate(keyword)
                        for url in ar:
                            GoogleImageSpider.start_urls.append(url)

    def parse(self, response):
        for url in response.xpath("//img/@src").extract():
            print("URL:", url)





