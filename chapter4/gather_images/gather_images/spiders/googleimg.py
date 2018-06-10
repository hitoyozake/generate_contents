# -* encoding: utf-8 *-

import scrapy
import json
import os
try:
    from gather_images.items import GatherImagesItem
except:
    from gather_images.gather_images.items import GatherImagesItem

class GoogleImageSpider(scrapy.Spider):

    name = "googleImageSpider"

    # allowed_domains = ["google.com", "https://encrypted-tbn0.gstatic.com"]

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
            jsdata = json.loads(f.read())

            for i in jsdata:
                """
                if 'start_urls' in jsdata[i]:
                    for url in jsdata[i]['start_urls']:
                        GoogleImageSpider.start_urls.append(url)
                """
                if 'keywords' in jsdata[i]:
                    for keyword in jsdata[i]['keywords']:
                        ar = self.generate(keyword)
                        for url in ar:
                            GoogleImageSpider.start_urls.append(url)

    def parse(self, response):
        item = GatherImagesItem()
        print("******PARSE*******")
        item["image_urls"] = []
        item["image_directory_name"]="dl"
        for url in response.xpath("//img/@src").extract():
            item['image_urls'].append(url)

        return item



