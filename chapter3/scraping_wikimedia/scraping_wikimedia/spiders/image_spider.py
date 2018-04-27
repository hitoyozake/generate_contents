# -* encoding: utf-8 *-

import scrapy


class WikimediaSpider(scrapy.Spider):
    """
       wikimediaの画像を集めてくるspider
    """
    name = "WikimediaSpider"

    allow_domains = []
    start_url = ""

    def parse(self, response):
        pass