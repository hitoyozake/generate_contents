# -* encoding: utf-8 *-

import scrapy


class WikimediaSpider(scrapy.Spider):
    name = "WikimediaSpider"

    allow_domains = []
    start_url = ""

    def parse(self, response):
        pass