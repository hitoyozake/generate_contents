# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class GatherImagesItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    image_directory_name = scrapy.Field()
    # Fieldは連想配列みたいなもの
    images = scrapy.Field()
    image_urls = scrapy.Field()

