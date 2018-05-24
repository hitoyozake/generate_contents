# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import scrapy
from scrapy.pipelines.images import ImagesPipeline

# そのままだとSHA-1でハッシュ化したファイル名にするのでカスタマイズする
class GatherImagesPipeline(ImagesPipeline):
    def process_item(self, item, spider):
        return item
