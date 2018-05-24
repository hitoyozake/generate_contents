# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import scrapy
from scrapy.utils.misc import md5sum
from scrapy.pipelines.images import ImagesPipeline

# そのままだとSHA-1でハッシュ化したファイル名にするのでカスタマイズする
class GatherImagesPipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            # metaは itemに定義したField名をKeyとして対応するvalueを持つ
            yield scrapy.Request(image_url, meta={'image_directory_name': item['image_directory_name']})

    def image_downloaded(self, response, request, info):
        checksum = None

        for path, image, buf in self.get_images(response, request, info):
            if checksum is None:
                buf.seek(0)
                checksum = md5sum(buf)

            width, height = image.size

            filename = request._url.rsplit("/", 1)[1]

            path = 'full/{0}/{1}'.format(response.meta['image_directory_name'], filename)

            self.store.persist_file(
                path, buf, info,
                meta={'width': width, 'height': height}
            )

            return checksum

