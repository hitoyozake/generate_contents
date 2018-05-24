# -* encoding: utf-8 *-

import scrapy


class GoogleImageSpider(scrapy.Spider):

    name = "googleImageSpider"

    allowed_domains = ["google.com"]


    start_urls = ['https://www.google.com/search?q=%E3%83%90%E3%83%89%E3%82%AC%E3%83%BC%E3%83%AB+%E5%B7%A8%E4%B9%B3&biw=1739&bih=1179&gbv=1&tbm=isch&ei=WjAEW7PjGsXM0gSe9awg&start=1&sa=N',
                  'https://www.google.com/search?q=%E3%83%90%E3%83%89%E3%82%AC%E3%83%BC%E3%83%AB+%E5%B7%A8%E4%B9%B3&biw=1739&bih=1179&gbv=1&tbm=isch&ei=WjAEW7PjGsXM0gSe9awg&start=20&sa=N',
                  'https://www.google.com/search?q=%E3%83%90%E3%83%89%E3%82%AC%E3%83%BC%E3%83%AB+%E5%B7%A8%E4%B9%B3&biw=1739&bih=1179&gbv=1&tbm=isch&ei=WjAEW7PjGsXM0gSe9awg&start=40&sa=N',
                  'https://www.google.com/search?q=%E3%83%90%E3%83%89%E3%82%AC%E3%83%BC%E3%83%AB+%E5%B7%A8%E4%B9%B3&biw=1739&bih=1179&gbv=1&tbm=isch&ei=WjAEW7PjGsXM0gSe9awg&start=60&sa=N'

                  ]

    def __init__(self):
        print('initialize')
        # もしもURLなどを初期化したいのならここで行う

    def parse(self, response):
        for url in response.xpath("//img/@src").extract():
            print("URL:", url)





