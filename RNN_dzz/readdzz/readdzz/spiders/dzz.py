import scrapy
import re

class DZZSpider(scrapy.Spider):
    name = "dzz"
    start_urls = ['http://www.piaotian.com/html/4/4317/index.html']
    title_pat = re.compile(r'第(\w+)章(.*)(?=$|【)')
    cnum_pat = re.compile(r'(.?千)?(.?百)?(.?十|零)?(.)?')
    c2n = {
        '零':0,
        '一':1,
        '二':2,
        '两':2,
        '三':3,
        '四':4,
        '五':5,
        '六':6,
        '七':7,
        '八':8,
        '九':9,
        '十':10,
        '百':100,
        '千':1000,
        '万':10000,
        '兆':1000000,
        '亿':100000000
    }

    def parse(self, response: scrapy.http.Response):
        selector_xpath = "//div[@class='mainbody']/div[@class='centent']/ul[position()>1]/li/a/@href"
        for i in response.xpath(selector_xpath).extract():
            yield response.follow(i, callback=self.parse_page)

    def parse_page(self, response:scrapy.http.Response):
        selector_xpath_title = "//h1/text()"
        selector_xpath_text = "//body/text()"                                                                           # should be //body/div[@id='main']/div[@id='content']/text(), I have no idea why it resolves to this
        title = response.xpath(selector_xpath_title).extract_first()
        match = re.search(self.title_pat, title)
        if match:
            cnum = match.group(1)
            cindex = 0
            cnum_match = re.match(self.cnum_pat, cnum)
            for group in cnum_match.groups():
                if group:
                    num = 1
                    for i in group:
                        num *= self.c2n[i]
                    cindex += num
            ctitle = match.group(2).strip(' ')
            ctext = ''.join([i for i in response.xpath(selector_xpath_text).extract()])
            ctext = re.sub(r'[\r\n]+', r'\n', ctext)
            page = dict()
            page['index'] = cindex
            page['title'] = ctitle
            page['text'] = ctext
            return page