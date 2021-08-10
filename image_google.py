from icrawler.builtin import BingImageCrawler
crawler = BingImageCrawler(storage={"root_dir": "man"})
crawler.crawl(keyword="俳優", max_num=100)