from my_crawling.model import ShopCrawler
class Service:
    def __init__(self):
        pass

    def crawling(self, flag):
        if flag == 'oliveyoung':
            print('oliveyoung 크롤링')
            oliveyoung = ShopCrawler('http://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010001&fltDispCatNo=&prdSort=01&pageIdx=1&rowsPerPage=48&searchTypeSort=btn_thumb&plusButtonFlag=N&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0')
            oliveyoung.scrap()