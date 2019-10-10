# only god knows  whats happening in the code
# if I forget the code structure
# please pray to god for help
import asyncio
import multiprocessing
import os
import random
import re
import socket
import sys
import time

import bs4
import pymongo
import requests
import socks
import ujson
import urllib3

try:
    import instagram_monitering.con_file as config
except:
    import con_file as config


class PorxyApplyingDecorator(object):

    def __init__(self):
        filename = os.getcwd() + "/" + "ipList.txt"
        with open(filename, "r") as f:
            ipdata = f.read()
        self._IP = random.choice(ipdata.split(","))

    def __call__(self, function_to_call_for_appling_proxy):
        SOCKS5_PROXY_HOST = self._IP
        # default_socket = socket.socket
        socks.set_default_proxy(socks.SOCKS5,
                                SOCKS5_PROXY_HOST,
                                config.SOCKS5_PROXY_PORT,
                                True,
                                config.auth,
                                config.passcode)
        socket.socket = socks.socksocket

        def wrapper_function(url):
            # this is used for applyting socks5 proxy over the request
            return function_to_call_for_appling_proxy(url)

        socks.set_default_proxy()
        return wrapper_function


async def dataprocess(htmldata):
    bs4obj = bs4.BeautifulSoup(htmldata, "html.parser")
    scriptsdata = bs4obj.findAll("script", {"type": "text/javascript"})
    datatext = ''
    for i in scriptsdata:
        datatext = i.text
        if "window._sharedData =" in datatext:
            break
    datajson = re.findall("{(.*)}", datatext)
    datajson = '{' + datajson[0] + '}'
    datadict = ujson.loads(datajson)
    maindict = {}
    datadict = datadict["entry_data"]["PostPage"][0]["graphql"]["shortcode_media"]
    tofind = ["owner", "location"]
    for i in tofind:
        try:
            maindict[i] = datadict[i]
        except:
            pass
    return maindict


async def datapullpost(future, url):
    while True:
        @PorxyApplyingDecorator()
        async def request_pull(url):
            data = None
            print(url)
            urllib3.disable_warnings()
            user_agent = {'User-agent': 'Mozilla/17.0'}
            try:
                data = requests.get(url=url, headers=user_agent, timeout=10, verify=False).text
            except:
                data = None
            finally:
                return data

        data = await request_pull(url)
        if data != None:
            break
    data = await dataprocess(htmldata=data)
    # here processing of data has to occur
    future.set_result(data)


class MoniteringClass():

    def __init__(self, user, tags, type, productId):

        try:
            self.mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
            db = self.mon[productId + ":" + user + ":insta"]
            self._collection = db[tags]
            if type == "hashtags":
                self._url = "https://www.instagram.com/explore/tags/" + tags + "/?__a=1"
            if type == "profile":
                self._url = "https://www.instagram.com/" + tags + "/?__a=1"
        except:
            print("error::MointeringClass.__init__>>", sys.exc_info()[1])

    def _dataProcessing(self, data):
        loop = asyncio.get_event_loop()
        userdata = []
        try:
            if not isinstance(data, dict):
                raise Exception
            media_post = data['tag']["media"]["nodes"]
            top_post = data['tag']["top_posts"]["nodes"]
            print("media post ::", len(media_post))
            print("top_post::", len(top_post))
            futures = []
            for i in media_post:
                tempdict = {}
                tempdict["url"] = "https://www.instagram.com/p/" + i["code"] + "/"
                tempdict["code"] = i["code"]
                userdata.append(tempdict)
            for i in top_post:
                tempdict = {}
                tempdict["url"] = "https://www.instagram.com/p/" + i["code"] + "/"
                tempdict["code"] = i["code"]
                userdata.append(tempdict)
            for i in userdata:
                i["future"] = asyncio.Future()
                futures.append(i["future"])
                asyncio.ensure_future(datapullpost(future=i["future"], url=i["url"]))
            loop.run_until_complete(asyncio.wait(futures))
            for i in userdata:
                i["data"] = i["future"].result()
        except:
            print("error::Monitering.dataProcessing>>", sys.exc_info()[1])
        finally:
            # loop.close()
            print("userdata::", len(userdata))
            print("media_post::", len(media_post))
            print("top post::", len(top_post))
            return userdata, media_post, top_post

    def _insertFunction(self, record):
        try:
            records = self._collection.find({"id": record["id"]})
            if records.count() == 0:
                # record["timestamp"] = time.time()
                self._collection.insert(record)
        except:
            print("error::Monitering.insertFunction>>", sys.exc_info()[1])

    def _lastProcess(self, userdata, media_post, top_post):
        mainlist = []
        try:
            for i in userdata:
                for j in media_post:
                    if i["code"] == j["code"]:
                        tempdict = j.copy()
                        tofind = ["owner", "location"]
                        for z in tofind:
                            try:
                                tempdict[z + "data"] = i["data"][z]
                            except:
                                pass
                        mainlist.append(tempdict)
                        self._insertFunction(tempdict.copy())
                for k in top_post:
                    if i["code"] == k["code"]:
                        tempdict = k.copy()
                        tofind = ["owner", "location"]
                        for z in tofind:
                            try:
                                tempdict[z + "data"] = i["data"][z]
                            except:
                                pass
                        mainlist.append(tempdict)
                        self._insertFunction(tempdict.copy())
        except:
            print("error::lastProcess>>", sys.exc_info()[1])

    def request_data_from_instagram(self):
        try:
            while True:
                @PorxyApplyingDecorator()
                def reqest_pull(url):
                    print(url)
                    data = None
                    urllib3.disable_warnings()
                    user_agent = {'User-agent': 'Mozilla/17.0'}
                    try:
                        data = requests.get(url=url, headers=user_agent, timeout=24, verify=False).text
                    except:
                        data = None
                    finally:
                        return data

                data = reqest_pull(self._url)
                if data != None:
                    break
            datadict = ujson.loads(data)
            userdata, media_post, top_post = self._dataProcessing(datadict)
            finallydata = (self._lastProcess(userdata=userdata, media_post=media_post, top_post=top_post))
            # print(ujson.dumps(finallydata))
        except:
            print("error::Monitering.request_data_from_instagram>>", sys.exc_info()[1])

    def __del__(self):
        self.mon.close()


def hashtags(user, tags, type, productId):
    try:
        temp = MoniteringClass(user=user, tags=tags, type=type, productId=productId)
        temp.request_data_from_instagram()
    except:
        print("error::hashtags>>", sys.exc_info()[1])


class theradPorcess(multiprocessing.Process):

    def __init__(self, user, tags, type, productId):
        try:
            multiprocessing.Process.__init__(self)
            self.user = user
            self.tags = tags
            self.type = type
            self.productId = productId
        except:
            print("errorthreadPorcess:>>", sys.exc_info()[1])

    def run(self):
        try:
            hashtags(user=self.user, tags=self.tags, type=self.type, productId=self.productId)
        except:
            print("error::run>>", sys.exc_info()[1])


class InstaPorcessClass():

    def _dbProcessReader(self, user, tags, productId):
        value = True
        mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
        try:
            db = mon["insta_process"]
            collection = db["process"]
            temp = {}
            temp["user"] = user
            temp["tags"] = tags
            temp["productId"] = productId
            records = collection.find(temp).count()
            if records == 0:
                raise Exception
            value = True
        except:
            value = False
            print("error::dbProcessReader:>>", sys.exc_info()[1])
        finally:
            mon.close()
            return value

    def _processstart(self, user, tags, productId):
        mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
        try:
            db = mon["insta_process"]
            collection = db["process"]
            temp = {}
            temp["user"] = user
            temp["tags"] = tags
            temp["productId"] = productId
            collection.insert(temp)
        except:
            print("error::processstart>>", sys.exc_info()[1])
        finally:
            mon.close()

    def startprocess(self, user, tags, type, productId):
        try:
            self._processstart(user=user, tags=tags, productId=productId)
            while True:
                # therad = theradPorcess(user=user, tags=tags, type=type)
                # therad.start()
                hashtags(user=user, tags=tags, type=type, productId=productId)
                check = self._dbProcessReader(user=user, tags=tags, productId=productId)
                print(check)
                if check == False:
                    break
                time.sleep(300)
                # therad.join()
        except:
            print("error::startPoress::>>", sys.exc_info()[1])

    def deletProcess(self, user, tags, productId):
        mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
        try:
            db = mon["insta_process"]
            collection = db["process"]
            temp = {}
            temp["user"] = user
            temp["tags"] = tags
            temp["productId"] = productId
            collection.delete_one(temp)
        except:
            print("error::deletProcess:>>", sys.exc_info()[1])
        finally:
            mon.close()
            print("deleted - task", temp)
            return True

    def statusCheck(self, user, tags, productId):
        mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
        try:
            db = mon["insta_process"]
            collection = db["process"]
            temp = {}
            temp["user"] = user
            temp["tags"] = tags
            temp["productId"] = productId
            records = collection.find(temp).count()
            if records == 0:
                result = False
            else:
                result = True
        except:
            print("error::dbProcessReader:>>", sys.exc_info()[1])
        finally:
            mon.close()
            return result


class DBDataFetcher():

    def __init__(self, user, tags, type, productId):
        try:
            self.mon = pymongo.MongoClient(host=config.host, port=config.mongoPort)
            db = self.mon[productId + ":" + user + ":insta"]
            self._collection = db[tags]
        except:
            print("error::DBDataFetcher.init>>", sys.exc_info()[1])

    def dbFetcher(self, limit=20):
        mainlist = []
        try:
            records = self._collection.find().sort("id", -1).limit(limit)
            for i in records:
                del i["_id"]
                mainlist.append(i)
        except:
            print("error::dbFetcher>>", sys.exc_info()[1])
        finally:
            return ujson.dumps(mainlist)

    def DBFetcherGreater(self, limit, date):
        mainlist = []
        postval = {}
        try:
            postval["posts"] = None
            if limit.isdigit() == False and date.isdigit() == False:
                raise Exception
            limit = int(limit)
            date = int(date)
            if date != 0:
                doc = self._collection.find({"date": {"$gt": date}}).sort("date", pymongo.ASCENDING).limit(limit)
            else:
                doc = self._collection.find().sort("date", pymongo.ASCENDING).limit(limit)
            for i in doc:
                del i["_id"]
                mainlist.append(i)
            postval["posts"] = mainlist
            postval["status"] = True
        except:
            print("error::", sys.exc_info()[1])
            postval["status"] = False
        finally:
            return ujson.dumps(postval)

    def DBFetcherLess(self, limit, date):
        mainlist = []
        postval = {}
        try:
            postval["posts"] = None
            if limit.isdigit() == False and date.isdigit() == False:
                raise Exception
            limit = int(limit)
            date = int(date)
            doc = self._collection.find({"date": {"$lt": date}}).limit(limit).sort("date", pymongo.DESCENDING)
            for i in doc:
                del i["_id"]
                mainlist.append(i)
            postval["posts"] = mainlist[::-1]
            postval["status"] = True
        except:
            print("error::", sys.exc_info()[1])
            postval["status"] = False
        finally:
            return ujson.dumps(postval)

    def __del__(self):
        self.mon.close()


def main():
    try:
        user = sys.argv[1]
        tags = sys.argv[2]
        type = sys.argv[3]
        productId = sys.argv[4]
        obj = InstaPorcessClass()
        obj.startprocess(user=user, tags=tags, type=type, productId=productId)
    except:
        print("error::main>>", sys.exc_info()[1])


if __name__ == '__main__':
    main()
