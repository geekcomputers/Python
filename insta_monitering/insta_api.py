from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web
from tornado.concurrent import run_on_executor
from tornado.gen import coroutine

# import file
try:
    from instagram_monitering.insta_datafetcher import *
    from instagram_monitering.subpinsta import *
except:
    from insta_datafetcher import *
    from subpinsta import *
MAX_WORKERS = 10


class StartHandlerinsta(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    @run_on_executor
    def background_task(self, user, tags, type, productId):
        try:
            instasubprocess(user=user, tags=tags, type=type, productId=productId)
        except:
            print("error::background_task>>", sys.exc_info()[1])

    @coroutine
    def get(self):
        try:
            q = self.get_argument("q")
            user = self.get_argument("userId")
            type = self.get_argument("type")
            productId = self.get_argument("productId")
        except:
            self.send_error(400)
        if " " in q:
            q = q.replace(" ", "")
        self.background_task(user=user, tags=q, type=type, productId=productId)
        temp = {}
        temp["query"] = q
        temp["userId"] = user
        temp["status"] = True
        temp["productId"] = productId
        print(
            "{0}, {1}, {2}, {3}".format(
                temp["userId"], temp["productId"], temp["query"], temp["status"]
            )
        )
        self.write(ujson.dumps(temp))


class StopHandlerinsta(tornado.web.RequestHandler):
    def get(self):
        try:
            q = self.get_argument("q")
            user = self.get_argument("userId")
            # tags = self.get_argument("hashtags")
            productId = self.get_argument("productId")
        except:
            self.send_error(400)
        obj = InstaPorcessClass()
        result = obj.deletProcess(tags=q, user=user, productId=productId)
        temp = {}
        temp["query"] = q
        temp["userId"] = user
        temp["productId"] = productId
        temp["status"] = result
        print(
            "{0}, {1}, {2}, {3}".format(
                temp["userId"], temp["productId"], temp["query"], temp["status"]
            )
        )
        self.write(ujson.dumps(temp))


class StatusHandlerinsta(tornado.web.RequestHandler):
    def get(self):
        try:
            q = self.get_argument("q")
            user = self.get_argument("userId")
            productId = self.get_argument("productId")
            # tags = self.get_argument("hashtags")
        except:
            self.send_error(400)
        obj = InstaPorcessClass()
        result = obj.statusCheck(tags=q, user=user, productId=productId)
        temp = {}
        temp["query"] = q
        temp["userId"] = user
        temp["status"] = result
        temp["productId"] = productId
        print(
            "{0}, {1}, {2}, {3}".format(
                temp["userId"], temp["productId"], temp["query"], temp["status"]
            )
        )
        self.write(ujson.dumps(temp))


# class SenderHandlerinsta(tornado.web.RequestHandler):
#     def get(self):
#         try:
#             q = self.get_argument("q")
#             user = self.get_argument("userId")
#             type = self.get_argument("type")
#             productId = self.get_argument("productId")
#         except:
#             self.send_error(400)
#         recordsobj = DBDataFetcher(user=user, tags=q, type=type, productId=productId)
#         data = recordsobj.dbFetcher()
#         self.write(data)


class SenderHandlerinstaLess(tornado.web.RequestHandler):
    def get(self):
        try:
            q = self.get_argument("q")
            user = self.get_argument("userId")
            type = self.get_argument("type")
            productId = self.get_argument("productId")
            date = self.get_argument("date")
            limit = self.get_argument("limit")
        except:
            self.send_error(400)
        recordsobj = DBDataFetcher(user=user, tags=q, type=type, productId=productId)
        data = recordsobj.DBFetcherLess(limit=limit, date=date)
        # print("{0}, {1}, {2}, {3}".format(temp["userId"], temp["productId"], temp["query"], temp["status"]))
        self.write(data)


class SenderHandlerinstaGreater(tornado.web.RequestHandler):
    def get(self):
        try:
            q = self.get_argument("q")
            user = self.get_argument("userId")
            type = self.get_argument("type")
            productId = self.get_argument("productId")
            date = self.get_argument("date")
            limit = self.get_argument("limit")
        except:
            self.send_error(400)
        recordsobj = DBDataFetcher(user=user, tags=q, type=type, productId=productId)
        data = recordsobj.DBFetcherGreater(limit=limit, date=date)
        # print("{0}, {1}, {2}, {3}".format(temp["userId"], temp["productId"], temp["query"], temp["status"]))
        self.write(data)


if __name__ == "__main__":
    application = tornado.web.Application(
        [
            (r"/instagram/monitoring/start", StartHandlerinsta),
            (r"/instagram/monitoring/stop", StopHandlerinsta),
            (r"/instagram/monitoring/status", StatusHandlerinsta),
            (r"/instagram/monitoring/less", SenderHandlerinstaLess),
            (r"/instagram/monitoring/greater", SenderHandlerinstaGreater),
        ]
    )

    application.listen(7074)
    print("server running")
    tornado.ioloop.IOLoop.instance().start()
