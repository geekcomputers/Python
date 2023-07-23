from pytube import *
import sys


class YouTubeDownloder:
    def __init__(self):
        self.url = str(input("Enter the url of video : "))
        self.youtube = YouTube(
            self.url, on_progress_callback=YouTubeDownloder.onProgress
        )
        self.showTitle()

    def showTitle(self):
        print("title : {0}\n".format(self.youtube.title))
        self.showStreams()

    def showStreams(self):
        self.streamNo = 1
        for stream in self.youtube.streams:
            print(
                "{0} => resolation:{1}/fps:{2}/type:{3}".format(
                    self.streamNo, stream.resolution, stream.fps, stream.type
                )
            )
            self.streamNo += 1
        self.chooseStream()

    def chooseStream(self):
        self.choose = int(input("please select one : "))
        self.validateChooseValue()

    def validateChooseValue(self):
        if self.choose in range(1, self.streamNo):
            self.getStream()
        else:
            print("please enter a currect option on the list.")
            self.chooseStream()

    def getStream(self):
        self.stream = self.youtube.streams[self.choose - 1]
        self.getFileSize()

    def getFileSize(self):
        global file_size
        file_size = self.stream.filesize / 1000000
        self.getPermisionToContinue()

    def getPermisionToContinue(self):
        print(
            "\n title : {0} \n author : {1} \n size : {2:.2f}MB \n resolution : {3} \n fps : {4} \n ".format(
                self.youtube.title,
                self.youtube.author,
                file_size,
                self.stream.resolution,
                self.stream.fps,
            )
        )
        if input("do you want it ?(defualt = (y)es) or (n)o ") == "n":
            self.showStreams()
        else:
            self.main()

    def download(self):
        self.stream.download()

    @staticmethod
    def onProgress(stream=None, chunk=None, remaining=None):
        file_downloaded = file_size - (remaining / 1000000)
        print(
            f"downloading ... {file_downloaded/file_size*100:0.2f} % [{file_downloaded:.1f}MB of {file_size:.1f}MB]",
            end="\r",
        )

    def main(self):
        try:
            self.download()
        except KeyboardInterrupt:
            print("Canceled. ")
            sys.exit(0)


if __name__ == "__main__":
    try:
        YouTubeDownloder()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
