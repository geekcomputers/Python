# libraraies

import pytube
import sys


class YouTubeDownloder:
    def __init__(self):
        self.url = str(input("Enter the URL of video : "))
        self.youtube = pytube.YouTube(
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
                "{0} => resolution:{1}/fps:{2}/type:{3}".format(
                    self.streamNo, stream.resolution, stream.fps, stream.type
                )
            )
            self.streamNo += 1
        self.chooseStream()

    def chooseStream(self):
        self.choose = int(input("Please select one : "))
        self.validateChooseValue()

    def validateChooseValue(self):
        if self.choose in range(1, self.streamNo):
            self.getStream()
        else:
            print("Please enter a correct option on the list.")
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
            "\n Title : {0} \n Author : {1} \n Size : {2:.2f}MB \n Resolution : {3} \n FPS : {4} \n ".format(
                self.youtube.title,
                self.youtube.author,
                file_size,
                self.stream.resolution,
                self.stream.fps,
            )
        )
        if input("Do you want it ?(default = (y)es) or (n)o ") == "n":
            self.showStreams()
        else:
            self.main()

    def download(self):
        self.stream.download()

    @staticmethod
    def onProgress(stream=None, chunk=None, remaining=None):
        file_downloaded = file_size - (remaining / 1000000)
        print(
            f"Downloading ... {file_downloaded/file_size*100:0.2f} % [{file_downloaded:.1f}MB of {file_size:.1f}MB]",
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
