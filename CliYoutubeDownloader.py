from pytube import *
import sys


class YouTubeDownloder:
    def __init__(self):
        self.url = str(input("Enter the url of video : "))
        self.youtube = YouTube(
            self.url, on_progress_callback=YouTubeDownloder.onProgress)
        self.showTitle()

    def showTitle(self):
        print("title : {0}\n".format(self.youtube.title))
        self.showStreams()

    def showStreams(self):
        """
        Show the available streams of a video.
        :param self: The object that is calling this function.
        """
        self.streamNo = 1
        for stream in self.youtube.streams:
            print("{0} => resolation:{1}/fps:{2}/type:{3}".format(self.streamNo,
                                                                  stream.resolution, stream.fps, stream.type))
            self.streamNo += 1
        self.chooseStream()

    def chooseStream(self):
        self.choose = int(input("please select one : "))
        self.validateChooseValue()

    def validateChooseValue(self):
        """
        Validates the user input for choosing a stream.

        :param self: The object that this method is bound to. 
                     This parameter must be given as a
        keyword argument, like so: `self=<obj>`.

        :returns: None if the input is valid, otherwise it prints an error message and calls itself again.

        .. note
        :: This function uses recursion to call itself until it gets a valid answer from the user. 
                   It also uses exception handling to catch any
        errors that may occur during execution of this function or others called by it in its stack trace (e.g., ValueError).  

                    # TODO add more
        examples here?  Or maybe just link to another docstring? I'm not sure how much detail we want here... -kmp 27-Jun-2020

                    # TODO add some
        doctests too! -kmp 27-Jun-2020
        """
        if self.choose in range(1, self.streamNo):
            self.getStream()
        else:
            print("please enter a currect option on the list.")
            self.chooseStream()

    def getStream(self):
        self.stream = self.youtube.streams[self.choose-1]
        self.getFileSize()

    def getFileSize(self):
        global file_size
        file_size = self.stream.filesize / 1000000
        self.getPermisionToContinue()

    def getPermisionToContinue(self):
        """
        getPermisionToContinue(self)
            Prints information about the video to be downloaded.
            Asks user if he wants to download it.

            Parameters:
        self (YoutubeDownloader): YoutubeDownloader object that contains all the information needed for downloading a video from youtube.

            Returns:
        None : If user doesn't want to download it, function returns nothing and exits program.
        """
        print("\n title : {0} \n author : {1} \n size : {2:.2f}MB \n resolution : {3} \n fps : {4} \n ".format(
            self.youtube.title, self.youtube.author, file_size, self.stream.resolution, self.stream.fps))
        if input("do you want it ?(defualt = (y)es) or (n)o ") == "n":
            self.showStreams()
        else:
            self.main()

    def download(self):
        self.stream.download()

    @staticmethod
    def onProgress(stream=None, chunk=None,  remaining=None):
        file_downloaded = (file_size-(remaining/1000000))
        print(
            f"downloading ... {file_downloaded/file_size*100:0.2f} % [{file_downloaded:.1f}MB of {file_size:.1f}MB]", end="\r")

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
