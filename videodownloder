from pytube import YouTube 

#location where you save.
PATH = "E:/" #to_do 

#link of video.
link=["https://www.youtube.com/watch?v=p8FuTenSWPI", 
	"https://www.youtube.com/watch?v=JWbnEt3xuos"
	]#list of video links. 
for i in link: 
	try: 
		yt = YouTube(i) 
	except: 
		print("Connection Error") #to handle exception 
	
	#check files with "mp4" extension 
	mp4files = yt.filter('mp4') 

	d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution) 
	try: 
		d_video.download(__PATH) 
	except: 
		print("Some Error!") 
print('Task Completed!') 
