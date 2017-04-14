import os
import sys
import shutil

Music = ['MP3', 'WAV', 'WMA', 'MKA', 'AAC', 'MID', 'RA', 'RAM', 'RM', 'OGG']
Codes = ['CPP', 'RB', 'PY', 'HTML', 'CSS', 'JS']
Compressed = ['RAR', 'JAR', 'ZIP', 'TAR', 'MAR', 'ISO', 'LZ', '7ZIP', 'TGZ', 'GZ', 'BZ2']
Documents = ['DOC', 'DOCX', 'PPT', 'PPTX', 'PAGES', 'PDF', 'ODT', 'ODP', 'XLSX', 'XLS', 'ODS', 'TXT', 'IN', 'OUT', 'MD']
Images = ['JPG', 'JPEG', 'GIF', 'PNG', 'SVG']
Executables = ['LNK','DEB', 'EXE', 'SH', 'BUNDLE']
Video = ['FLV', 'WMV', 'MOV', 'MP4', 'MPEG', '3GP', 'MKV','AVI']


def getVideo():
	return Video

def getMusic():
	return Music

def getCodes():
	return Codes

def getCompressed():
	return Compressed

def getImages():
	return Images

def getExe():
	return Executables

def getDoc():
	return Documents

# taking the location of the Folder to Arrange
try:
	arrange_dir = str(sys.argv[1])
except IndexError:
	arrange_dir = str(raw_input("Enter the Path of directory: "))

# when we make a folder that already exist then WindowsError happen
# changing directory may give WindowsError

def change(direc):
	try:
		os.chdir(direc)
		#print "path changed"
	except WindowsError:
		print "Error! Cannot change the Directory"
		print "Enter a valid directory!"
		direc = str(raw_input("Enter the Path of directory: "))
		change(direc)

change(arrange_dir)

# now we will get the list of all the directories in the folder

list_dir = os.listdir(os.getcwd())

#print list_dir

#check_Folder = False # for organising Folders
check_Music = False
check_Video = False
check_Exe = False
check_Code = False
check_Compressed = False
check_Img = False
check_Docs = False


main_names = ['Video','Folders','Images','Documents','Music','Codes','Executables','Compressed']

for name in list_dir:
	#print name.split('.')
	if len(name.split('.')) == 2:

		if name.split('.')[1].upper() in getVideo():
			try:
				os.mkdir("Video")
				print "Video Folder Created"
			except WindowsError:
				print "Images Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Video"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getImages():
			try:
				os.mkdir("Images")
				print "Images Folder Created"
			except WindowsError:
				print "Images Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Images"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getMusic():
			try:
				os.mkdir("Music")
				print "Music Folder Created"
			except WindowsError:
				print "Music Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Music"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getDoc():
			try:
				os.mkdir("Documents")
				print "Documents Folder Created"
			except WindowsError:
				print "Documents Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Documents"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getCodes():
			try:
				os.mkdir("Codes")
				print "Codes Folder Created"
			except WindowsError:
				print "Codes Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Codes"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getCompressed():
			try:
				os.mkdir("Compressed")
				print "Compressed Folder Created"
			except WindowsError:
				print "Compressed Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Compressed"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
		elif name.split('.')[1].upper() in getExe():
			try:
				os.mkdir("Executables")
				print "Executables Folder Created"
			except WindowsError:
				print "Executables Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Executables"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)
			#print "It is a folder"
	else:
		if name not in main_names:
			try:
				os.mkdir("Folders")
				print "Folders Folder Created"
			except WindowsError:
				print "Folders Folder Exists"
	
			old_dir = arrange_dir + "\\" + name
			new_dir = arrange_dir + "\Folders"
			os.chdir(new_dir)
			shutil.move(old_dir, new_dir + "\\" + name)
			print os.getcwd()
			os.chdir(arrange_dir)



print "Done Arranging Files and Folder in your specified directory"