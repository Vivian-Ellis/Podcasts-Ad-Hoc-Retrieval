import os, random, shutil

destination="C:/Users/15099/Documents/Projects/Spotify/LDA Trainning/10001-30000/"
source="C:/Users/15099/Documents/Projects/Spotify/dataset/6to7/"
for i in range(1,6668):
	#grab random file
	filename = random.choice(os.listdir(source))
	filepath=source+"/"+filename

	#move file
	shutil.move(filepath,destination)
