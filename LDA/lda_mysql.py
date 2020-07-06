#Insert all lda training transcripts into mysql table
import json,os
import mysql.connector

con = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='lda_db',
                              allow_local_infile = "True")
cursor = con.cursor()

directory = os.fsencode("C:/Users/15099/Documents/Projects/Spotify/LDA Trainning/1-10000/")

transcript_id=1

for file in os.listdir(directory):
    filename = os.fsdecode(directory+file)
    # Opening JSON file
    jsonfile = open(filename,)
    # returns JSON object as a dictionary
    data = json.load(jsonfile)
    full_transcript=""
    # Iterating through the json list
    for results in data['results']:
        for alternatives in results['alternatives']:
            #if the transcript is not null
            if alternatives.get('transcript'):
                full_transcript+=alternatives.get('transcript')
    add_transcript = ("INSERT INTO training "
    "(transcript,transcript_id) "
    "VALUES (%s, %s)")
    data_transcript = (full_transcript,transcript_id)

    #insert into table
    cursor.execute(add_transcript,data_transcript)
    #make sure data is committed to table
    con.commit()
    transcript_id+=1
#close mysql connection
cursor.close()
con.close()
