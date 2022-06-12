
import logging

def write_to_log(txt):
    logging.info(txt)

#> 2019-02-17 11:40:38,254 :: INFO :: Just like that!

def writeToLog(second):
    f = open('./text.txt', 'a')
    f.write("This is a drone, the second: "+str(second)+"\n")
    f.close()

def writeToLog_buf(txt,second,place):
    f = open('./text.txt', 'a')
    f.write(txt+" second: "+str(second)+" place: "+str(place)+"\n")
    f.close()

def writeToFile(txt):
    f = open('./results_of_3models.txt', 'a')
    f.write(txt+"\n")
    f.close()

def writeToFileRes(type_model,kind):
    f = open('./results_of_3models.txt', 'a')
    f.write("-model: "+type_model+" -class: "+kind+"\n")
    f.close()

# import logging
#
# def write_to_log(message):
#   logging.basicConfig(filename='test.log', format='%(asctime)s - %(message)s', datefmt='%d-  %b-%y %H:%M:%S',   level=logging.INFO)
#   logging.info(message)


def getLog():
    with open('./text.txt', 'r') as f:
        f_contents = f.read()
        return f_contents

def send_buf_to_log(buf):
    for i in buf:
        if i.objects!=[]:
            for j in i.objects:
                if j.kindC=='Drone':
                    writeToLog_buf("Found Drone",i.secondC,j.placeC)
                if j.kindC=='Airplain':
                    writeToLog_buf("Found Airplain",i.secondC,j.placeC)
                if j.kindC=='Bird':
                    writeToLog_buf("Found Bird",i.secondC,j.placeC)
                if j.kindC=='Helicopter':
                    writeToLog_buf("Found Helicopter",i.secondC,j.placeC)


def CreateLog():
    logging.basicConfig(filename="log.txt", level=logging.DEBUG)


if __name__ == '__main__':
    CreateLog()