from djitellopy import Tello

tello = Tello()
tello.connect(False)



def mesurments():
    responses = tello.send_read_command('EXT tof?').split()
    if len(responses) < 4:
        return mesurments()
    else:
        #for i in range(1,4):
            # if int(responses[i]) == 0:
            #     responses[i] = "999"
        return [int(responses[1]),int(responses[2]),int(responses[3])]
    

#while True:
    print(mesurments())

