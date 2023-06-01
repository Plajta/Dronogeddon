from djitellopy import Tello

tello = Tello()
tello.connect(False)



def mesurments():
    responses = tello.send_read_command('EXT tof?').split()
    if len(responses) < 4:
        return mesurments()
    else:
        return [int(responses[1]),int(responses[2]),int(responses[3])]
    

#while True:
    mesurments()

