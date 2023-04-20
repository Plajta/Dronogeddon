from djitellopy import Tello

tello = Tello()
tello.connect(False)

def mesurments():
    responses = tello.send_read_command('EXT tof?').split()
    mesurments = [int(responses[1]),int(responses[2]),int(responses[3])]
    return mesurments

