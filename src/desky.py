from djitellopy import Tello

# create and connect
# 创建Tello对象并连接
tello = Tello()
tello.connect()

# configure drone
# 设置无人机
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(1)  # forward detection only  只识别前方

tello.takeoff()
tello.move_up(200)
pad = tello.get_mission_pad_id()

print(tello.get_battery())
# detect and react to pads until we see pad #1
# 发现并识别挑战卡直到看见1号挑战卡
holdingDistance = 90
dvere = True
poc = 0
while dvere:
    if pad == 4:

        print("x",tello.get_mission_pad_distance_x(),"z",tello.get_mission_pad_distance_z(),"y",tello.get_mission_pad_distance_y())
        #print("y",tello.get_mission_pad_distance_y())
        #print("z",tello.get_mission_pad_distance_z())
        #print(tello.get_battery())

        """
        disy = tello.get_mission_pad_distance_y()
        if disy > 19:
            #print("hor")
            tello.move_up(disy)
        elif disy<-19:
            #print("dol")
            tello.move_down(abs(disy))
        #else:
            #print("stred")
        """


        disx = tello.get_mission_pad_distance_x()
        if disx > 9:
            #print("pravá")
            tello.move_right(disx+10)
            poc = 0
        elif disx<-9:
            #print("levá")
            tello.move_left(abs(disx)+10)
            poc = 0
        elif True:
            #print("stred")
            disz = tello.get_mission_pad_distance_z()
            if disz < holdingDistance-20:
                #print("před")
                tello.move_back(holdingDistance-disz)
            elif disz > holdingDistance+20:
                #print("zad")
                tello.move_forward(abs(disz-holdingDistance))
            else:
                poc+=1
                print(f"pocet: {poc}")
                if poc == 69:
                    dvere = False
    pad = tello.get_mission_pad_id()
tello.move_down(abs(tello.get_mission_pad_distance_y())+70)
tello.move_forward(300)
# graceful termination
# 安全结束程序
tello.disable_mission_pads()
tello.land()
tello.end()