from djitellopy import Tello

def dvere(tello):
    # create and connect
    # 创建Tello对象并连接

    # configure drone
    # 设置无人机
      # forward detection only  只识别前方

    tello.move_up(150)
    holdingDistance = 90
    dvere = True
    poc = 0
    while dvere:
        pad = tello.get_mission_pad_id()
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
        
    tello.move_down(abs(tello.get_mission_pad_distance_y())+70)
    tello.move_forward(300)
    tello.rotate_counter_clockwise(90)