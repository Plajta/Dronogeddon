/*
 * Copyright (C) 2020 DJI.
 *
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-08-25     robomaster   first version
 */

#include <SoftwareSerial.h>   
SoftwareSerial mySerial(13, 14); 
 
int dist;                     
int strength;                 
int check;                    
int i;
int uart[9];                   
const int HEADER = 0x59;  


#include <Wire.h>
#include <VL53L1X.h>

// The number of sensors in your system.
const uint8_t sensorCount = 3;

// The Arduino pin connected to the XSHUT pin of each sensor.
const uint8_t xshutPins[sensorCount] = { 4, 5, 2 };

VL53L1X sensors[sensorCount];
#include <Arduino.h>
#include <stdio.h>
#include <string.h>
#include <Wire.h>
#include "FS.h"
#include "SPIFFS.h"
#include <RMTT_Libs.h>
#include <RMTT_Shell.h>
#include <RMTT_Protocol.h>
#include <RMTT_GamesirT1d.h>

// #define __UART0_DEBUG__
// #define __DEFAULT_LOG__

#ifdef __UART0_DEBUG__
#define CommonSerial Serial
#else
#define CommonSerial Serial1
#endif

#define SDK_VERSION "esp32v1.0.0.11"

/* key doubleclick */
#define DOUBLECLICK_INTTERVAL_TIME 500

int led_callback(int argc, char *argv[], char argv2[]);
void led_task(void *pParam);

/* matrix */
#define MLED_BRIGHT   0xFF

int tof_range1 = 0;
int tof_range2 = 0;
int tof_range3 = 0;

int tof_callback(int argc, char *argv[], char argv2[]);
int version_callback(int argc, char *argv[], char argv2[]);
int rmtt_callback(int argc, char *argv[], char argv2[]);
int custom_callback(int argc, char *argv[], char argv2[]);

int unknown_cmd_callback(int argc, char *argv[], char argv2[]);

bool rmtt_int_is_valid();
bool rmtt_bool_is_valid();
bool rmtt_joystick_mac_is_valid();

int get_rmtt_int();
bool get_rmtt_bool();
uint8_t *get_rmtt_joystick_mac();

void matrix_show_graph_from_file();

RMTT_RGB tt_rgb;
RMTT_Matrix tt_matrix;
RMTT_GamesirT1d *p_tt_gamesir;
RMTT_TOF tt_tof;
RMTT_Protocol tt_sdk;

bool pair_mode = false;

TaskHandle_t gamesirPairingTaskHandle = NULL;
TaskHandle_t gamesirTaskHandle = NULL;
TaskHandle_t tofBatteryReadTaskHandle = NULL;
TaskHandle_t bleStatusTaskHandle = NULL;

void gamesir_pairing_task(void *arg);
void gamesir_task(void *arg);
void tof_battery_read_task(void *arg);
void ble_status_task(void *arg);
void wifi_upgrade();

// 16x8 heart figure.
static uint8_t tt_graph_buff[] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0, 255, 255, 255,   0, 255,   0, 255,   0, 255,   0, 255,   0,
      0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,
      0,   0,   0,   0, 255, 255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

int tof_callback(int argc, char *argv[], char argv2[])
{
    CommonSerial.printf("tof %d %d %d %d %d", tof_range1, tof_range2, tof_range3, dist, strength);
}

void IRAM_ATTR key_doubleclick()
{
    static uint32_t last_press_time = 0, now_press_time = 0;
    static uint8_t key_toggle = 0;

    uint32_t interval;

    if (digitalRead(34) != 0)
    {
        return;
    }

    if (last_press_time == 0)
    {
        last_press_time = millis();
    }

    now_press_time = millis();

    interval = now_press_time - last_press_time;

    if ((interval > 50) && (interval < DOUBLECLICK_INTTERVAL_TIME))
    {
         if (!key_toggle)
        {
            CommonSerial.printf("[TELLO] motoron");
#ifdef __DEFAULT_LOG__
            Serial.printf("[TELLO] motoron");
#endif
        }
        else
        {
            CommonSerial.printf("[TELLO] motoroff");
#ifdef __DEFAULT_LOG__
            Serial.printf("[TELLO] motoroff");
#endif
        }
        key_toggle = !key_toggle;
    }
    last_press_time = now_press_time;
}

int i2c_init_failed = 0;

/**
 * Initialize the Matrix and show a TT Logo
 */
void setup_graph()
{
    if (!i2c_init_failed)
    {
        RMTT_Matrix::Init(127);
        RMTT_Matrix::SetLEDStatus(RMTT_MATRIX_CS, RMTT_MATRIX_SW,
                                  RMTT_MATRIX_LED_ON);
        // Set LED brightness for all LEDs from an array.
        RMTT_Matrix::SetAllPWM((uint8_t *)tt_graph_buff);
        delay(200);
        matrix_show_graph_from_file();
    }

}

void setup()
{
  mySerial.begin(115200);

    while (!Serial) {}
  Serial.begin(115200);
  Wire.begin(27, 26);
  Wire.setClock(400000);

  // Disable/reset all sensors by driving their XSHUT pins low.
  for (uint8_t i = 0; i < sensorCount; i++)
  {
    pinMode(xshutPins[i], OUTPUT);
    digitalWrite(xshutPins[i], LOW);
  }

  // Enable, initialize, and start each sensor, one by one.
  for (uint8_t i = 0; i < sensorCount; i++)
  {
    // Stop driving this sensor's XSHUT low. This should allow the carrier
    // board to pull it high. (We do NOT want to drive XSHUT high since it is
    // not level shifted.) Then wait a bit for the sensor to start up.
    pinMode(xshutPins[i], INPUT);
    delay(10);

    sensors[i].setTimeout(500);
    if (!sensors[i].init())
    {
      Serial.print("Failed to detect and initialize sensor ");
      Serial.println(i);
      while (1);
    }
    

    // Each sensor must have its address changed to a unique value other than
    // the default of 0x29 (except for the last one, which could be left at
    // the default). To make it simple, we'll just count up from 0x2A.
    sensors[i].setAddress(0x2A + i);

    sensors[i].startContinuous(50);
  }
    pinMode(34, INPUT_PULLUP);
    if (digitalRead(34) == 0)
    {
        wifi_upgrade();
    }

    // put your setup code here, to run once:
    
    Serial1.begin(1000000, 23, 18, SERIAL_8N1);
    Serial.println();

    // user key, 0:press, 1:up
    pinMode(34, INPUT_PULLUP);
    attachInterrupt(34, key_doubleclick, FALLING);

    Serial.println("*********RoboMaster Tello Talent********");
    Serial.println(SDK_VERSION);
    Serial.println();

    RMTT_RGB::Init();
    RMTT_RGB::SetRGB(255, 0, 0);
    delay(1000);
    RMTT_RGB::SetRGB(0, 255, 0);
    delay(1000);
    RMTT_RGB::SetRGB(0, 0, 255);
    delay(1000);
    RMTT_RGB::SetRGB(0, 0, 0);
    //p_tt_gamesir = RMTT_GamesirT1d::GetInstance();
    Serial.println("1");
    shell_cmd_init();



    cmd_register((char*)"tof?", tof_callback);
    xTaskCreateUniversal(tof_battery_read_task, "tof_battery_read_task", 4096, NULL, 2, &tofBatteryReadTaskHandle, 1);

    cmd_register((char*)"led", led_callback);
    cmd_register((char*)"version?", version_callback);
    cmd_register((char*)"ETT", rmtt_callback);
    cmd_register((char*)"DIY", custom_callback);

    cmd_unknown_handle_register(unknown_cmd_callback);
    Serial.println("2");

}

void loop()
{
    /* Serial receive process from drone */
    while (CommonSerial.available())
    {
        int ret = cmd_process(CommonSerial.read());
        if (ret != 0)
        {
            CommonSerial.printf("command error: %d\r\n", ret);
        }
    }

#ifndef __UART0_DEBUG__
    if (Serial.available())
    {
        int i = 0;
        char buff[20] = {0};

        while (Serial.available() && (i < 20))
        {
            buff[i++] = Serial.read();
        }
        buff[19] = '\0';

        if (!strcmp(buff, "esp32version?"))
        {
            Serial.printf(SDK_VERSION);
        }
        else if (!strcmp(buff, "wifiversion?"))
        {
            Serial1.printf("[TELLO] wifiversion?");
        }
        else if (!strcmp(buff, "wifiupgrade"))
        {
            int upgrade_cnt = 0;
            while (1)
            {
                if (Serial.available())
                {
                    Serial1.write(Serial.read());
                    RMTT_RGB::SetGreen(255);
                }
                if (Serial1.available())
                {
                    Serial.write(Serial1.read());
                    RMTT_RGB::SetRed(255);
                }
                if (upgrade_cnt > 1000)
                {
                    RMTT_RGB::SetGreen(0);
                    RMTT_RGB::SetRed(0);
                    upgrade_cnt = 0;
                }
                upgrade_cnt++;
            }
        }
    }
#endif
    /* -------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------  */
    /* DO NOT ADD ANY CODE HERE FOR NOT BLOCKING THE RECEIVE FROM THE SERIAL */
    /*     YOU CAN ADD YOUR USER CODE TO THE 'user_task' FUNCTION ABOVE      */
    /* -------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------- */
}

/**
 * unknown command
 *
 *  @param argc Control argument
 *  @param argv[] Value argument 1
 *  @param argv2[] Value argument2
 */
int unknown_cmd_callback(int argc, char *argv[], char argv2[])
{
    Serial.printf("unknown cmd %s\r\n", argv[0]);
}

/*******************Matrix control part**************************/

/**
 * Show the graph pre-settled in the file onto the Matrix
 */
void matrix_show_graph_from_file()
{
    SPIFFS.begin(true);

    if(SPIFFS.exists("/graph_enable.txt"))
    {
        int i = 0;
        uint8_t graph_buff[128];
        File file = SPIFFS.open("/matrix_graph.txt", FILE_READ);
        if (file)
        {
            while (file.available() && (i < 128))
            {
                graph_buff[i++] = (char)file.read();
            }
            matrix_effect_set_graph(graph_buff);
        }
        file.close();
    }
}

/**
 * Handling the commands about the Matrix
 *
 * @param argc Control argument
 * @param argv[] Value argument 1
 * @param argv2[] Value argument2
 */


int move_param_is_valid(char *argv, char *color)
{
    if (!strcmp(argv, "l")) {}
    else if (!strcmp(argv, "r")) {}
    else if (!strcmp(argv, "u")) {}
    else if (!strcmp(argv, "d")) {}
    else
    {
        return -1;
    }

    if (!strcmp(color, "r")) {}
    else if (!strcmp(color, "b")) {}
    else if (!strcmp(color, "p")) {}
    else if (!strcmp(color, "g")) {}
    else
    {
        return -1;
    }

    return 0;
}

/*******************Drone control part*****************************/
int rmtt_int = 0;
bool rmtt_bool = false;

bool int_is_valid = false;
bool bool_is_valid = false;

bool mac_is_valid = false;
uint8_t rmtt_mac[6] = {0};

/**
 * Drone connection handling, measure the connection state by serial data
 * feedback from the drone
 *  - Update connection state by parsing data feeded back from inner serial1
 *
 *  @param argc Control argument
 *  @param argv[] Value argument 1
 *  @param argv2[] Value argument2
 */
int rmtt_callback(int argc, char *argv[], char argv2[])
{
    if (!strcmp(argv[1], "ok"))
    {
        bool_is_valid = true;
        rmtt_bool = true;
    }
    else if (!strcmp(argv[1], "error"))
    {
        bool_is_valid = true;
        rmtt_bool = false;
    }
    else if (!strncmp(argv[1], "wifiv", 5))
    {
        /* report wifi version to PC */
        Serial.printf(argv[1]);
    }
    else if (!strcmp(argv[1], "mac"))
    {
        if ((sscanf(argv[2], "%02x%02x%02x%02x%02x%02x",
                    &rmtt_mac[0], &rmtt_mac[1], &rmtt_mac[2],
                    &rmtt_mac[3], &rmtt_mac[4], &rmtt_mac[5]) == 6)
         &&(argc == 3))
        {
            mac_is_valid = true;
#ifdef __DEFAULT_LOG__
            Serial.println("rmtt_callback(): mac get ok");
#endif
        }
        else
        {
#ifdef __DEFAULT_LOG__
            Serial.println("rmtt_callback(): mac get error");
#endif
        }

    }
    else if (sscanf(argv[1], "%d", &rmtt_int) && (argc == 2))
    {
        int_is_valid = true;
    }
    return 0;
}

/**
 * Get MAC valid state of the joystick
 *
 * @return the valid state of the MAC
 */
bool rmtt_joystick_mac_is_valid()
{
    return mac_is_valid;
}

/**
 * Get MAC of the joystick
 *
 * @return MAC of the joystick
 */
uint8_t *get_rmtt_joystick_mac()
{
    mac_is_valid = false;
    return rmtt_mac;
}

/**
 * Get valid state of whether having an Int type data received
 * from the drone
 *
 * @return valid state
 */
bool rmtt_int_is_valid()
{
    return int_is_valid;
}

/**
 * Get valid state of whether having an Boolean type data received
 * from the drone
 *
 * @return valid state
 */
bool rmtt_bool_is_valid()
{
    return bool_is_valid;
}

/**
 * Get the already received Int type data
 *
 * @return Int data from the drone
 */
int get_rmtt_int()
{
    int_is_valid = false;
    return rmtt_int;
}

/**
 * Get the already received Boolean type data
 *
 * @return Boolean data from the drone
 */
bool get_rmtt_bool()
{
    bool_is_valid = false;
    return rmtt_bool;
}

int version_callback(int argc, char *argv[], char argv2[])
{
    CommonSerial.printf("version %s", SDK_VERSION);
}

/**
 * Led command handling, control the color of the LED by input commands
 *
 *  @param argc Control argument
 *  @param argv[] Value argument 1
 *  @param argv2[] Value argument2
 */
int led_callback(int argc, char *argv[], char argv2[])
{
    int r1,b1,g1,r2,b2,g2;
    if (!strcmp(argv[1], "bl"))
    {
        float blink_freq = 1;
        if ((argc == 9)
          &&sscanf(argv[2], "%f", &blink_freq)
          &&sscanf(argv[3], "%d", &r1)
          &&sscanf(argv[4], "%d", &g1)
          &&sscanf(argv[5], "%d", &b1)
          &&sscanf(argv[6], "%d", &r2)
          &&sscanf(argv[7], "%d", &g2)
          &&sscanf(argv[8], "%d", &b2))
        {
            if ((blink_freq >= 0.09) && (blink_freq <= 10.1))
            {
                led_effect_blink(r1, g1, b1, r2, g2, b2, blink_freq);
                CommonSerial.print("led ok");
            }
            else
            {
                goto end;
            }

        }
        else
        {
            goto end;
        }

    }
    else if (!strcmp(argv[1], "br"))
    {
        float breath_freq = 1;
        if ((argc == 6)
          &&sscanf(argv[2], "%f", &breath_freq)
          &&sscanf(argv[3], "%d", &r1)
          &&sscanf(argv[4], "%d", &g1)
          &&sscanf(argv[5], "%d", &b1))
        {
            if ((breath_freq >= 0.09) && (breath_freq <= 2.51))
            {
                led_effect_breath(r1, g1, b1, breath_freq);
                CommonSerial.print("led ok");
            }
            else
            {
                goto end;
            }
        }
        else
        {
            goto end;
        }
    }
    else if (argc == 4)
    {
        if (sscanf(argv[1], "%d", &r1)
          &&sscanf(argv[2], "%d", &g1)
          &&sscanf(argv[3], "%d", &b1))
        {
            led_effect_set_rgb(r1, g1, b1);
            CommonSerial.print("led ok");
        }
        else
        {
            goto end;
        }
    }
    else
    {
        goto end;
    }

    return 0;
end:
    CommonSerial.print("led error");
    return 0;
}

/**
 * ToF measuring command handling
 *
 * @param arg Parameter about task control
 */
void tof_battery_read_task(void *arg)
{
    
    for (;;)
    {
      tof_range1 = sensors[0].read();
      tof_range2 = sensors[1].read();
      tof_range3 = sensors[2].read();

      if (mySerial.available())                
        {
        
            if (mySerial.read() == HEADER)       
            {
            uart[0] = HEADER;
                if (mySerial.read() == HEADER)     
                {
                    uart[1] = HEADER;
                    for (i = 2; i < 9; i++)    
                    {
                    uart[i] = mySerial.read();
                    }
                    check = uart[0] + uart[1] + uart[2] + uart[3] + uart[4] + uart[5] + uart[6] + uart[7];
                    if (uart[8] == (check & 0xff))  
                    {
                    dist = uart[2] + uart[3] * 256;  
                    strength = uart[4] + uart[5] * 256; 
                    Serial.print("distance = ");
                    Serial.print(dist);               
                    Serial.print('\t');
                    Serial.print("strength = ");
                    Serial.print(strength);          
                    Serial.print('\n');
                    }
                }
            }
        }

      Serial.printf("tof tof1: %d tof2: %d tof3: %d mini LiDAR: %d LiDAR strength: %d\n", tof_range1, tof_range2, tof_range3, dist, strength);;
      delay(100);
    }
}

/**
 * Gamesir joystick pairing process handling
 *  - Detect the press event of the pairing button
 *  - Pair the joystick with the TT Plugin Module
 *
 * @param arg Parameter about task control
 */
void gamesir_pairing_task(void *arg)
{
    int __key_cnt = 0;
    for (;;)
    {
        if (digitalRead(34) == 0)
        {
            __key_cnt++;
        }
        else
        {
            __key_cnt = 0;
        }

        if (__key_cnt >= 20)
        {
            if (!p_tt_gamesir->GetConnectedStatus())
            {
                pair_mode = true;
                p_tt_gamesir->SetMACFilterEnable(false);
            }
        }

        delay(100);
    }
}

#define TAKEOFF_TIMEOUT 200

int takeoff_status = 0;

int now_time  = 0;
int last_clean_time  = 0;

/**
 * Gamesir joystick control handling
 *  - Receive command from the joystick
 *  - Control the drone by received command
 *
 * @param arg Parameter about task control
 */
void gamesir_task(void *arg)
{
    uint8_t command_init = 0;
    uint8_t mac_init = 0;

    uint8_t stop_cnt = 5;

    for (;;)
    {
        if (mac_init == 0)
        {
            CommonSerial.print("[TELLO] getmac?");
            delay(100);
#ifdef __DEFAULT_LOG__
            Serial.println("gamesir_task(): mac is ok?");
#endif
            if (rmtt_joystick_mac_is_valid())
            {
#ifdef __DEFAULT_LOG__
                Serial.println("gamesir_task(): ble mac init");
#endif
                p_tt_gamesir->Init(get_rmtt_joystick_mac());
                mac_init = 1;
            }
        }
        else if ((command_init == 0) && (p_tt_gamesir->GetConnectedStatus()))
        {
            tt_sdk.SDKOn();
            delay(100);
            if (rmtt_bool_is_valid())
            {
                command_init = 1;
            }
        }
        else if (p_tt_gamesir->DataIsValid())
        {
            // Serial.println("data is update");
            PlainData data = p_tt_gamesir->GetData();

            int lx = ((float)data.left_x_3d - 512) / 512.0 * 100;
            int ly = ((float)data.left_y_3d - 512) / 512.0 * 100;
            int rx = ((float)data.right_x_3d - 512) / 512.0 * 100;
            int ry = ((float)data.right_y_3d - 512) / 512.0 * 100;

            if ((data.btn3 == 0x01) && (data.L2))
            {
                tt_sdk.Flip('f');
            }
            else if ((data.btn3 == 0x03) && (data.L2))
            {
                tt_sdk.Flip('r');
            }
            else if ((data.btn3 == 0x05) && (data.L2))
            {
                tt_sdk.Flip('b');
            }
            else if ((data.btn3 == 0x07) && (data.L2))
            {
                tt_sdk.Flip('l');
            }
            else if ((data.Y) && (data.R2))
            {
                if (takeoff_status == 0)
                {
                    tt_sdk.TakeOff();
                    takeoff_status = 1;
                }
                else
                {
                    tt_sdk.Land();
                    takeoff_status = 0;
                }
            }
            else
            {
#ifdef BLE_JAPAN_CTRL
                tt_sdk.SetRC(lx, -ly, -ry, rx);
#else
                tt_sdk.SetRC(rx, -ry, -ly, lx);
#endif
            }
        }

        /* 避免rc指令粘包 */
        delay(10);

        /* Regularly send data packet to ensure the drone
        floating steadily after the controller was offline*/
        if ((now_time - last_clean_time > 300) && command_init)
        {
            if (p_tt_gamesir->GetDataOffline())
            {
                if (stop_cnt)
                {
                    tt_sdk.SetRC(0, 0, 0, 0);
                    stop_cnt--;
                }
            }
            else
            {
                /* keepactive */
                CommonSerial.print("[TELLO] keepalive");
                stop_cnt = 5;
            }
            last_clean_time = millis();
        }
        else
        {

        }

        now_time = millis();
    }
}

/**
 * Gamesir joystick Bluetooth(BLE) connection handling
 *
 * @param arg Parameter about task control
 */
void ble_status_task(void *arg)
{
    static int __led_cnt = 0;
    static uint8_t toggle = 0;
    uint8_t ble_mac[6] = {0};

    while (1)
    {
        if (get_led_effect_mode() == LED_EFFECT_FACTORY_MODE)
        {
            if (p_tt_gamesir->GetConnectedStatus())
            {
                if (pair_mode == true)
                {
                    memcpy(ble_mac, p_tt_gamesir->GetMAC(), 6);
                    p_tt_gamesir->SetMACFilterEnable(true);
                    CommonSerial.printf("[TELLO] setmac %02x%02x%02x%02x%02x%02x",
                                        ble_mac[0], ble_mac[1], ble_mac[2],
                                        ble_mac[3], ble_mac[4], ble_mac[5]);
                    delay(50);
                    if (rmtt_bool_is_valid())
                    {
#ifdef __DEFAULT_LOG__
                        Serial.println("ble_status_task(): peer is successful");
#endif
                        if (get_rmtt_bool())
                        {
                            pair_mode = false;
                        }
                    }
                }
                RMTT_RGB::SetBlue(255);
            }
            else if (pair_mode == true)
            {
                if (__led_cnt % 4 == 0)
                {
                    toggle = ~toggle;
                }

                if (!toggle)
                {
                    RMTT_RGB::SetBlue(0);
                }
                else
                {
                    RMTT_RGB::SetBlue(255);
                }
            }
            else
            {
                RMTT_RGB::SetBlue(0);
            }
        }

        __led_cnt++;
        delay(100);
    }
}

void wifi_upgrade()
{
    int cnt = 0;
    // put your setup code here, to run once:
    Serial.begin(921600);
    Serial1.begin(1000000, 23, 18, SERIAL_8N1);
    RMTT_RGB::Init();
    RMTT_RGB::SetRGB(0, 255, 0);
    delay(500);
    while (1)
    {
        if (Serial.available())
        {
            Serial1.write(Serial.read());
            RMTT_RGB::SetGreen(255);
        }
        if (Serial1.available())
        {
            Serial.write(Serial1.read());
            RMTT_RGB::SetRed(255);
        }
        if (cnt > 1000)
        {
            RMTT_RGB::SetGreen(0);
            RMTT_RGB::SetRed(0);
            cnt = 0;
        }
        cnt++;
    }
}


/**
 * DIY command callback
 *
 *  @param argc Control argument
 *  @param argv[] Value argument 1
 *  @param argv2[] Value argument2
 */
int custom_callback(int argc, char *argv[], char argv2[])
{
    // you can do what you want after receive diy command.
}
