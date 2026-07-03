from pymycobot.mypalletizer260 import MyPalletizer260
import time
import RPi.GPIO as GPIO
# PI版本
mc = MyPalletizer260("/dev/ttyAMA0", 1000000)
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)


def pump_on():
    GPIO.output(20, 0)
    GPIO.output(21, 0)
# 停止吸泵 m5
def pump_off():
    GPIO.output(20, 1)
    GPIO.output(21, 1)