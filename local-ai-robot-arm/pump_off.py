from pymycobot.mycobot280 import MyCobot280
import time, sys

mc = MyCobot280("/dev/tty.usbserial-5AE20107941", 115200)
time.sleep(1)

if "--on" in sys.argv:
    mc.set_basic_output(5, 0)
    mc.set_basic_output(2, 0)
    print("Pump ON")
else:
    mc.set_basic_output(2, 1)
    mc.set_basic_output(5, 1)
    print("Pump OFF")
