#code by conduttanza
#
#created the 19/01/2026

#imports
import pigpio
import time

#local imports
from mediapipe_logic import Hands_Reckon
reck = Hands_Reckon()
# --------------------
# SETUP
# --------------------
GPIO_SERVO = 17

# SG90 safe pulse range (microseconds)
MIN_PW = 500     # ~0
MID_PW = 1500    # ~90
MAX_PW = 2000    # ~180

pi = pigpio.pi()

if not pi.connected:
	exit("pigpio daemon not running")

pi.set_mode(GPIO_SERVO, pigpio.OUTPUT)

delay = 0

class GPIO():
	def __init__(self):
		self.angle = 90
	
	def set_angle(self, angle):
		"""Move servo to angle 180"""
		angle = max(0, min(180, angle))
		pulse = MID_PW + (angle / 180) * (MAX_PW - MIN_PW)
		pi.set_servo_pulsewidth(GPIO_SERVO, pulse)
		
	def moveUp(self):
		self.angle += reck.returnTrackerSpeed()
		self.set_angle(self.angle)
		if self.angle > 90:
			pass
			time.sleep(delay)
			return
			
	def moveDown(self):
		self.angle -= reck.returnTrackerSpeed()
		self.set_angle(self.angle)
		if self.angle < 35:
			pass
			time.sleep(delay)
			return
	
	def stop(self):
		"""Stop PWM signal (servo relaxes)"""
		pi.set_servo_pulsewidth(GPIO_SERVO, 0)

	def cleanup(self):
		"""Clean shutdown"""
		self.stop()
		pi.stop()	
