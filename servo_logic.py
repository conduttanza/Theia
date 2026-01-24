#code by conduttanza
#
#created the 19/01/2026

#imports
import pigpio
import time
from threading import Thread

# --------------------
# SETUP
# --------------------
GPIO_SERVO = 17

# SG90 safe pulse range (microseconds)
MIN_PW = 500 # ~0
MID_PW = 1500 # ~90
MAX_PW = 2000 # ~180

pi = pigpio.pi()

if not pi.connected:
	exit("pigpio daemon not running")

pi.set_mode(GPIO_SERVO, pigpio.OUTPUT)

delay = 0

class GPIO():
	def __init__(self):
		self.angle = 90
		self.reck = None
		self.listToKnowWhatToMove = []
		Thread(target=self.update, daemon=True).start()
	
	def update(self):
		if self.reck is None:
			from mediapipe_logic import Hands_Reckon
			self.reck = Hands_Reckon()
		self.listToKnowWhatToMove = self.reck.returnTrackerSpeed()
		if range(self.listToKnowWhatToMove) > 1:
			if self.listToKnowWhatToMove[1] == 'up':
				self.moveDown()
			if self.listToKnowWhatToMove[1] == 'down':
				self.moveUp()
		else:
			pass
		
	def set_angle(self, angle):
		"""Move servo to angle 180"""
		angle = max(0, min(180, angle))
		pulse = MID_PW + (angle / 180) * (MAX_PW - MIN_PW)
		pi.set_servo_pulsewidth(GPIO_SERVO, pulse)
		
	def moveUp(self):
		self.angle += self.listToKnowWhatToMove[0]
		self.set_angle(self.angle)
		if self.angle > 90:
			pass
			time.sleep(delay)
			return
			
	def moveDown(self):
		self.angle -= self.listToKnowWhatToMove[0]
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
