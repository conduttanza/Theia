#imports
import pigpio
import time

# --------------------
# SETUP
# --------------------
GPIO_SERVO = 17

# SG90 safe pulse range (microseconds)
MIN_PW = 500     # ~0
MID_PW = 1500    # ~90
MAX_PW = 2400    # ~180

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
		pulse = MAX_PW + (angle / 180) * (MAX_PW - MIN_PW)
		pi.set_servo_pulsewidth(GPIO_SERVO, pulse)
		
	def moveUp(self,angleChange):
		if angleChange == True:
			self.angle += 0.5
			self.set_angle(self.angle)
			
			if self.angle > 90:
				pass
			time.sleep(delay)
			return
		else:
			pass
			
	def moveDown(self,angleChange):
		if angleChange == True:
			self.angle -= 0.5
			self.set_angle(self.angle)
			
			if self.angle < 35:
				pass
			time.sleep(delay)
			return
		else:
			pass
	
	def stop(self):
		"""Stop PWM signal (servo relaxes)"""
		pi.set_servo_pulsewidth(GPIO_SERVO, 0)

	def cleanup(self):
		"""Clean shutdown"""
		self.stop()
		pi.stop()	
