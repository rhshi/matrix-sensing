import os


class Logger(object):
	def __init__(self, log_file):
		self.log_file = "logs/{}.txt".format(log_file)
		os.mknod(self.log_file)

	def log(self, string):
		with open(self.log_file, "a") as f:
			f.write(string)
			f.write("\n")

		print(string)

		return

