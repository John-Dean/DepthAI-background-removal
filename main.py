import depthai as dai
import numpy as np
import time
import contextlib
import dai_pipeline

if __name__ == "__main__":
	
	pipeline = dai_pipeline.create_pipeline()

	with contextlib.ExitStack() as stack:
		dai_available_devices = dai.Device.getAllAvailableDevices()
		if len(dai_available_devices) == 0:
			raise RuntimeError("No devices found!")
		else:
			print("Found", len(dai_available_devices), "devices")

		devices	=	[];
		for available_device in dai_available_devices:
			device = stack.enter_context(dai.Device(pipeline, available_device))
			devices.append(dai_pipeline.setup_device(device))
		
		print("Running")
		fps_limit = 90;
		frame = 0
		while True:
			start_time = time.time()
			
			for i in range(len(devices)):
				device = devices[i]
				device.update_pointcloud()
			
			end_time	=	time.time();
			# print( str(1/(end_time - start_time)) + "fps")
			time.sleep(max(1./fps_limit - (end_time- start_time), 0))
			frame += 1
