import depthai as dai
import numpy as np
from background_removal import DepthAIBackgroundRemoval

def create_pipeline():
	# Start defining a pipeline
	pipeline = dai.Pipeline()

	pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

	# color = pipeline.create(dai.node.ColorCamera)
	# color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	# # Color cam: 1920x1080
	# # Mono cam: 640x400
	# color.setIspScale(2,3) # To match 400P mono cameras
	# color.setBoardSocket(dai.CameraBoardSocket.RGB)
	# color.initialControl.setManualFocus(130)

	# # For deeplabv3
	# color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
	# color.setPreviewSize(nn_shape, nn_shape)
	# color.setInterleaved(False)

	# NN output linked to XLinkOut
	# isp_xout = pipeline.createXLinkOut()
	# isp_xout.setStreamName("cam")
	# cam.isp.link(isp_xout.input)

	# # Define a neural network that will make predictions based on the source frames
	# detection_nn = pipeline.createNeuralNetwork()
	# detection_nn.setBlobPath(nn_path)
	# detection_nn.input.setBlocking(False)
	# detection_nn.setNumInferenceThreads(2)
	# color.preview.link(detection_nn.input)

	# # NN output linked to XLinkOut
	# xout_nn = pipeline.create(dai.node.XLinkOut)
	# xout_nn.setStreamName("nn")
	# detection_nn.out.link(xout_nn.input)

	# Left mono camera
	left = pipeline.create(dai.node.MonoCamera)
	left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	left.setBoardSocket(dai.CameraBoardSocket.LEFT)
	# Right mono camera
	right = pipeline.create(dai.node.MonoCamera)
	right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

	# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
	depth = pipeline.create(dai.node.StereoDepth)
	
	# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
	depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
	depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
	depth.setSubpixel(True)
	depth.setLeftRightCheck(True)
	depth.setExtendedDisparity(False)
	config = depth.initialConfig.get()
	# config.postProcessing.speckleFilter.enable = False
	# config.postProcessing.speckleFilter.speckleRange = 50
	# config.postProcessing.temporalFilter.enable = True
	config.postProcessing.spatialFilter.enable = True
	config.postProcessing.spatialFilter.holeFillingRadius = 5
	config.postProcessing.spatialFilter.numIterations = 1
	# print(config.postProcessing.spatialFilter.delta)
	# config.postProcessing.spatialFilter.alpha = 0.25
	# config.postProcessing.spatialFilter.delta = 15
	config.postProcessing.thresholdFilter.minRange = 400
	config.postProcessing.thresholdFilter.maxRange = 15000
	config.postProcessing.decimationFilter.decimationFactor = 2

	depth.initialConfig.set(config)
	depth.initialConfig.setConfidenceThreshold(50)

	
	# stereo.initialConfig.setBilateralFilterSigma(64000)
	depth.setDepthAlign(dai.CameraBoardSocket.RGB)
	left.out.link(depth.left)
	right.out.link(depth.right)




	# Create depth output
	xout_depth = pipeline.create(dai.node.XLinkOut)
	xout_depth.setStreamName("depth")
	depth.depth.link(xout_depth.input)
	
	return pipeline

def setup_device(device):
	output = DepthAIBackgroundRemoval(device)
	
	return output
