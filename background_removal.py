#!/usr/bin/env python3
import depthai as dai
import numpy as np
import open3d as o3d
	
class DepthAIBackgroundRemoval():
	def __init__(self, device):
		self.pointcloud_all	=	o3d.geometry.PointCloud()
		self.pointcloud_objects	=	o3d.geometry.PointCloud()
		self.pointcloud_room	=	o3d.geometry.PointCloud()
		self.device	=	device;
		
		self.depth_queue	=	device.getOutputQueue("depth", 1, blocking=False)
		
		self.width = None
		self.height = None
		
		self.camera_parameters	=	o3d.camera.PinholeCameraParameters()
		self.camera_parameters_voxel	=	o3d.camera.PinholeCameraParameters()
		
		angle = np.pi		
		self.camera_parameters.extrinsic = np.array([
			[np.cos(angle)  , -np.sin(angle)    , 0, 0],
            [np.sin(angle)  , np.cos(angle)     , 0, 0],
            [0              , 0                 , 1, 0],
			[0,0,0,1]
		])
		
		
		self.min_depth_meters	=	0
		self.max_depth_meters	=	5
		
		self.number_of_frames_to_check = 7
		self.max_depth_variance_per_frame = 0.01
		
		self.number_of_object_frames_to_check = 2
		self.wall_distance = 0.075
		
		
		
		self.previous_depth_maps = []
		self.previous_object_depth_maps = []
		
		self.visualizer = None
		self.frame = 0
		
		self.max_depth_map = None
		
	def generate_intrinsics(self, width = None, height = None):
		if width is None:
			width = self.width
		if height is None:
			height = self.height
		
		device_calibration_data = self.device.readCalibration()
		intrinsic_matrix = 	np.array(device_calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, width, height))
		
		return  o3d.camera.PinholeCameraIntrinsic(width,height,intrinsic_matrix[0][0],intrinsic_matrix[1][1],intrinsic_matrix[0][2],intrinsic_matrix[1][2])
	
	def average_depth_map(self, depth_map_list, max_variance):
		number_of_frames = len(depth_map_list)
	
		depth_average = np.mean(depth_map_list, axis=0)
		np_depth_average = np.asarray([depth_average] * number_of_frames)
		np_depth_maps = np.asarray(depth_map_list)
		
		np_difference = np.amax(np.absolute(np.subtract(np_depth_average, np_depth_maps)), axis=0)
		
		np_difference[np_difference < max_variance] = 0
		np_difference[np_difference > max_variance] = 1
		
		np_difference = 1 - np_difference
		
		depth_map_output = np.multiply(np_difference, depth_average)
		return depth_map_output
		
		
	
	def update_pointcloud(self, ):
		if self.depth_queue == None:
			return False
		data = self.depth_queue.tryGet()
		if data == None:
			return False
		
		data_width	=	data.getWidth();
		data_height	=	data.getHeight();
		depth_map	=	np.ascontiguousarray(data.getFrame()).astype(np.float32)
		
		if (self.width != data_width) or (self.height != data_height):
			self.width = data_width
			self.height = data_height
			self.camera_parameters.intrinsic = self.generate_intrinsics()
			self.max_depth_map = np.full(depth_map.shape, self.min_depth_meters, depth_map.dtype)
			self.previous_depth_maps = []
			self.previous_object_depth_maps = []
		
		depth_map /= 1000.0
		
		depth_map[depth_map > self.max_depth_meters] = self.min_depth_meters
		depth_map[depth_map < self.min_depth_meters] = self.min_depth_meters
		
		self.previous_depth_maps.append(depth_map)
		if len(self.previous_depth_maps) > self.number_of_frames_to_check:
			del self.previous_depth_maps[0]
			depth_map_average =  self.average_depth_map(self.previous_depth_maps, self.max_depth_variance_per_frame)
			self.max_depth_map = np.maximum(depth_map_average, self.max_depth_map)
			
		
		
		
		
		
		depth_map_objects = np.subtract(self.max_depth_map, depth_map)
		depth_map_objects[depth_map_objects < self.wall_distance] = 0
		depth_map_objects[depth_map_objects > 0] = 1
		depth_map_objects = np.multiply(depth_map_objects, depth_map)
		
		
		self.previous_object_depth_maps.append(depth_map_objects)
		if len(self.previous_object_depth_maps) > self.number_of_object_frames_to_check:
			del self.previous_object_depth_maps[0]
			depth_map_objects =  self.average_depth_map(self.previous_object_depth_maps, self.max_depth_variance_per_frame)
		else:
			depth_map_objects = 0 * depth_map_objects
		
		
		
		
		
		
		depth_image = o3d.geometry.Image(depth_map)
		depth_image_objects = o3d.geometry.Image(depth_map_objects)
		depth_image_room = o3d.geometry.Image(self.max_depth_map)
		
		
		
		
		pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
			depth=depth_image,
			intrinsic=self.camera_parameters.intrinsic,
			extrinsic=self.camera_parameters.extrinsic,
			depth_scale=1,
			depth_trunc=self.max_depth_meters,
			stride=1)
		
		pointcloud_objects = o3d.geometry.PointCloud.create_from_depth_image(
			depth=depth_image_objects,
			intrinsic=self.camera_parameters.intrinsic,
			extrinsic=self.camera_parameters.extrinsic,
			depth_scale=1,
			depth_trunc=self.max_depth_meters,
			stride=1)
		pointcloud_room = o3d.geometry.PointCloud.create_from_depth_image(
			depth=depth_image_room,
			intrinsic=self.camera_parameters.intrinsic,
			extrinsic=self.camera_parameters.extrinsic,
			depth_scale=1,
			depth_trunc=self.max_depth_meters,
			stride=1)
		
		self.pointcloud_objects.points = pointcloud_objects.points
		self.pointcloud_room.points = pointcloud_room.points
		self.pointcloud_all.points = pointcloud.points
		
		self.pointcloud_all.paint_uniform_color([1,0,0])
		self.pointcloud_objects.paint_uniform_color([0,0,1])
		self.pointcloud_room.paint_uniform_color([0,1,0])
		
		self.frame += 1
		if self.frame > 20:
			self.visualize()
		
		return True
		
	def visualize(self):		
		if self.visualizer is None:
			self.visualizer = o3d.visualization.Visualizer()
			self.visualizer.create_window()
			self.visualizer.add_geometry(self.pointcloud_objects)
			self.visualizer.add_geometry(self.pointcloud_room)
			# self.visualizer.add_geometry(self.pointcloud_all)
			origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
			self.visualizer.add_geometry(origin)
			self.is_visualizer_started = True
		else:
			self.visualizer.update_geometry(self.pointcloud_objects)
			self.visualizer.update_geometry(self.pointcloud_room)
			# self.visualizer.update_geometry(self.pointcloud_all)
			self.visualizer.poll_events()
			self.visualizer.update_renderer()

	def close_window(self):
		self.visualizer.destroy_window()
