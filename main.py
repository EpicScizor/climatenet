import os
import json
import pathlib
import matplotlib.pyplot as plt
import gdal
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import numpy as np
import sys

import globalvars as g
from data_generator import DataGenerator
from model import get_model
from included_vars import data_vars, vars_to_plot, operators

def get_band_identifier(band_data):
	desc = band_data.GetDescription()
	metadata = band_data.GetMetadata()     
	d = str(metadata['GRIB_ELEMENT']) + " -- "
	d += str(metadata['GRIB_COMMENT']) + " -- "
	d += str(desc)
	return d
def print_band_identifier(ttl, data = None, used = True):
	if used and g.PRINT_USED:
		print(ttl)
		if g.SHOULD_DISPLAY_BAND_STATS:
				print("MAX: " + str(np.max(data)))
				print("MIN: " + str(np.min(data)))
	elif not used and g.PRINT_UNUSED:
		print(ttl + " # UNUSED")
		if g.SHOULD_DISPLAY_BAND_STATS:
				print("MAX: " + str(np.max(data)))
				print("MIN: " + str(np.min(data)))
	
def get_input_dimensions(nparrays):
	size = g.RADIUS * 2 + 1
	return size * size * len(nparrays) + 4 # x, y, day, time
def get_output_dimensions(nparrays):
	return len(nparrays)


# Initialize horovod
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

if hvd.rank() == 0:
	print('Python version: %s' % sys.version)
	print('TensorFlow version: %s' % tf.__version__)
	print('Keras version: %s' % tf.keras.__version__)

# Loop over all data files
path = os.path.join(pathlib.Path(__file__).parent.absolute(), "data")
finished_files_count = 0
data_by_days = []
for ff in os.listdir(path):
	if g.JUST_ONE_FILE and finished_files_count > 0: break;
	print("Rank " + hvd.rank() + " opening: " + ff)
	file_path = os.path.join(path, ff)

	# Open the file
	grib = gdal.Open(file_path)

	present_data_vars = []

	# Loop over all data fields of a grib file and load those requested to present_data_fields
	for a in range(grib.RasterCount):
		a += 1
		# Read an specific band
		band = grib.GetRasterBand(a)
		ttl = get_band_identifier(band)
		# Read the band as a Python array
		data = band.ReadAsArray()
		if ttl in data_vars:
			print_band_identifier(ttl, data = data, used = True)
		else:
			print_band_identifier(ttl, data = data, used = False)
		
		# Show the image
		if g.SHOULD_PLOT and ttl in vars_to_plot:
			plt.imshow(data, cmap='jet')
			plt.title(ttl)
			plt.show()

		# Add data from this layer to data fields
		if ttl in data_vars:
			# transform data
			if ttl in operators:
				op = operators[ttl]
				data *= op[1]
				data += op[0]
			else:
				raise SystemError("MISSING OPERATOR FOR: " + str(ttl))
			present_data_vars.append((ttl, data))
	
	# Verify that all requested data fields are present and that we don't have any excess fields either
	requested_data_vars = data_vars.copy()
	for a in present_data_vars:
		ttl = a[0]
		data = a[1]
		if ttl in requested_data_vars:
			requested_data_vars.remove(ttl)
		else:
			raise SystemError("PRESENT_DATA_VARS HAS AN ENTRY THAT WASN'T REQUESTED OR THERE IS A DUPLICATE!")
	if len(requested_data_vars) > 0:
		raise SystemError("NOT ALL REQUESTED FIELDS WERE PRESENT! MISSING: " + str(requested_data_vars))
	# Sort present_data_vars by ttl and the order in data_vars
	grib_data = []
	for i in range(len(data_vars)):
		ttl = data_vars[i]
		for a in present_data_vars:
			if a[0] == ttl:
				grib_data.append(a[1])
				continue
	data_by_days.append(grib_data)
	finished_files_count += 1
	print("--- complete ---")

g.GLOBAL_MAP_DIMENSIONS = data_by_days[0][0].shape
g.INPUT_SIZE = get_input_dimensions(data_by_days[0])
g.OUTPUT_SIZE = get_output_dimensions(data_by_days[0])

if hvd.rank() == 0:
	print("GLOBAL MAP DIMENSIONS: " + str(g.GLOBAL_MAP_DIMENSIONS))
	print("INPUT SIZE: " + str(g.INPUT_SIZE))
	print("OUTPUT SIZE: " + str(g.OUTPUT_SIZE))

# Model goes here
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(g.INPUT_SIZE, activation='relu', input_dim=g.INPUT_SIZE),
	tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
	tf.keras.layers.Dropout(rate=0.15),
	tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
	tf.keras.layers.Dropout(rate=0.15),
	tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
	tf.keras.layers.Dropout(rate=0.15),
	tf.keras.layers.Dense(g.OUTPUT_SIZE, activation='relu')
])

opt = tf.keras.optimizers.Adadelta(learning_rate=0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
			  loss='mean_squared_error',
			  experimental_run_tf_function=False,
			  metrics=['mean_absolute_error', 
					   'mean_squared_logarithmic_error',
					   'mean_squared_error',
					   'logcosh'])		   
					   
callbacks = [
	# Horovod: broadcast initial variable states from rank 0 to all other processes.
	# This is necessary to ensure consistent initialization of all workers when
	# training is started with random weights or restored from a checkpoint.
	hvd.callbacks.BroadcastGlobalVariablesCallback(0),

	# Horovod: average metrics among workers at the end of every epoch.
	#
	# Note: This callback must be in the list before the ReduceLROnPlateau,
	# TensorBoard or other metrics-based callbacks.
	hvd.callbacks.MetricAverageCallback(),

	# Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
	# accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
	# the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
	hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
]
if hvd.rank() == 0:
	callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
	
verbose = g.VERBOSITY if hvd.rank() == 0 else 0

# NOTE !!! EVEN THO BELOW WE USE A WORD "DAY" WE REALLY MEAN "TICK"
if len(data_by_days) >= 2:
	generator = DataGenerator(
		data_by_days, 
		batch_size=g.BATCH_SIZE, 
		len_multiplier=g.EPOCH_LENGHT_MULTIPLIER)
	validation_generator = DataGenerator(
		data_by_days, 
		batch_size=g.BATCH_SIZE,
		len_multiplier=g.VALIDATION_LENGTH_MULTIPLIER)
	if hvd.rank == 0:
		print("Generator len: " + str(len(generator)))
		print(model.summary())
	
	epochs_count = g.EPOCHS
	history = model.fit(generator,
		epochs=epochs_count,
		verbose=verbose,
		callbacks=callbacks,
		validation_data=validation_generator)
	if hvd.rank() == 0:
		# LOG STUFF
		print(history.history.keys())
		#  "Accuracy"
		plt.plot(history.history['mean_absolute_error'])
		plt.plot(history.history['val_mean_absolute_error'])
		plt.title('mean_absolute_error')
		plt.ylabel('mean_absolute_error')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.show()
		# "Loss"
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.show()


		# SAVE MODEL
		if g.SHOULD_SAVE_MODEL:
			dirname = os.path.dirname(__file__)
			filename = os.path.join(dirname, 'models')
			tf.keras.models.save_model(model, filename, overwrite=True)
else:
	if hvd.rank() == 0:
		print("NOT ENOUGH GRIB FILES FOR ACTUAL LEARNING!")


