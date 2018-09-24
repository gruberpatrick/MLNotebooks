
import numpy as np
from PIL import Image, ImageDraw
import math
import h5py
import json
import matplotlib.pyplot as plt
import copy
import math

# RELATIONAL QUESTION EXAMPLES:
# - What is the shape of the object that is furthest from the gray object?      TYPE=0
# - How many objects have the shape of the orange object?						TYPE=1
# - What is the color of the object that is closest to the blue object?			TYPE=2

# TO TRY:
# - How many objects have the size of the orange object?						TYPE=3

shapes = ["square", "circle"]#, "triangle"]
colors = ["gray", "green", "red", "blue", "yellow"]
sizes = ["small", "big"]

counter = {
	'right':0, 
	'bottom':0, 
	'circle':0, 
	'green':0, 
	'square':0, 
	'blue':0, 
	'gray':0,
	'yellow':0, 
	'left':0, 
	'top':0, 
	'red':0
}

sizes_widths = [12, 25]
colors_encodings = [
	(86, 86, 86),
	(70, 234, 68),
	(219, 41, 71),
	(50, 39, 216),
	(213, 219, 30),
	(255, 124, 233)
]

# ------------------------------------------------------------------------------
def getCenter(obj, loc):

	return (loc[0] + (sizes_widths[obj[2]] / 2), loc[1] + (sizes_widths[obj[2]] / 2))

# ------------------------------------------------------------------------------
def getDistance(obj1, obj2):

	return math.sqrt(math.pow(obj2[0] - obj1[0], 2) + math.pow(obj2[1] - obj1[1], 2))

# ------------------------------------------------------------------------------
def generateQuestion(objects, locations, data_size):

	global counter

	ret = np.random.randint(0, 2)

	idx = np.random.randint(0, len(locations))
	obj = objects[idx]
	cent = getCenter(obj, locations[idx])
	qtype = np.random.randint(0, 2)
	nrtype = np.random.randint(0, 3)
	question = ""
	nrquestion = ""
	answer = ""
	nranswer = ""

	# get color type;
	color_out = [0 for it in range(len(colors))]
	color_out[obj[0]] = 1
	# get question type relational;
	qtype_out = [0, 0, 0, 1, 0]
	qtype_out[qtype] = 1
	# get question type non-relational;
	nrqtype_out = [0, 0, 0, 0, 1]
	nrqtype_out[nrtype] = 1

	# simple shape question;
	'''if qtype == 2:
		question = "How many objects have the shape of the " + colors[obj[0]] + " object?"
		count = 0
		for it in range(len(locations)):
			if objects[it][1] == obj[1]: count += 1
		answer = str(count)
		qtype_out = [0, 0, 0, 0, 0, 1]'''
	# learn the distance relationships;
	if qtype == 0:
		question = "What is the shape of the object that is furthest from the " + colors[obj[0]] + " object?"
		maxim = 0
		for it in range(len(locations)):
			if obj == objects[it]: continue
			center = getCenter(objects[it], locations[it])
			dist = getDistance(cent, center)
			if dist >= maxim:
				maxim = dist
				answer = shapes[objects[it][1]]
		qtype_out = [0, 0, 0, 1, 0]
	# learn closeness relationships;
	elif qtype == 1:
		question = "What is the color of the object that is closest to the " + colors[obj[0]] + " object?"
		maxim = 500
		for it in range(len(locations)):
			if obj == objects[it]: continue
			center = getCenter(objects[it], locations[it])
			dist = getDistance(cent, center)
			if dist < maxim:
				maxim = dist
				answer = colors[objects[it][0]]
		qtype_out = [0, 0, 0, 0, 1]

	if nrtype == 0:
		nrquestion = "What is the shape of the " + colors[obj[0]] + " object?"
		nranswer = shapes[obj[1]]
		nrqtype_out = [1, 0, 0, 0, 0]
	elif nrtype == 1:
		nrquestion = "Is the " + colors[obj[0]] + " object on the left or on the right?"
		if getCenter(obj, locations[idx])[0] < (128.0 / 2): nranswer = "left"
		else: nranswer = "right"
		nrqtype_out = [0, 1, 0, 0, 0]
	elif nrtype == 2:
		nrquestion = "Is the " + colors[obj[0]] + " object on the top or on the bottom?"
		if getCenter(obj, locations[idx])[1] < (128.0 / 2): nranswer = "top"
		else: nranswer = "bottom"
		nrqtype_out = [0, 0, 1, 0, 0]

	v1 = copy.deepcopy(color_out)
	v2 = copy.deepcopy(color_out)
	v1.extend(qtype_out)
	v2.extend(nrqtype_out)

	if ret == 0:
		if counter[answer] >= math.ceil(data_size / 11): return False, "", "", -1
		counter[answer] += 1
		return question, json.dumps(v1), answer, ret
	else:
		if counter[nranswer] >= math.ceil(data_size / 11): return False, "", "", -1
		counter[nranswer] += 1
		return nrquestion, json.dumps(v2), nranswer, ret

# ------------------------------------------------------------------------------
def createImage(objects, locations):

	img = Image.new('RGB', (128, 128), (255,255,255))
	canvas = ImageDraw.Draw(img)
	for it in range(len(locations)):
		location = locations[it]
		obj = objects[it]
		# square;
		if obj[1] == 0:
			canvas.rectangle([
				location, 
				( 
					location[0] + sizes_widths[obj[2]],
					location[1] + sizes_widths[obj[2]]
				)
			], fill=colors_encodings[obj[0]])
		# circle;
		elif obj[1] == 1:
			canvas.ellipse([
				location, 
				( 
					location[0] + sizes_widths[obj[2]],
					location[1] + sizes_widths[obj[2]]
				)
			], fill=colors_encodings[obj[0]])
		# triangle;
		elif obj[1] == 2:
			canvas.polygon([
				( 
					location[0],
					location[1] + sizes_widths[obj[2]]
				),
				( 
					location[0] + (sizes_widths[obj[2]] / 2),
					location[1]
				),
				( 
					location[0] + sizes_widths[obj[2]],
					location[1] + sizes_widths[obj[2]]
				)
			], fill=colors_encodings[obj[0]])
	del canvas
	#img.save("test.png")
	return np.asarray(img.convert("RGB"))

# ------------------------------------------------------------------------------
def isOverlapping(locations, location, sizes, size):

	if location == (): return True
	for it in range(len(locations)):
		box1 = [locations[it][0]-8, locations[it][1]-8, locations[it][0]+sizes[it]+8, locations[it][1]+sizes[it]+8]
		box2 = [location[0], location[1], location[0]+size, location[1]+size]
		xi1 = np.maximum(box1[0], box2[0])
		yi1 = np.maximum(box1[1], box2[1])
		xi2 = np.minimum(box1[2], box2[2])
		yi2 = np.minimum(box1[3], box2[3])
		if np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0) > 0: return True
	return False

# ------------------------------------------------------------------------------
def generateLocations(objects, fallback=50):

	loc = []
	sizes = []
	for it in range(len(objects)):
		location = ()
		cc = 0
		while isOverlapping(loc, location, sizes, sizes_widths[objects[it][2]]):
			location = (np.random.randint(0, 103), np.random.randint(0, 103))
			if cc >= fallback:
				return False
			cc += 1
		loc.append(location)
		sizes.append(sizes_widths[objects[it][2]])
	return loc

# ------------------------------------------------------------------------------
def generateObjects(debug=False):

	global colors, shapes, sizes
	env = []
	#num_objects = np.random.randint(4, len(colors))
	num_objects = 5
	locked_colors = []
	for it in range(num_objects):
		# choose color;
		color = np.random.randint(0, len(colors))
		while color in locked_colors: color = np.random.randint(0, len(colors))
		# choose shape;
		shape = np.random.randint(0, len(shapes))
		# choose size;
		size = np.random.randint(1, len(sizes))
		# show us what you got!
		if debug: print(colors[color], shapes[shape], sizes[size])
		# add to the environment;
		env.append([color, shape, size])
		locked_colors.append(color)
	return env

# ------------------------------------------------------------------------------
def main(data_size=1, image_file="data_train", buffer_size=10000):

	file = open("./" + image_file + ".csv", "w+")
	file.write("index,question,encoding,answer,type\n")
	images = h5py.File("./" + image_file + ".h5", "w")
	images_dataset = images.create_dataset(image_file, shape=(0, 128, 128, 3), maxshape=(None, 128, 128, 3), chunks=True, compression="gzip")
	buff = []
	cc = -1

	#for it in range(data_size):
	while True:

		objects = generateObjects()
		loc = generateLocations(objects)
		if not loc: continue
		image = createImage(objects, loc)
		question, question_encoding, answer, qtype = generateQuestion(objects, loc, data_size)
		if not question: continue
		cc += 1
		# add to csv;
		#plt.imshow(image)
		#print(question, answer)
		#print(nrquestion, nranswer)
		file.write(str(cc)+",\""+question+"\",\""+question_encoding+"\",\""+answer+"\","+str(qtype)+"\n")
		# add to h5py;
		buff.append(image)
		if len(buff) >= buffer_size or cc == data_size - 1:
			size = len(images_dataset)
			images_dataset.resize(size + len(buff), axis=0)
			images_dataset[size:size + len(buff)] = buff
			buff = []
		# show some feedback;
		if cc % 1000:
			#print(counter)
			print("\r[STATUS]", cc / data_size * 100, "%                                         ", end="")
		if cc >= data_size - 1:
			size = len(images_dataset)
			images_dataset.resize(size + len(buff), axis=0)
			images_dataset[size:size + len(buff)] = buff
			break

	print("\n[STATUS] Done.\n")
	print(images_dataset.shape)
	file.flush()
	file.close()
	images.close()

################################################################################
main(20000, image_file="data_test", buffer_size=1000)