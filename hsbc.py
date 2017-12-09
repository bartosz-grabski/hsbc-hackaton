from darkflow.net.build import TFNet
import cv2
import sys

def syntax(argv):
	print("Please use the following syntax - ./hsbc.py <MODEL_FILEPATH> <WEIGHTS_FILEPATH> <THRESHOLD>")

def parse_arguments(argv):
	if len(argv) != 4:
		syntax(argv)
		exit();
	return argv[1:]



args = parse_arguments(sys.argv);
options = {"model": args[0] , "load": args[1], "threshold": float(args[2])}

label_translation = dict([
	('cell phone',('mobile',(0,0,255))),
	('remote',('mobile',(0,0,255))),
	('person',('person',(0,255,0))),
	('aeroplane',('drone',(0,0,255)))
])

tfnet = TFNet(options)

cap = cv2.VideoCapture(0)

def translate_label(a):
	current_label = a['label']
	label = label_translation[current_label][0]
	color = label_translation[current_label][1]
	a['label'] = label
	a['color'] = color
	return a;

#cv2.namedWindow("hsbc",cv2.WINDOW_NORMAL);

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	result = tfnet.return_predict(frame)
	result =  list(map(lambda x: translate_label(x),filter(lambda x: x['label'] in label_translation ,result)))
	for el in result:
		x1 = el['topleft']['x']
		y1 = el['topleft']['y']
		x2 = el['bottomright']['x']
		y2 = el['bottomright']['y']
		cv2.rectangle(frame, (x1, y1), (x2, y2), el['color'] , 2)
		cv2.putText(frame,el['label'], (x1+10,y1-10), cv2.FONT_HERSHEY_PLAIN,1,el['color'],2)
	cv2.imshow("hsbc",frame)
	cv2.resizeWindow("hsbc",frame.shape[0],frame.shape[1])
	if cv2.waitKey(1) == 27:
			break
	#print(result)
cv2.destroyAllWindows()
