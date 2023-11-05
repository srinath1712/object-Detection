import cv2


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


classes = []
with open("dnn_model/classes.txt", "r") as file_object:
	for class_name in file_object.readlines():
		class_name = class_name.strip()
		classes.append(class_name)

print("objects lists")
print(classes)

cap = cv2.VideoCapture(0)

button_person = False

def click_button(event, x, y, flags, params):
	global button_person
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)

		
while True:
	ret, frame = cap.read()

	(class_ids, scores, bboxes) = model.detect(frame)
	for class_id, score, bbox in zip(class_ids, scores, bboxes):
		(x, y, w, h) = bbox
		class_name = classes[class_id]


		cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2 , (200, 0,50),2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50),3)



	cv2.imshow("Object Detection", frame)
	key = cv2.waitKey(1)
	if key == 2:
		break
cap.release()
cv2.destoryAllWindows()