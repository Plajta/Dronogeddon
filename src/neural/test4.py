from pretrained import model, convert_to_tensor, process_data, compute_dev, SCREEN_CENTER
import threading
import cv2


cap = cv2.VideoCapture(0)
#image = self.get_picture()
ret, image = cap.read()

torch_tensor = convert_to_tensor(image)

output = model(torch_tensor)
result = process_data(output, image)
print(result)
cv2.rectangle(image, (result[0],result[1]), (result[2],result[3]), (0, 255, 0), 2)

cv2.imshow("output_drone", image)