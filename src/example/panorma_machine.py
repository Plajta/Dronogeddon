from djitellopy import Tello
import cv2
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
image_paths = []
tello = Tello()
tello.connect()
print("connect")
tello.streamoff()
print("off")
tello.streamon()
print("on")
print(tello.get_battery())
tello.takeoff()
print("takeoff")
tello.move_up(75)
take = 6
for i in range(24):
    img = tello.get_frame_read().frame
    cv2.imwrite(f'{take}Panorama-half-clockwise_{i}.jpg', img)
    image_paths.append(f'{take}Panorama-half-clockwise_{i}.jpg')
    time.sleep(0.5)
    tello.rotate_clockwise(15)

img = tello.get_frame_read().frame
cv2.imwrite(f'{take}Panorama-half-clockwise_{24}.jpg', img)
image_paths.append(f'{take}Panorama-half-clockwise_{24}.jpg')
time.sleep(1)


tello.streamoff()
tello.land()
print(image_paths)
#image_paths=['imgs/a.jpg','imgs/b.jpg']
# initialized a list of images
imgs = []
for i in range(len(image_paths)):
	imgs.append(cv2.imread(image_paths[i]))
	imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4)
	# this is optional if your input images isn't too large
	# you don't need to scale down the image
	# in my case the input images are of dimensions 3000x1200
	# and due to this the resultant image won't fit the screen
	# scaling down the images
# showing the original pictures
cv2.imshow('1',imgs[0])
cv2.imshow('2',imgs[1])

stitchy=cv2.Stitcher.create()
(dummy,output)=stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
# checking if the stitching procedure is successful
# .stitch() function returns a true value if stitching is
# done successfully
	print("stitching ain't successful")
else:
	print('Your Panorama is ready!!!')

# final output

feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# load and resize the input image
image = Image.open(f"output{take}.jpg")
new_height = 240 if image.height > 240 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# get the prediction from the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# remove borders
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))
height, width = output.shape

x = 0

for i in range (0,height-1,3):
    for j in range(0,width-1,3):	
        px = output[i,j]
        x = x + px - 2000
        #print(f"{j} {i} : {px-2000}")
x = x/(height*width)
print(x)
img1 = Image.fromarray(output)
img1 = img1.convert("L")
img1.save('test.jpg')
img1.save('test.png')

#print(f"{height} {width}")
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(0)