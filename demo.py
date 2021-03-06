
import streamlit as st
import numpy as np
import os, urllib
import glob
import random 
from PIL import Image
import skimage
import skimage.io
import torch
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model = model.eval()


# Model
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_detection= model_detection.eval()

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
	# Render the readme as markdown using st.markdown.

	img_url = st.text_input('Please enter the url of your photo', 'http://farm7.staticflickr.com/6065/6135516797_5409d7201d_z.jpg')
	button = st.button('Run!')
	mode = st.sidebar.selectbox('Select your choice:', ["Detection","Segmentation"])
	
	threshold = st.sidebar.slider("Select value", 0.1, 1.0, 0.5, 0.1)


	if button:

		#downlaod the image
		img = skimage.io.imread(img_url)


		# load image
		_img = Image.fromarray(img)
		st.image(_img) 
		
		
		
		if mode == "Detection":
			
			model_detection.conf = threshold
			det_result = detect(model_detection, img)
			st.image(det_result)
			
		if mode == "Segmentation":
		  
			#st.image(_img)
			seg_result = segment(model, img)
			blended = Image.blend(_img, seg_result, alpha=threshold)
			st.image(blended)


			

def detect(model, input_image):

# 	#move the input and model to GPU for speed if available
# 	if torch.cuda.is_available():
# 		input_batch = input_batch.to('cuda')
# 		model.to('cuda')

	with torch.no_grad():
		output = model([input_image])
		output = output.render()
		
	return output[0]

def segment(model, input_image):
	preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	input_tensor = preprocess(input_image)
	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

	# move the input and model to GPU for speed if available
	if torch.cuda.is_available():
		input_batch = input_batch.to('cuda')
		model.to('cuda')

	with torch.no_grad():
		output = model(input_batch)['out'][0]
		#b, c, w, h

	output_predictions = output.argmax(0)

	palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
	colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
	colors = (colors % 255).numpy().astype("uint8")

	# plot the semantic segmentation predictions of 21 classes in each color
	r = Image.fromarray(output_predictions.byte().cpu().numpy())
	r.putpalette(colors)
	r = r.convert('RGB')

	return r

if __name__ == "__main__":
	main()
