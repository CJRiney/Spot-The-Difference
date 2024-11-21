'''
python3 onnxtest.py ../configs/onnxtest_config.yaml
'''

import onnxruntime as ort
import timm, torch, urllib, json, os, sys, numpy
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from alignment import H4ArgumentParser

result_path = './outputs/onnx-test.json'
onnx_model_path = "../saved_models/mobilenetv3_large_snn.onnx"  # Path to your exported ONNX model
session = ort.InferenceSession(onnx_model_path)

base_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
data_config = resolve_data_config(base_model.pretrained_cfg, model=base_model)
transform = create_transform(**data_config)

tests = ['Against itself', 'Against the same house', 'Against a different house', 'Against a dog']

input_url_filename = ("https://photos.zillowstatic.com/fp/d5803d56647b5f92a6bd5a4eb4d4f6cc-cc_ft_384.webp", "./images/6682Trigo2.jpg")

target_url_filenames = [
    ("https://photos.zillowstatic.com/fp/d5803d56647b5f92a6bd5a4eb4d4f6cc-cc_ft_384.webp", "./images/6682Trigo2.jpg"),
    ("https://photos.zillowstatic.com/fp/2aa50b463633bf1cfec84bd4d1d9cb47-cc_ft_768.webp", "./images/6682Trigo1.jpg"),
    ("https://photos.zillowstatic.com/fp/5a59f5c1175789171cf60e9df904183f-cc_ft_1536.webp", "./images/diffHouse1.jpg"),
    ("https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*", "./images/dog.jpg")
]

for image in [input_url_filename] + target_url_filenames: urllib.request.urlretrieve(image[0], image[1])

input_image = Image.open(input_url_filename[1]).convert('RGB')
target_images = [Image.open(pair[1]).convert('RGB') for pair in target_url_filenames]

input_tensor = transform(input_image).unsqueeze(0).numpy()
target_tensors = [transform(img).unsqueeze(0).numpy() for img in target_images]

test_tensor_mapping = {test: tensor for test, tensor in zip(tests, target_tensors)}

input_1_name = session.get_inputs()[0].name
input_2_name = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name 

def evaluate(input_tensor_1, input_tensor_2):
    return session.run([output_name], {input_1_name: input_tensor_1, input_2_name: input_tensor_2})[0].item()

results = {test: evaluate(input_tensor, target_tensor) for test, target_tensor in test_tensor_mapping.items()}

with open(result_path, 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)