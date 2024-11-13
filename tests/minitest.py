'''
python3 minitest.py ../configs/minitest_config.yaml
'''

import timm, torch, urllib, json, os, sys
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from alignment import H4ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.configuration_SNN import SNN
from configs.dataclasses import SNNArguments

parser = H4ArgumentParser((SNNArguments))
snn_args = parser.parse()

test_path = './outputs/testing-eval-SNN.json'
train_path = './outputs/testing-training-SNN.json'
base_model = timm.create_model(snn_args.model_name, pretrained=True)
model = SNN(snn_args)

data_config = resolve_data_config(base_model.pretrained_cfg, model=base_model)
transform = create_transform(**data_config)

input_url_filenames = [
    ("https://photos.zillowstatic.com/fp/2aa50b463633bf1cfec84bd4d1d9cb47-cc_ft_768.webp", "./images/6682Trigo1.jpg"),
    ("https://photos.zillowstatic.com/fp/d5803d56647b5f92a6bd5a4eb4d4f6cc-cc_ft_384.webp", "./images/6682Trigo2.jpg")
]

similar_url_filenames = [
    ("https://photos.zillowstatic.com/fp/d5803d56647b5f92a6bd5a4eb4d4f6cc-cc_ft_384.webp", "./images/6682Trigo2.jpg"),
    ("https://photos.zillowstatic.com/fp/72bf9bed0435ab6301aecbe2e0f41c7c-uncropped_scaled_within_1536_1152.webp", "./images/6682Trigo3.jpg")
]

dissimilar_url_filenames = [
    ("https://photos.zillowstatic.com/fp/d5803d56647b5f92a6bd5a4eb4d4f6cc-cc_ft_384.webp", "./images/6682Trigo2.jpg"),
    ("https://photos.zillowstatic.com/fp/5a59f5c1175789171cf60e9df904183f-cc_ft_1536.webp", "./images/diffHouse1.jpg"),
    ("https://photos.zillowstatic.com/fp/fb4fcfd9dfaba9b8e0e1617bceb61001-cc_ft_1536.webp", "./images/diffHouse2.jpg"),
    ("https://photos.zillowstatic.com/fp/20b8d592c0974189c4c6cd64cc60ec57-cc_ft_1536.webp", "./images/diffHouse3.jpg"),
    ("https://photos.zillowstatic.com/fp/45bd6ab1e945e3cdcd620aeb55b05e51-cc_ft_1536.webp", "./images/diffHouse4.jpg"),
    ("https://photos.zillowstatic.com/fp/a61ab9c27ba1285847f3586944f94036-cc_ft_1536.webp", "./images/diffHouse5.jpg"),
    ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "./images/dog.jpg")
]

for image in input_url_filenames + similar_url_filenames + dissimilar_url_filenames: urllib.request.urlretrieve(image[0], image[1])

input_images = [Image.open(pair[1]).convert('RGB') for pair in input_url_filenames]
similar_images = [Image.open(pair[1]).convert('RGB') for pair in similar_url_filenames]
dissimilar_images = [Image.open(pair[1]).convert('RGB') for pair in dissimilar_url_filenames]

input_tensor = torch.stack([transform(img) for img in input_images])
similar_tensors = torch.stack([transform(img) for img in similar_images])
dissimilar_tensors = torch.stack([transform(img) for img in dissimilar_images])

torch.manual_seed(1)

# Test training here
output = model(input_tensor, similar_tensors, dissimilar_tensors)
results = { "Loss": output.item() }

with open(train_path, 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)

# Test evaluation here
# similar_outputs = model(input_tensor, similar_tensors)
# dissimilar_outputs = model(input_tensor, dissimilar_tensors)

# results = {
#     "Similar image scores": similar_outputs.tolist(),
#     "Dissimilar image scores": dissimilar_outputs.tolist(),
# }

# with open(test_path, 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)