import requests

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_dataset.house_dataset_creator import HouseDatasetCreator

api_key = os.getenv("GOOGLE_API_KEY")
creator = HouseDatasetCreator(api_key)

iphone_res = '3024x4032'
address = '6682 Trigo Rd, Goleta, CA 93117', '3080 Deseret Dr, El Sobrante, CA 94803'
radius = 200
output_dir = 'nearby_houses'
creator.fetch_images_houses_nearby(size=iphone_res, 
                                   address=address, 
                                   radius=radius, 
                                   output_dir=output_dir)
