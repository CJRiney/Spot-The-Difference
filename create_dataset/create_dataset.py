import logging
import os
import random

from house_dataset_creator import HouseDatasetCreator
from configs.log_config import logger
from dotenv import load_dotenv

logger.setLevel(logging.INFO)

load_dotenv('../.env')
api_key = os.getenv("GOOGLE_API_KEY")
creator = HouseDatasetCreator(api_key)

radius = 100
iphone_res = '3024x4032'
cities = [
    'Richmond',
    'Santa Barbara',
    'Vallejo',
    'Isla Vista',
    'Goleta',
    'Alameda',
    'San Rafael',
    'Pacifica',
    'Redwood City',
    'Walnut Creek',
    'Mill Valley',
    'San Mateo',
    'Pleasanton',
    'Santa Clara',
    'Menlo Park',
    'Encinitas',
    'Oceanside',
    'San Marcos',
    'El Cerrito',
    'Foster City',
]


idx = -1
for city in cities:
    logger.info(f'Creating sets for {city}')
    
    addresses = creator.get_addresses(city)
    for addr in addresses:
        idx += 1
        
        dataset_dir = (f'./datasets2/train/set_{idx}' 
                       if random.randint(0, 100) < 80 
                       else f'./datasets2/eval/set_{idx}')

        if os.path.isdir(dataset_dir):
            logger.info(f'{dataset_dir} exists, proceeding.')
            continue
        
        lat_change = random.randint(-10, 10) * .00001
        lon_change = random.randint(-10, 10) * .00001
        pitch = random.randint(-10, 20)
        brightness = random.randint(5, 20) * .1
        zoom = random.randint(10, 20) * .1
        flip = None
        
        logger.info(f'Creating set for {addr}')
        
        try:
            creator.fetch_house_image(address=addr, 
                                size=iphone_res,
                                output_path=f'{dataset_dir}/input.jpg',
                                lat_change=lat_change,
                                lon_change=lon_change,
                                pitch=pitch,
                                brightness=brightness,
                                zoom=zoom,
                                flip=flip)
            
        except: continue
        
        creator.fetch_house_image(address=addr, 
                                size=iphone_res,
                                output_path=f'{dataset_dir}/positive.jpg')
        
        creator.fetch_images_houses_nearby(size=iphone_res, 
                                   address=addr, 
                                   radius=radius, 
                                   output_dir=f'{dataset_dir}/negatives')

