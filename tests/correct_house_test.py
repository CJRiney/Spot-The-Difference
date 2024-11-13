import requests, logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_dataset.house_dataset_creator import HouseDatasetCreator

from configs.log_config import logger

from dotenv import load_dotenv

logger.setLevel(logging.INFO)

load_dotenv('../.env')
api_key = os.getenv("GOOGLE_API_KEY")
creator = HouseDatasetCreator(api_key)

iphone_res = '3024x4032'
addresses = [
    '6682, Trigo, Rd, Goleta', 
    '3080 Deseret Dr, El Sobrante, CA 94803',
    '6645 Del Playa Dr, Goleta, CA 93117', 
    '6643 Del Playa Dr, Goleta, CA 93117',
    '7216 Fordham Pl, Goleta, CA 93117',
    '6649 Abrego Rd, Goleta, CA 93117, USA'
    ]

city = 'Test'
# addresses = creator.get_addresses(city)

for addr in addresses:
    try:
        creator.fetch_house_image(address=addr, 
                              size=iphone_res,
                              output_path=f'./{city}/' + addr.replace(' ', '').replace(',', '') + '.jpg')
        
    except: continue
    
for addr in addresses:
    creator.fetch_house_image(address=addr, 
                              size=iphone_res,
                              output_path=f'./{city}_distorted/' + addr.replace(' ', '').replace(',', '') + '.jpg',
                              lat_change=.0000,
                              lon_change=.0000,
                              zoom=.5,
                              rotation=0)
    