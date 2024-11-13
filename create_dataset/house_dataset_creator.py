import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.log_config import logger

from pathlib import Path
from PIL import Image, ImageEnhance

import requests
import math
import overpy

class HouseDatasetCreator:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_addresses(self, city):
        # Initialize the API
        api = overpy.Overpass()

        # Define the query
        query = f"""
            area["name"="{city}"]->.searchArea;
            (
            way["building"="residential"](area.searchArea);
            );
            out body;
        """

        # Execute the query
        result = api.query(query)

        # Collect addresses
        addresses = []
        for way in result.ways:
            address = {}
            
            for tag in way.tags:
                if tag.startswith("addr:"):
                    address[tag] = way.tags[tag]
                    
            # Only add if there's meaningful address information
            if "addr:housenumber" in address and "addr:street" in address and "addr:city" in address:
                address_str = ", ".join(
                    filter(
                        None,  # Filters out any None or empty string values
                        [
                            address.get('addr:housenumber', ''),
                            address.get('addr:street', ''),
                            address.get('addr:city', ''),
                            address.get('addr:state', ''),
                            address.get('addr:postcode', '')
                        ]
                    )
                )
                
                addresses.append(address_str)

        # Print and return the addresses
        logger.info(f"Found {len(addresses)} addresses in {city}")
        return addresses

    
    def fetch_coords_from_addr(self, address):
        """Convert an address to coordinates with the Google Geocode API"""
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': self.api_key
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                geometry = data['results'][0]['geometry']['location']
                lat = geometry['lat']
                lon = geometry['lng']

                return lat, lon
            else:
                logger.error(f"Geocoding API error: {data['status']}, {data.get('error_message', '')}")
                return None, None
        else:
            logger.error(f"HTTP error: {response.status_code}")
            return None, None
        
    def fetch_addr_from_coords(self, lat, lon):
        """Convert coordinates to an address with the Google Geocode API"""
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
        
        params = {
            "latlng": f"{lat},{lon}",
            "key": self.api_key
        }
        
        response = requests.get(geocode_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "OK":
                address = data["results"][0]["formatted_address"]
                return address
            else:
                print(f"Error: {data['status']}")
                return None
        else:
            print(f"Failed to fetch address: {response.status_code}")
            return None
            
    def fetch_camera_pos(self, lat: float, lon: float):
        """Fetch camera location so we can point the heading at the house."""
        metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={self.api_key}"
        response = requests.get(metadata_url)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                camera_lat = data['location']['lat']
                camera_lon = data['location']['lng']
                    
                return camera_lat, camera_lon
            
            else:
                logger.error(f"Error: {data['status']}")
                return None
        else:
            logger.error(f"Failed to fetch metadata: {response.status_code}")
            return None
    
    def calculate_heading(self, cam_lat, cam_lon, house_lat, house_lon):
        """Calculate what the heading needs to be to point at the house"""
        cam_lat, cam_lon = math.radians(cam_lat), math.radians(cam_lon)
        house_lat, house_lon = math.radians(house_lat), math.radians(house_lon)

        # Correct difference in longitudes (house to camera)
        dLon = house_lon - cam_lon

        # Calculate the bearing
        x = math.sin(dLon) * math.cos(house_lat)
        y = math.cos(cam_lat) * math.sin(house_lat) - (math.sin(cam_lat) * math.cos(house_lat) * math.cos(dLon))

        initial_bearing = math.atan2(x, y)

        # Convert bearing from radians to degrees and normalize it to 0-360
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing
        
    def fetch_google_street_view(self, size: str,
                                 lat: float, lon: float, 
                                 heading: float, pitch: float,
                                 zoom: int, brightness: float,
                                 flip: str, output_path: str
                                 ) -> bool:
        
        """Fetch Google Street View image given coordinates."""
        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': size, 
            'location': f'{lat},{lon}', 
            'heading': heading, 
            'pitch': pitch, 
            'zoom': zoom,
            'key': self.api_key 
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
                
            # Open image
            img = Image.open(output_path)

            # Apply zoom
            if zoom != 1.0:
                width, height = img.size
                new_width = int(width * zoom)
                new_height = int(height * zoom)
                
                # Calculate the crop area based on the zoom
                crop_width = (new_width - width) / 2
                crop_height = (new_height - height) / 2
                
                # Set crop boundaries
                left = int(crop_width)
                top = int(crop_height)
                right = int(crop_width + width)
                bottom = int(crop_height + height)
                
                # Crop and resize the image
                box = (left, top, right, bottom)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                img = img.crop(box)
                
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            
            # Flip the image
            if flip == 'horizontal' or flip == 'both': img = img.transpose(Image.FLIP_LEFT_RIGHT) 
            if flip == 'vertical' or flip == 'both': img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if flip not in ['horizontal', 'vertical', 'both'] and flip != None:
                logger.error("Flip must be set to 'horizontal', 'vertical', or 'both'. This image will not be flipped.")
            
            # Save image
            img.save(output_path)
                
            logger.info(f"Image saved as {output_path}")
            return True
        
        else:
            logger.error(f"Error fetching image: {response.status_code}, {response.text}")
            return False

    def fetch_house_image(
        self, 
        size: str, 
        address: str, 
        output_path: str, 
        lat_change: float = 0, 
        lon_change: float = 0,
        pitch = 0,
        zoom: int = 1,  # Default zoom level
        brightness: float = 1.0,
        flip: str = None # Default rotation angle
    ) -> bool:
        """
        Fetch a Google Street View image of a house using its address,
        with options for zoom and rotation for data augmentation.

        Args:
            size (str): Image size for Street View, e.g., "640x640".
            address (str): The house's physical address.
            output_path (str): File path where the image will be saved.
            lat_change (float): Latitude offset for changing angle.
            lon_change (float): Longitude offset for changing angle.
            brightness (float): Brightness factor of image.
            zoom (int): Zoom level for Street View (0-5).
            flip (str): 'horizontal', 'vertical', or 'both'.
                decides whether to flip image horizontally, vertically, or both.

        Returns:
            bool: True if the image was successfully fetched, False otherwise.
        """
        # If output_path has a directory, create it if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch latitude and longitude from the house address
        house_lat, house_lon = self.fetch_coords_from_addr(address)
        
        if house_lat is None or house_lon is None:
            logger.error(f"Could not fetch coordinates for address: {address}")
            return False

        # Fetch the camera's location (latitude and longitude) from Street View metadata
        camera_lat, camera_lon = self.fetch_camera_pos(house_lat, house_lon)
        
        if camera_lat is None or camera_lon is None:
            logger.error(f"Could not fetch Street View metadata for coordinates: {house_lat}, {house_lon}")
            return False

        # Calculate the heading to point the camera towards the house
        heading = self.calculate_heading(camera_lat + lat_change, camera_lon + lon_change, house_lat, house_lon)

        # Fetch and save the Google Street View image with the specified zoom level
        success = self.fetch_google_street_view(
            size=size,
            lat=house_lat,
            lon=house_lon,
            heading=heading,
            pitch=pitch,
            zoom=zoom, 
            brightness=brightness,
            flip=flip,
            output_path=output_path
        )

        if not success:
            logger.error(f"Failed to fetch image for address: {address}")
            return False

        return True

    
    def fetch_images_houses_nearby(self, size: int, address: str, radius: float, output_dir: str) -> bool:
        """
        Fetch a Google Street View images of nearby houses using an address.

        Args:
            size (str): Image size for Street View, e.g., "640x640".
            address (str): The house's physical address.
            radius (float): 
                The radius (in meters) around the target house within which to search for nearby houses.
            output_dir (str): Directory path where the images will be saved.

        Returns:
            bool: True if the image was successfully fetched, False otherwise.
        """
        dir = Path(output_dir)
        dir.mkdir(parents=True, exist_ok=True)
        
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        lat, lon = self.fetch_coords_from_addr(address=address)
        
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "key": self.api_key
        }
        
        response = requests.get(places_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "OK":
                places = data.get("results", [])
                
                for place in places:
                    place_location = place.get("geometry", {}).get("location", {})
                    place_lat = place_location.get("lat")
                    place_lon = place_location.get("lng")
                    
                    nearby_address = self.fetch_addr_from_coords(place_lat, place_lon)
                    clean_address = nearby_address.replace(' ', '').replace(',', '')
                    output_path = f"{output_dir}/{clean_address}.jpg"
                    
                    try:
                        self.fetch_house_image(size=size, 
                                            address=nearby_address, 
                                            output_path=output_path)
                    except: continue
                
                return True
            
            else:
                logger.error(f"Error: {data.get('status')}")
                return False
        
        else:
            logger.error(f"Failed to fetch places: {response.status_code}")
            return False

