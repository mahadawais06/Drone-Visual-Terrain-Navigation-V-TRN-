"""
Satellite Imagery Visual Terrain Recognition (V-TRN) Engine

This script implements a visual localization system that matches drone camera images against pre-loaded satellite map tiles to determine the drone's geographical position in GPS denied regions.

Property of Scry Aero LLC

"""

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform
import glob
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Folder containing all your GeoTIFF map tiles
MAP_DIRECTORY = r"D:\ScryAero\ReconDrone\maps"

class VTRNEngine:
    """
    Visual Terrain Recognition (V-TRN) Engine class.
    
    This class handles the loading of satellite map tiles, feature extraction and caching,
    and performs visual localization by matching drone images against the map database.
    """
    def __init__(self, map_path):
        """
        Initializes the V-TRN Engine by loading and processing satellite map tiles.
        
        Args:
            map_path (str): Path to the directory containing GeoTIFF map files or a glob pattern.
        """
        print(f"[SYSTEM] Initializing V-TRN Engine...")
        
        self.map_database = []  # List to store loaded map data
        
        # ASYMMETRIC FEATURE EXTRACTION:
        # ORB (Oriented FAST and Rotated BRIEF) is a fast feature detection algorithm
        # Use a massive number of features for the giant offline maps so we don't miss small landmarks
        self.map_orb = cv2.ORB_create(nfeatures=150000, scaleFactor=1.2, nlevels=8)
        # Use a smaller number of features for the fast, live drone camera to ensure real-time performance
        self.drone_orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        
        # BFMatcher (Brute Force Matcher) for matching ORB descriptors using Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Smart path handling: Check if user provided a folder or a specific pattern
        # This allows flexibility in specifying map file locations
        if os.path.isdir(map_path):
            # If it's a directory, look for .tif and .tiff files inside it
            map_files = glob.glob(os.path.join(map_path, "*.tif"))
            map_files.extend(glob.glob(os.path.join(map_path, "*.tiff")))
        else:
            # If it's a specific pattern (like "maps/*.tif"), just use that
            map_files = glob.glob(map_path)
            
        # Safety check: filter out any directories that might have accidentally been matched
        map_files = [f for f in map_files if os.path.isfile(f)]
        
        if not map_files:
            print(f"[ERROR] No .tif or .tiff files found in '{map_path}'. Please check the folder.")
            exit()
            
        print(f"[SYSTEM] Found {len(map_files)} map tiles. Loading database...")
        
        # Loop through each map file and load it into the database
        for file in map_files:
            try:
                print(f"  -> Loading {os.path.basename(file)}...")
                # Open the GeoTIFF file using Rasterio for geospatial data handling
                dataset = rasterio.open(file)
                
                # Load the full image array (Band 1 - Grayscale)
                # Satellite imagery is often stored as single-band grayscale for simplicity
                img_array = dataset.read(1)
                
                # --- FEATURE CACHING LOGIC ---
                # To speed up initialization, we cache the ORB features for each map tile
                # Feature extraction is computationally expensive, so we save/load from disk
                cache_file = file + ".cache.npz"
                
                if os.path.exists(cache_file):
                    print(f"     Found cached features! Loading from disk...")
                    # Load pre-computed keypoints and descriptors from cache
                    data = np.load(cache_file)
                    kp_data = data['keypoints']
                    des = data['descriptors']
                    
                    # Reconstruct OpenCV KeyPoint objects from the saved numpy array
                    # KeyPoints contain position, size, angle, etc. for each feature
                    kp = [cv2.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3], 
                                       response=row[4], octave=int(row[5]), class_id=int(row[6])) 
                          for row in kp_data]
                    print(f"     Successfully loaded {len(kp)} features from cache.")
                else:
                    # Pre-compute features using the MASSIVE feature limit
                    # This ensures we capture as many landmarks as possible in the map
                    print(f"     Extracting up to 150,000 ORB features (this takes time)...")
                    kp, des = self.map_orb.detectAndCompute(img_array, None)
                    
                    print(f"     Found {len(kp)} usable features. Saving to cache for next time...")
                    # Pack OpenCV KeyPoints into a standard numpy array so it can be saved
                    kp_data = np.array([(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp])
                    # Save compressed to reduce file size
                    # Save compressed to reduce file size
                    np.savez_compressed(cache_file, keypoints=kp_data, descriptors=des)
                
                # Store everything we need for this specific map tile in a dictionary
                # This includes the image data, geospatial info, and computed features
                self.map_database.append({
                    'filename': os.path.basename(file),
                    'dataset': dataset,  # Rasterio dataset object for geospatial operations
                    'image': img_array,  # Numpy array of the grayscale image
                    'transform': dataset.transform,  # Affine transform for pixel to coordinate conversion
                    'crs': dataset.crs,  # Coordinate Reference System (e.g., EPSG code)
                    'keypoints': kp,  # List of OpenCV KeyPoint objects
                    'descriptors': des  # Numpy array of feature descriptors
                })
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
                
        print("[SYSTEM] V-TRN Engine Database Ready.")

    def localize(self, drone_image_path):
        """
        Takes a real image file, searches ALL loaded maps, 
        and returns the absolute geographical coordinates.
        
        This method performs visual localization by:
        1. Extracting features from the drone image
        2. Matching features against each map in the database
        3. Using RANSAC to find the best geometric transformation
        4. Converting pixel coordinates to geographical coordinates
        
        Args:
            drone_image_path (str): Path to the drone camera image file
            
        Returns:
            tuple: ((geo_x, geo_y), (drone_frame, match_result)) or (None, None) if failed
        """
        start_time = time.time()
        
        # Load the external image in grayscale for feature extraction
        drone_frame = cv2.imread(drone_image_path, cv2.IMREAD_GRAYSCALE)
        if drone_frame is None:
             print(f"[ERROR] Could not load image: {drone_image_path}")
             return None, None
             
        print(f"\n[V-TRN] Processing live image: {os.path.basename(drone_image_path)}")
             
        # Extract features from the live drone frame using the optimized ORB detector
        kp_drone, des_drone = self.drone_orb.detectAndCompute(drone_frame, None)
        
        # Check if enough features were found for reliable matching
        if des_drone is None or len(des_drone) < 10:
            print("[WARNING] Not enough features in drone image.")
            return None, None

        # Initialize variables to track the best match across all maps
        best_match_result = None
        highest_inliers = 0  # Number of inliers from RANSAC, indicates match quality

        # --- SEARCH THE DATABASE ---
        # We check the drone image against every map we loaded
        for map_data in self.map_database:
            print(f"  -> Scanning against {map_data['filename']}...")
            
            # Match features between drone image and current map using brute force matching
            matches = self.matcher.match(des_drone, map_data['descriptors'])
            # Sort matches by distance (lower is better) and take top 50
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:50]
            
            # Need at least 10 matches to attempt homography estimation
            if len(good_matches) < 10:
                continue

            # Extract pixel coordinates of matched keypoints
            src_pts = np.float32([kp_drone[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([map_data['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Calculate Homography using RANSAC (RANdom SAmple Consensus)
            # Homography is a 3x3 matrix that describes the perspective transformation
            # RANSAC threshold of 10.0 accounts for distortions between Google Maps and USGS data
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            
            if M is not None:
                # Count inliers: points that fit the homography model within the threshold
                inliers = np.sum(mask) 
                # Keep track of the map with the most inliers (best geometric fit)
                if inliers > highest_inliers:
                    highest_inliers = inliers
                    best_match_result = {
                        'map': map_data,
                        'matrix': M,  # Homography matrix
                        'good_matches': good_matches,
                        'kp_drone': kp_drone,
                        'inliers': inliers
                    }

        # --- PROCESS THE WINNING MAP ---
        # If we found a match with enough inliers for confidence
        if best_match_result and best_match_result['inliers'] > 10:  # Threshold for a confident match
            winning_map = best_match_result['map']
            M = best_match_result['matrix']  # Homography matrix
            
            # Find where the center of the drone camera is pointing on the winning map
            # Transform the center point of the drone image using the homography
            h, w = drone_frame.shape
            drone_center = np.float32([[[w/2, h/2]]])
            map_center_pixel = cv2.perspectiveTransform(drone_center, M)[0][0]
            
            # Convert pixel coordinates to geographical coordinates using the map's transform
            # Rasterio's xy function converts from pixel coordinates to spatial coordinates
            geo_x, geo_y = rasterio.transform.xy(winning_map['transform'], map_center_pixel[1], map_center_pixel[0])
            
            # Convert the map's native coordinates to standard WGS84 Lat/Lon (EPSG:4326) for Google Maps
            lon_list, lat_list = transform(winning_map['crs'], 'EPSG:4326', [geo_x], [geo_y])
            lat, lon = lat_list[0], lon_list[0]
            google_maps_link = f"https://www.google.com/maps?q={lat},{lon}"
            
            # Calculate and display processing time
            process_time = time.time() - start_time
            print(f"\n[V-TRN] Localization Success! | Latency: {process_time:.2f}s")
            print(f"[V-TRN] Matched Map Tile: {winning_map['filename']}")
            print(f"[V-TRN] Confidence: {best_match_result['inliers']} confirmed feature matches.")
            print(f"[V-TRN] Native Map Coordinates: X: {geo_x:.2f}, Y: {geo_y:.2f} (CRS: {winning_map['crs']})")
            print(f"[V-TRN] Google Maps Link: {google_maps_link}")
            
            return (geo_x, geo_y), (drone_frame, best_match_result)
        else:
            print(f"\n[WARNING] Localization Failed: Could not find a confident match in the database.")
            return None, None

    def visualize_match(self, drone_frame, match_result):
        """
        Draws the visual proof of the match with zoomed cropping.
        
        This method creates visualizations to show:
        1. The features detected in the drone image
        2. The matched features between drone image and map with a bounding box
        """
        winning_map = match_result['map']
        kp_drone = match_result['kp_drone']
        good_matches = match_result['good_matches']
        matrix = match_result['matrix']  # The homography transformation matrix
        
        df_h, df_w = drone_frame.shape
        
        # --- VISUALIZATION 1: What the drone "sees" (ORB Features) ---
        # Draw circles around the features the AI found in the live photo
        drone_features_img = cv2.drawKeypoints(drone_frame, kp_drone, None, color=(0, 255, 0), flags=0)
        cv2.imshow("V-TRN Step 1: Drone Feature Extraction", cv2.resize(drone_features_img, (600, int(600 * df_h / df_w))))

        # --- VISUALIZATION 2: The Match & Projection (Zoomed In) ---
        # 1. Find the exact footprint of the drone on the massive offline map
        # Define the four corners of the drone image and transform them to map coordinates
        corners = np.float32([[0, 0], [0, df_h - 1], [df_w - 1, df_h - 1], [df_w - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, matrix)
        
        # 2. Calculate the center of that footprint on the map
        center_x = int(np.mean(transformed_corners[:, 0, 0]))
        center_y = int(np.mean(transformed_corners[:, 0, 1]))
        
        # 3. Create a zoomed-in crop window around the match
        # Increased pad from 800 to 1200 to show more of the surrounding neighborhood
        pad = 1200
        x_min = max(0, center_x - pad)
        y_min = max(0, center_y - pad)
        x_max = min(winning_map['image'].shape[1], center_x + pad)
        y_max = min(winning_map['image'].shape[0], center_y + pad)
        
        cropped_map = winning_map['image'][y_min:y_max, x_min:x_max]
        
        # 4. Shift the keypoints so the green lines draw correctly on the newly cropped map
        # Adjust keypoint coordinates to match the cropped region
        new_matches = []
        new_kp_map = []
        
        # Only process the top 20 matches we are actually drawing to guarantee index alignment
        for idx, m in enumerate(good_matches[:20]):
            orig_kp = winning_map['keypoints'][m.trainIdx]
            
            # Offset the X/Y coordinates by the crop amount
            shifted_kp = cv2.KeyPoint(
                x=orig_kp.pt[0] - x_min, 
                y=orig_kp.pt[1] - y_min, 
                size=orig_kp.size
            )
            new_kp_map.append(shifted_kp)
            
            # Re-map the match index explicitly to prevent OpenCV C++ binding issues
            new_match = cv2.DMatch()
            new_match.queryIdx = m.queryIdx
            new_match.trainIdx = idx
            new_match.distance = m.distance
            new_matches.append(new_match)
            
        # 5. Draw the side-by-side matches using the CROPPED map
        # Create a visualization showing matched features between drone image and cropped map
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
        img_matches = cv2.drawMatches(drone_frame, kp_drone, cropped_map, new_kp_map, new_matches, None, **draw_params)
        
        # 6. Shift the red bounding box corners to match the cropped map
        # Adjust the transformed corners to the cropped coordinate system
        shifted_corners = transformed_corners.copy()
        shifted_corners[:, 0, 0] -= x_min
        shifted_corners[:, 0, 1] -= y_min
        
        # Offset X by the drone image width (because the cropped map is on the right side of the screen)
        shifted_corners[:, 0, 0] += df_w 
        
        # Draw the thick red box representing the drone's position on the map
        img_matches = cv2.polylines(img_matches, [np.int32(shifted_corners)], True, (0, 0, 255), 6, cv2.LINE_AA)

        # 7. Scale down if the resulting image is still too tall for a standard monitor
        h_match, w_match = img_matches.shape[:2]
        max_display_height = 1000.0  # Increased display height to accommodate the wider crop
        
        if h_match > max_display_height:
            scale = max_display_height / h_match
            img_matches = cv2.resize(img_matches, (int(w_match * scale), int(h_match * scale)))
        
        cv2.imshow("V-TRN Step 2: Live Match & Projection Footprint (Zoomed)", img_matches)
        cv2.waitKey(0)  # Wait for user to press a key before closing

if __name__ == "__main__":
    # --- SETUP ---
    # Create an instance of the V-TRN Engine with the specified map directory
    engine = VTRNEngine(MAP_DIRECTORY)
    
    # Path to the drone image to be localized
    test_image = r"D:\ScryAero\ReconDrone\drone_test_3.png"
    
    # Check if the test image exists
    if os.path.exists(test_image):
        # Perform localization on the test image
        coords, vis_data = engine.localize(test_image)
        
        # If localization was successful, visualize the match
        if coords and vis_data:
            live_frame, match_result = vis_data
            print("\nPress any key in the image window to exit.")
            engine.visualize_match(live_frame, match_result)
    else:
        # Inform the user if the test image is not found
        print(f"\n[!] Please create a test image named '{test_image}' in this folder.")