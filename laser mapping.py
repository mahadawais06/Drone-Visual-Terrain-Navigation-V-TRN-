import cv2
import numpy as np
import time
import math

# --- CONFIGURATION ---
# Camera Setup
CAMERA_INDEX = 0  # 0 is usually the default webcam. Change to 1 if using external USB.
RESOLUTION = (640, 480)

# Laser Color Thresholds (HSV)
# Adjust these if your green laser isn't being detected!
# Green is typically Hue 40-80.
LOWER_GREEN = np.array([40, 100, 100])
UPPER_GREEN = np.array([80, 255, 255])

# Geometry Settings (The Math)
# Distance between Camera lens and Laser output (in cm)
BASELINE_DISTANCE_CM = 15.0 
# Field of View of your camera (approx 60 degrees for most webcams)
FOV_VERTICAL_DEG = 48.0 

# Simulation Mode (Set to True to test without a real camera)
SIMULATION_MODE = False

class TerrainMapper:
    def __init__(self):
        self.points_3d = [] # Stores [x, y, z]
        self.map_image = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        
        if not SIMULATION_MODE:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(3, RESOLUTION[0])
            self.cap.set(4, RESOLUTION[1])
        
        # Calculate focal length in pixels (approximation)
        self.focal_length_px = (RESOLUTION[1] * 0.5) / math.tan(math.radians(FOV_VERTICAL_DEG * 0.5))

    def get_frame(self):
        """Captures a frame from camera or generates a fake one for simulation."""
        if SIMULATION_MODE:
            # Generate a fake frame with a moving green dot
            frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
            t = time.time()
            x = int((math.sin(t * 2) + 1) * 0.5 * RESOLUTION[0])
            # Simulate a "bump" in the terrain (dot moves up/down)
            y_terrain = 240 + int(math.sin(x/50) * 50) 
            cv2.circle(frame, (x, y_terrain), 5, (0, 255, 0), -1)
            # Add some fake noise
            noise = np.random.randint(0, 50, (RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            return frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def process_frame(self, frame):
        """Finds the laser dot and calculates depth."""
        # 1. Convert to HSV color space (better for color detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Threshold the image to get only green colors
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        
        # 3. Clean up noise (erode and dilate)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # 4. Find the center of the laser dot
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        if len(contours) > 0:
            # Assumes the largest green object is the laser
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                # --- THE TRIANGULATION MATH ---
                # Basic Principle: The lower the dot is in the image, the closer the ground is.
                # The higher the dot, the further away.
                
                # Pixel offset from center of image
                pixel_offset_y = center[1] - (RESOLUTION[1] / 2)
                
                # Calculate Angle
                if pixel_offset_y != 0:
                    angle_from_center = math.atan(pixel_offset_y / self.focal_length_px)
                    
                    # Distance Z = Baseline / tan(LaserAngle + CameraAngle)
                    # For this simple demo, we assume Laser is pointing straight down (0 deg)
                    # and we calculate depth relative to the camera angle.
                    # NOTE: This is a simplified "planar" approximation.
                    try:
                        depth = abs(BASELINE_DISTANCE_CM / math.tan(angle_from_center))
                    except ZeroDivisionError:
                        depth = 999
                else:
                    depth = 999 # Infinite/Horizon

                # Limit depth to reasonable values for display (e.g., 0 to 200cm)
                depth = min(max(depth, 0), 200)
                
                # Store the point for the heatmap
                self.update_map(center[0], center[1], depth)
                
                # Draw on the live feed
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.putText(frame, f"Depth: {depth:.1f} cm", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, mask

    def update_map(self, x, y, depth):
        """Draws the scanned terrain onto a persistent map."""
        # Convert depth to color (Red = Close, Blue = Far)
        # Normalize depth 0-200cm to 0-255 color
        color_val = int((depth / 200.0) * 255)
        # BGR Color: (Blue, Green, Red)
        # Close (Low Depth) -> Hot/Red. Far (High Depth) -> Cold/Blue.
        color = (color_val, 255 - abs(128 - color_val)*2, 255 - color_val)
        
        # "Paint" the map at the current x location
        # We draw a vertical strip representing the ground height at this X
        cv2.circle(self.map_image, (x, y), 3, color, -1)

    def run(self):
        print("--- Scry Aero Laser Mapper Started ---")
        print("Press 'q' to quit.")
        print(*"meowwwwwwwwwwwwwwwwwwwwwwwwwww", sep="\n")
        
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            
            processed_frame, mask = self.process_frame(frame)
            
            # Show the windows
            cv2.imshow("Drone Live Feed (Green Tracker)", processed_frame)
            cv2.imshow("Laser Mask", mask)
            cv2.imshow("Real-Time Terrain Map", self.map_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if not SIMULATION_MODE:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mapper = TerrainMapper()
    mapper.run()