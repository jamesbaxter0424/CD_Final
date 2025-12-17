import numpy as np
import json
import os
import time
import pygame
from collections import deque

class StrokeRecorder:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.strokes = []
        self.learned_gestures = []
        
        # Buffer for trajectory recognition
        # Assuming FPS=30, 2 seconds is about 60 frames
        self.gesture_buffer = deque(maxlen=30) # 1 second average should be stable enough, 2 seconds might be too slow
        
        self.load_database()

    def add_point(self, x, y, landmarks):
        lm_data = []
        if landmarks:
            for lm in landmarks:
                lm_data.append([lm.x, lm.y, lm.z])
        
        self.strokes.append({
            'x': x,
            'y': y,
            'landmarks': np.array(lm_data)
        })

    def update_buffer(self, landmarks):
        """
        Update gesture buffer in real-time for current gesture recognition
        """
        lm_data = []
        for lm in landmarks:
            lm_data.append([lm.x, lm.y, lm.z])
        self.gesture_buffer.append(np.array(lm_data))

    def get_gesture_in_region(self, rect):
        rx, ry, rw, rh = rect
        selected_strokes = []
        
        for s in self.strokes:
            if rx <= s['x'] <= rx + rw and ry <= s['y'] <= ry + rh:
                selected_strokes.append(s)
        
        if not selected_strokes:
            return None, []

        all_landmarks = np.array([s['landmarks'] for s in selected_strokes])
        avg_gesture = np.mean(all_landmarks, axis=0)
        
        return avg_gesture, selected_strokes

    def _generate_thumbnail(self, stroke_points, texture_surface=None, size=(160, 80)):
        """
        Generate thumbnail: Left side is stroke path, Right side is texture preview
        """
        thumbnail = pygame.Surface(size, pygame.SRCALPHA)
        # Draw semi-transparent background
        pygame.draw.rect(thumbnail, (255, 255, 255, 200), (0, 0, size[0], size[1]), border_radius=5)
        
        # 1. Draw stroke path on left half (80x80)
        stroke_area_w = size[0] // 2
        
        if stroke_points:
            xs = [p['x'] for p in stroke_points]
            ys = [p['y'] for p in stroke_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w, h = max_x - min_x, max_y - min_y
            if w == 0: w = 1
            if h == 0: h = 1

            scale = min((stroke_area_w-10)/w, (size[1]-10)/h)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            prev_pt = None
            for p in stroke_points:
                tx = (p['x'] - center_x) * scale + stroke_area_w/2
                ty = (p['y'] - center_y) * scale + size[1]/2
                curr_pt = (tx, ty)
                if prev_pt:
                    pygame.draw.line(thumbnail, (0, 0, 0), prev_pt, curr_pt, 2)
                prev_pt = curr_pt

        # 2. Draw texture on right half (if exists)
        if texture_surface:
            # Scale texture to fit right half
            tex_w = size[0] // 2 - 4
            tex_h = size[1] - 4
            scaled_tex = pygame.transform.smoothscale(texture_surface, (tex_w, tex_h))
            thumbnail.blit(scaled_tex, (size[0]//2 + 2, 2))
            
        return thumbnail

    def save_learned_gesture(self, gesture_vector, stroke_points, texture_surface=None):
        serializable_strokes = []
        for s in stroke_points:
            serializable_strokes.append({'x': s['x'], 'y': s['y']})
            
        timestamp_id = int(time.time())
        texture_filename = None

        if texture_surface:
            texture_filename = f"texture_{timestamp_id}.png"
            full_path = os.path.join(self.data_dir, texture_filename)
            pygame.image.save(texture_surface, full_path)

        new_gesture = {
            'id': timestamp_id,
            'gesture_vector': gesture_vector.tolist(),
            'stroke_example': serializable_strokes,
            'texture_file': texture_filename
        }
        
        thumb = self._generate_thumbnail(serializable_strokes, texture_surface)

        self.learned_gestures.append({
            'id': new_gesture['id'],
            'gesture_vector': gesture_vector, 
            'stroke_example': stroke_points,
            'thumbnail': thumb,
            'texture': texture_surface,
            'is_new_session': True
        })
        
        json_filename = os.path.join(self.data_dir, f"gesture_{timestamp_id}.json")
        with open(json_filename, 'w') as f:
            json.dump(new_gesture, f)
            
        print(f"✅ Learned and saved new brush! ID: {timestamp_id}")

    def load_database(self):
        print("📂 Loading gesture history database...")
        self.learned_gestures = [] 
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json") and filename.startswith("gesture_"):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                        texture_surf = None
                        if 'texture_file' in data and data['texture_file']:
                            tex_path = os.path.join(self.data_dir, data['texture_file'])
                            if os.path.exists(tex_path):
                                texture_surf = pygame.image.load(tex_path)

                        thumb = self._generate_thumbnail(data['stroke_example'], texture_surf)

                        self.learned_gestures.append({
                            'id': data['id'],
                            'gesture_vector': np.array(data['gesture_vector']),
                            'stroke_example': data['stroke_example'],
                            'thumbnail': thumb,
                            'texture': texture_surf,
                            'is_new_session': False
                        })
                except Exception as e:
                    print(f"❌ Failed to load {filename}: {e}")

    def recognize_gesture(self, threshold=0.15):
        """
        Use average gesture features in buffer for comparison (Old method, now mainly for static gestures or single frame)
        """
        if not self.learned_gestures or len(self.gesture_buffer) < 5: 
            return None

        curr_vec_avg = np.mean(np.array(self.gesture_buffer), axis=0)
        
        best_match = None
        min_dist = float('inf')

        for learned in self.learned_gestures:
            target_vec = learned['gesture_vector']
            dist = np.mean(np.linalg.norm(curr_vec_avg - target_vec, axis=1))
            
            if dist < min_dist:
                min_dist = dist
                best_match = learned

        if min_dist < threshold:
            return best_match
        return None

    def recognize_stroke(self, stroke_points, threshold=0.2): # Slightly relaxed threshold as full path variance might be larger
        """
        Use average vector of the full stroke path for comparison
        stroke_points: list of dict {'x', 'y', 'landmarks': np.array}
        """
        if not self.learned_gestures or not stroke_points:
            return None

        # Extract landmarks from all points and calculate average vector
        all_landmarks = np.array([p['landmarks'] for p in stroke_points])
        stroke_avg_vec = np.mean(all_landmarks, axis=0)

        best_match = None
        min_dist = float('inf')

        for learned in self.learned_gestures:
            target_vec = learned['gesture_vector']
            # Calculate mean distance between two 21x3 matrices
            dist = np.mean(np.linalg.norm(stroke_avg_vec - target_vec, axis=1))
            
            if dist < min_dist:
                min_dist = dist
                best_match = learned

        if min_dist < threshold:
            return best_match
        return None

    def clear(self):
        self.strokes = []
        self.gesture_buffer.clear()
