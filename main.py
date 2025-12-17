import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import time
import io
import os
from dotenv import load_dotenv
from sd_generator import SDGenerator
from stroke_recorder import StrokeRecorder

print("Step 1/4: Loading libraries...")
load_dotenv()

API_KEY = os.getenv("STABILITY_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: STABILITY_API_KEY not found in environment variables")

# Get screen dimensions for dynamic sizing
pygame.init()
info = pygame.display.Info()
CANVAS_W, CANVAS_H = int(info.current_w * 0.85), int(info.current_h * 0.85)
CAM_W, CAM_H = 640, 480

COLOR_BTN = (109, 109, 109)
COLOR_TEXT = (255, 255, 255)
COLOR_INK_BLACK = (0, 0, 0, 20)
COLOR_INK_RED = (200, 40, 40, 20)
COLOR_SKELETON = (200, 200, 200)
COLOR_SELECTION = (0, 255, 0)
COLOR_HIGHLIGHT = (255, 215, 0)

STATE_DRAWING = 0      
STATE_GENERATING = 1   
STATE_ANNOTATION = 2   

current_state = STATE_DRAWING
current_round = 1 

print("Step 2/4: Initializing main canvas window...")
try:
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("Final Project: Inquiry to CD (Dual Hand Control)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    big_font = pygame.font.SysFont("Arial", 36)
    screen.fill((255, 255, 255))
    pygame.display.flip()
except Exception as e:
    print(f"Pygame initialization failed: {e}")
    sys.exit(1)

drawing_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
current_ai_image = None 

sd_gen = SDGenerator(api_key=None) 
recorder = StrokeRecorder()

print("Step 3/4: Initializing AI model...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, # Enable dual hand detection
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    print(f"MediaPipe initialization failed: {e}")
    sys.exit(1)

print("Step 4/4: Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
if not cap.isOpened(): print("❌ CRITICAL ERROR: Cannot open camera!")

# --- Global Variables ---
prev_index = None
prev_middle = None
selection_start = None
selection_rect = None
mouse_dragging = False
matched_gesture_id = None
auto_advance_after_gen = False
gesture_trigger_locked = False 
current_stroke_path = [] # Buffer for current stroke path

# Left hand state (default None)
control_hand_mode = None # "PAINT" or "STAMP"

btn_w, btn_h = 200, 45
btn_gen_rect = pygame.Rect(10, CANVAS_H - 60, btn_w, btn_h)
btn_clr_rect = pygame.Rect(220, CANVAS_H - 60, btn_w, btn_h)
btn_phase_rect = pygame.Rect(CANVAS_W - 220, CANVAS_H - 60, btn_w, btn_h)

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def draw_ink_particles(surface, x, y, px, py, color, base_size, density_mult):
    if px is None or py is None: return
    speed = dist((x, y), (px, py))
    speed = min(speed, 60)
    spread = map_range(speed, 0, 50, base_size * 2.0, base_size * 0.5)
    particle_count = map_range(speed, 0, 50, 2 * density_mult, 0.5 * density_mult)
    particle_count = int(max(1, particle_count))
    steps = int(speed / 6) + 1
    for i in range(steps + 1):
        t = i / steps
        cx = px + (x - px) * t
        cy = py + (y - py) * t
        for _ in range(particle_count):
            offset_x = np.random.uniform(-spread, spread)
            offset_y = np.random.uniform(-spread, spread)
            p_size = int(np.random.uniform(base_size * 0.6, base_size * 1.4))
            target_pos = (int(cx + offset_x), int(cy + offset_y))
            if 0 <= target_pos[0] < CANVAS_W and 0 <= target_pos[1] < CANVAS_H:
                pygame.draw.circle(surface, color, target_pos, p_size)

def is_fist(landmarks):
    wrist = landmarks[0]
    tips = [8, 12, 16, 20]
    threshold = 0.15
    all_folded = True
    for tip_idx in tips:
        d = dist((wrist.x, wrist.y), (landmarks[tip_idx].x, landmarks[tip_idx].y))
        if d > threshold:
            all_folded = False
            break
    return all_folded

def is_palm_open(landmarks):
    wrist = landmarks[0]
    tips = [4, 8, 12, 16, 20]
    threshold = 0.2
    count_extended = 0
    for tip_idx in tips:
        d = dist((wrist.x, wrist.y), (landmarks[tip_idx].x, landmarks[tip_idx].y))
        if d > threshold:
            count_extended += 1
    return count_extended >= 4

def draw_hand_skeleton(surface, landmarks, connections, color=(200, 200, 200)):
    # Transform all points
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * CAM_W), int(lm.y * CAM_H)
        sx = int(map_range(cx, 0, CAM_W, 0, CANVAS_W))
        sy = int(map_range(cy, 0, CAM_H, 0, CANVAS_H))
        coords.append((sx, sy))
        
    # Draw lines
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        pygame.draw.line(surface, color, coords[start_idx], coords[end_idx], 2)
        
    # Draw points
    for point in coords:
        pygame.draw.circle(surface, color, point, 4)

def clear_canvas():
    global drawing_surface, prev_index, prev_middle, current_ai_image, matched_gesture_id, current_round, current_state, current_stroke_path
    drawing_surface.fill((0,0,0,0))
    current_ai_image = None
    sd_gen.reset()
    recorder.clear() 
    prev_index = None
    prev_middle = None
    matched_gesture_id = None
    current_round = 1
    current_state = STATE_DRAWING
    current_stroke_path = []
    print("🔄 Canvas reset, back to Round 1")

def trigger_generation(advance_after=False):
    global auto_advance_after_gen, current_state
    if sd_gen.is_generating: return
    
    auto_advance_after_gen = advance_after
    if advance_after:
        current_state = STATE_GENERATING
    
    temp_surf = pygame.Surface((CANVAS_W, CANVAS_H))
    temp_surf.fill((255, 255, 255))
    if current_ai_image: temp_surf.blit(current_ai_image, (0,0))
    temp_surf.blit(drawing_surface, (0,0))
    
    img_bytes = io.BytesIO()
    pygame.image.save(temp_surf, img_bytes, "PNG")
    img_bytes.seek(0)
    sd_gen.generate(img_bytes, CANVAS_W, CANVAS_H)

def next_phase():
    global current_state, selection_rect, current_round, drawing_surface, current_stroke_path
    
    if current_state == STATE_DRAWING:
        print(f"Round {current_round} finished, starting generation...")
        trigger_generation(advance_after=True)
        
    elif current_state == STATE_ANNOTATION:
        if selection_rect and current_ai_image:
            avg_gesture, strokes = recorder.get_gesture_in_region(selection_rect)
            
            try:
                clip_rect = pygame.Rect(selection_rect)
                texture_crop = current_ai_image.subsurface(clip_rect).copy() 
            except Exception as e:
                print(f"Crop failed: {e}")
                texture_crop = None

            if avg_gesture is not None and texture_crop:
                recorder.save_learned_gesture(avg_gesture, strokes, texture_surface=texture_crop)
                drawing_surface.fill((0,0,0,0))
                current_state = STATE_DRAWING
                current_round += 1
                current_stroke_path = []
                print(f"✅ Annotation complete! Starting Round {current_round}")
            else:
                print("⚠️ Invalid selection or no gesture data")
        else:
             print("⚠️ Please draw a selection box first")

# --- Main Loop ---
running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Prioritize button clicks, even in ANNOTATION mode
                    if btn_phase_rect.collidepoint(event.pos):
                         next_phase()
                    elif btn_clr_rect.collidepoint(event.pos):
                        clear_canvas()
                    elif btn_gen_rect.collidepoint(event.pos) and current_state != STATE_ANNOTATION:
                        trigger_generation()
                    
                    elif current_state == STATE_ANNOTATION:
                        selection_start = event.pos
                        mouse_dragging = True
                        selection_rect = None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: # Enter key
                    next_phase()
                elif event.key == pygame.K_c: # C key to clear
                    clear_canvas()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and current_state == STATE_ANNOTATION:
                    mouse_dragging = False
                    if selection_start:
                        w = event.pos[0] - selection_start[0]
                        h = event.pos[1] - selection_start[1]
                        x, y = selection_start
                        if w < 0: x += w; w = abs(w)
                        if h < 0: y += h; h = abs(h)
                        selection_rect = (x, y, w, h)
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and current_state == STATE_ANNOTATION:
                    w = event.pos[0] - selection_start[0]
                    h = event.pos[1] - selection_start[1]
                    x, y = selection_start
                    if w < 0: x += w; w = abs(w)
                    if h < 0: y += h; h = abs(h)
                    selection_rect = (x, y, w, h)

        result_img = sd_gen.get_result()
        if result_img:
            current_ai_image = result_img
            drawing_surface.fill((0,0,0,0))
            if auto_advance_after_gen and current_state == STATE_GENERATING:
                current_state = STATE_ANNOTATION
                auto_advance_after_gen = False
                print("Generation complete, please select a sampling region.")

        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        screen.fill((255, 255, 255)) 
        
        if current_ai_image: screen.blit(current_ai_image, (0, 0))
        else: screen.blit(drawing_surface, (0, 0))
        
        if current_ai_image: screen.blit(drawing_surface, (0, 0))

        if current_state == STATE_ANNOTATION and selection_rect:
            pygame.draw.rect(screen, COLOR_SELECTION, selection_rect, 2)
            sel_text = font.render("Selected! Press 'Start Next Round' to confirm.", True, (0, 100, 0))
            screen.blit(sel_text, (selection_rect[0], selection_rect[1] - 25))

        matched_gesture_id = None
        control_hand_mode = None # Reset every frame
        drawing_hand_landmarks = None # Reset every frame

        if results.multi_hand_landmarks and not sd_gen.is_generating:
            # MediaPipe return order is not guaranteed, need to use multi_handedness
            # But since we flipped the image, 'Left' on screen corresponds to user's left hand (mirror)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine handedness
                label = results.multi_handedness[idx].classification[0].label
                # Note: MediaPipe Left/Right is from "camera's view of person"
                # We flipped, so label="Left" is displayed on left screen, corresponding to user's left hand
                
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- Right Hand (Mode Controller) ---
                # Note: After flip, user's right hand is labeled "Left" by MediaPipe
                if label == "Left": 
                    # Draw Right Hand indicator
                    
                    h, w, c = img.shape
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    # Mapping
                    wrist_x = int(map_range(wrist_x, 0, w, 0, CANVAS_W))
                    wrist_y = int(map_range(hand_landmarks.landmark[0].y * h, 0, h, 0, CANVAS_H))
                    
                    if is_fist(hand_landmarks.landmark):
                        control_hand_mode = "STAMP" # Fist = Stamp
                        pygame.draw.circle(screen, (255, 0, 0), (wrist_x, wrist_y), 30) # Red Light
                    elif is_palm_open(hand_landmarks.landmark):
                        control_hand_mode = "PAINT" # Open = Paint
                        pygame.draw.circle(screen, (0, 255, 0), (wrist_x, wrist_y), 30) # Green Light
                    else:
                        control_hand_mode = "NEUTRAL" # Other
                
                # --- Left Hand (Action Executor) ---
                # Note: After flip, user's left hand is labeled "Right" by MediaPipe
                elif label == "Right":
                    # ⭐ Draw Left Hand (Action Hand) Skeleton ⭐
                    draw_hand_skeleton(screen, hand_landmarks.landmark, mp_hands.HAND_CONNECTIONS, color=(180, 180, 180))
                    drawing_hand_landmarks = hand_landmarks

            # Only execute action if Left Hand is detected
            if drawing_hand_landmarks and current_state == STATE_DRAWING:
                hand_landmarks = drawing_hand_landmarks # Alias variable
                
                h, w, c = img.shape
                def get_coord(idx):
                    cx, cy = int(hand_landmarks.landmark[idx].x * w), int(hand_landmarks.landmark[idx].y * h)
                    sx = int(map_range(cx, 0, w, 0, CANVAS_W))
                    sy = int(map_range(cy, 0, h, 0, CANVAS_H))
                    return sx, sy

                index_pos = get_coord(8)
                
                # ⭐ Left Hand Logic: Open = Pause/Reset/Trigger; Non-Open = Action/Record ⭐
                if is_palm_open(hand_landmarks.landmark):
                    # [PAUSE/TRIGGER MODE]
                    # If there is a accumulated stroke path, perform recognition (Stamp)
                    if len(current_stroke_path) > 10: # Only judge if path is long enough
                        # Perform full path recognition
                        matched_gesture = recorder.recognize_stroke(current_stroke_path, threshold=0.25)
                        
                        if matched_gesture:
                            matched_gesture_id = matched_gesture['id']
                            if matched_gesture.get('texture'):
                                tex = matched_gesture['texture']
                                # Stamp at the last point
                                last_pt = current_stroke_path[-1]
                                tex_x = last_pt['x'] - tex.get_width() // 2
                                tex_y = last_pt['y'] - tex.get_height() // 2
                                drawing_surface.blit(tex, (tex_x, tex_y))
                            else:
                                pygame.draw.circle(screen, (255, 0, 0), index_pos, 20)
                    
                    # Clear buffer and state
                    current_stroke_path = []
                    prev_index = None
                    gesture_trigger_locked = False 
                    
                    # Visual Feedback: Draw hollow circle at index finger
                    pygame.draw.circle(screen, (150, 150, 150), index_pos, 10, 2)
                    
                else:
                    # [ACTION MODE]
                    if control_hand_mode == "PAINT":
                        # Paint Mode: Draw black ink (do not accumulate path)
                        current_stroke_path = [] # Ensure stamp path is cleared
                        
                        draw_ink_particles(drawing_surface, index_pos[0], index_pos[1], 
                                         prev_index[0] if prev_index else None, 
                                         prev_index[1] if prev_index else None, 
                                         COLOR_INK_BLACK, 12, 0.8)
                        recorder.add_point(index_pos[0], index_pos[1], hand_landmarks.landmark)
                        prev_index = index_pos
                        
                    elif control_hand_mode == "STAMP":
                        # Stamp Mode: Record trajectory, no drawing, no immediate stamp
                        # Visual Feedback: Draw red line to indicate recording
                        if prev_index:
                            pygame.draw.line(screen, (255, 0, 0), prev_index, index_pos, 2)
                        
                        # Add current point to path buffer
                        lm_data = []
                        for lm in hand_landmarks.landmark:
                            lm_data.append([lm.x, lm.y, lm.z])
                        
                        current_stroke_path.append({
                            'x': index_pos[0],
                            'y': index_pos[1],
                            'landmarks': np.array(lm_data)
                        })
                        
                        prev_index = index_pos
                        
                    else:
                        prev_index = None
                        current_stroke_path = []
        else:
            prev_index = None

        # --- UI ---
        if current_state != STATE_GENERATING:
            pygame.draw.rect(screen, COLOR_BTN, btn_gen_rect, border_radius=8)
            text_gen = font.render("Regenerate", True, COLOR_TEXT)
            screen.blit(text_gen, (btn_gen_rect.centerx - text_gen.get_width()//2, btn_gen_rect.centery - text_gen.get_height()//2))

            pygame.draw.rect(screen, COLOR_BTN, btn_clr_rect, border_radius=8)
            text_clr = font.render("Reset All", True, COLOR_TEXT)
            screen.blit(text_clr, (btn_clr_rect.centerx - text_clr.get_width()//2, btn_clr_rect.centery - text_clr.get_height()//2))
            
            phase_text_str = f"Next Round ({current_round} -> {current_round+1})"
            if current_state == STATE_ANNOTATION: phase_text_str = f"Start Round {current_round+1}"
            
            pygame.draw.rect(screen, (50, 100, 200), btn_phase_rect, border_radius=8)
            text_phase = font.render(phase_text_str, True, COLOR_TEXT)
            screen.blit(text_phase, (btn_phase_rect.centerx - text_phase.get_width()//2, btn_phase_rect.centery - text_phase.get_height()//2))

        if current_state == STATE_ANNOTATION:
            info = big_font.render("Sample a Texture: Draw a box!", True, (0, 0, 0))
            screen.blit(info, (20, 20))
        elif current_state == STATE_DRAWING:
            lh_text = control_hand_mode if control_hand_mode else "NO HAND"
            
            # Action Hand Status Text
            if drawing_hand_landmarks:
                if is_palm_open(drawing_hand_landmarks.landmark):
                    status_text = "PAUSED (Check Gesture)"
                elif control_hand_mode == "STAMP":
                    status_text = "RECORDING GESTURE..."
                else:
                    status_text = "PAINTING"
            else:
                status_text = "NO HAND"

            info = font.render(f"Ctrl Hand (R): {lh_text} | Action Hand (L): {status_text}", True, (100, 100, 100))
            screen.blit(info, (20, 20))

        inventory_start_y = 60
        inventory_w = 160 
        inventory_h = 80
        padding = 10
        
        for i, gesture in enumerate(recorder.learned_gestures):
            thumb = gesture.get('thumbnail')
            if thumb:
                scaled_thumb = pygame.transform.scale(thumb, (inventory_w, inventory_h))
                item_y = inventory_start_y + i * (inventory_h + padding + 20)
                item_x = CANVAS_W - inventory_w - padding
                bg_rect = pygame.Rect(item_x - padding, item_y - padding, inventory_w + padding*2, inventory_h + padding*2)
                border_color = COLOR_HIGHLIGHT if gesture['id'] == matched_gesture_id else (200, 200, 200)
                line_width = 4 if gesture['id'] == matched_gesture_id else 2
                
                is_new = gesture.get('is_new_session', False)
                bg_color = (230, 245, 255) if is_new else (240, 240, 240) # Light blue for new, grey for old
                
                pygame.draw.rect(screen, bg_color, bg_rect, border_radius=8)
                pygame.draw.rect(screen, border_color, bg_rect, line_width, border_radius=8)
                screen.blit(scaled_thumb, (item_x, item_y))
                
                # ID and Tag
                id_str = f"#{i+1}"
                if is_new: id_str += " (NEW)"
                text_color = (0, 100, 200) if is_new else (50, 50, 50)
                
                id_text = font.render(id_str, True, text_color)
                screen.blit(id_text, (item_x, item_y - 20))

        if sd_gen.is_generating or current_state == STATE_GENERATING:
            overlay = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            load_text = big_font.render("AI Dreaming...", True, (255, 255, 255))
            screen.blit(load_text, (CANVAS_W//2 - load_text.get_width()//2, CANVAS_H//2))

        pygame.display.flip()
        
        preview_w = 480
        preview_h = int(preview_w * (CAM_H / CAM_W))
        preview_img = cv2.resize(img, (preview_w, preview_h)) 
        cv2.imshow("Camera Preview", preview_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): running = False
        clock.tick(60)

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"Runtime error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'cap' in locals() and cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    pygame.quit()