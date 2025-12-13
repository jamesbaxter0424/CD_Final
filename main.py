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
from sd_generator import SDGenerator  # 引入新模組

print("步驟 1/4: 載入函式庫...")

# 載入 .env 檔案
load_dotenv()

# --- 設定與常數 ---
# 優先從環境變數讀取，如果沒有則使用預設值 (或報錯)
API_KEY = os.getenv("STABILITY_API_KEY")
if not API_KEY:
    print("⚠️ 警告: 未找到 STABILITY_API_KEY 環境變數，請檢查 .env 檔案")


CANVAS_W, CANVAS_H = 1152, 896
CAM_W, CAM_H = 640, 480

# 顏色定義
COLOR_BTN = (109, 109, 109)
COLOR_TEXT = (255, 255, 255)
COLOR_INK_BLACK = (0, 0, 0, 20)
COLOR_INK_RED = (200, 40, 40, 20)
COLOR_SKELETON = (200, 200, 200)

# --- 初始化 Pygame (主畫布視窗) ---
print("步驟 2/4: 初始化主畫布視窗...")
try:
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("Final Project: Inquiry to CD (Main Canvas + AI)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    big_font = pygame.font.SysFont("Arial", 36)
    
    screen.fill((255, 255, 255))
    pygame.display.flip()
    
except Exception as e:
    print(f"Pygame 初始化失敗: {e}")
    sys.exit(1)

# 畫布圖層
drawing_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
current_ai_image = None # 當前顯示的 AI 圖

# 初始化生成器
sd_gen = SDGenerator(API_KEY)

# --- 初始化 MediaPipe ---
print("步驟 3/4: 初始化 AI 模型...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    print(f"MediaPipe 初始化失敗: {e}")
    sys.exit(1)

# --- 初始化 攝影機 ---
print("步驟 4/4: 開啟攝影機...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    print("❌ 嚴重錯誤: 無法開啟攝影機！")
else:
    print("✅ 攝影機開啟成功！")

# --- 全域變數 ---
prev_index = None
prev_middle = None
btn_gen_rect = pygame.Rect(10, CANVAS_H - 60, 250, 45)
btn_clr_rect = pygame.Rect(270, CANVAS_H - 60, 250, 45)

# --- 輔助函式 ---
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

def clear_canvas():
    global drawing_surface, prev_index, prev_middle, current_ai_image
    drawing_surface.fill((0,0,0,0))
    current_ai_image = None
    sd_gen.reset() # 重置生成器狀態
    prev_index = None
    prev_middle = None

def trigger_generation():
    if sd_gen.is_generating: return
    
    # 準備圖片資料
    temp_surf = pygame.Surface((CANVAS_W, CANVAS_H))
    temp_surf.fill((255, 255, 255))
    temp_surf.blit(drawing_surface, (0,0))
    
    img_bytes = io.BytesIO()
    pygame.image.save(temp_surf, img_bytes, "PNG")
    img_bytes.seek(0)
    
    # 呼叫生成器
    sd_gen.generate(img_bytes, CANVAS_W, CANVAS_H)

# --- 主迴圈 ---
running = True
try:
    while running:
        # 1. Pygame 輸入處理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if btn_gen_rect.collidepoint(event.pos):
                    trigger_generation()
                elif btn_clr_rect.collidepoint(event.pos):
                    clear_canvas()

        # 檢查生成器是否有結果
        result_img = sd_gen.get_result()
        if result_img:
            current_ai_image = result_img

        # 2. 讀取攝影機
        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # --- 處理 Pygame 主畫布 ---
        screen.fill((255, 255, 255)) 
        
        # 繪圖順序：AI圖(如果有) -> 水墨筆觸 -> 遮罩
        if current_ai_image:
            screen.blit(current_ai_image, (0, 0))
        else:
            screen.blit(drawing_surface, (0, 0))
        
        if results.multi_hand_landmarks and not sd_gen.is_generating:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                h, w, c = img.shape
                def get_coord(idx):
                    cx, cy = int(hand_landmarks.landmark[idx].x * w), int(hand_landmarks.landmark[idx].y * h)
                    sx = int(map_range(cx, 0, w, 0, CANVAS_W))
                    sy = int(map_range(cy, 0, h, 0, CANVAS_H))
                    return sx, sy

                # 繪製手部關鍵點
                for i in range(21):
                    px, py = get_coord(i)
                    pygame.draw.circle(screen, COLOR_SKELETON, (px, py), 4)

                wrist_pos = get_coord(0)
                index_pos = get_coord(8)
                middle_pos = get_coord(12)

                if is_fist(hand_landmarks.landmark):
                    pygame.draw.circle(screen, (200, 50, 50), (wrist_pos[0], wrist_pos[1]-50), 30)
                    prev_index = None
                    prev_middle = None
                else:
                    if not current_ai_image: 
                        draw_ink_particles(drawing_surface, index_pos[0], index_pos[1], 
                                         prev_index[0] if prev_index else None, 
                                         prev_index[1] if prev_index else None, 
                                         COLOR_INK_BLACK, 12, 0.8)
                        prev_index = index_pos
                        draw_ink_particles(drawing_surface, middle_pos[0], middle_pos[1], 
                                         prev_middle[0] if prev_middle else None, 
                                         prev_middle[1] if prev_middle else None, 
                                         COLOR_INK_RED, 5, 0.3)
                        prev_middle = middle_pos
        else:
            prev_index = None
            prev_middle = None

        # 介面
        pygame.draw.rect(screen, COLOR_BTN, btn_gen_rect, border_radius=8)
        text_gen = font.render("Generate", True, COLOR_TEXT)
        screen.blit(text_gen, (btn_gen_rect.centerx - text_gen.get_width()//2, btn_gen_rect.centery - text_gen.get_height()//2))

        pygame.draw.rect(screen, COLOR_BTN, btn_clr_rect, border_radius=8)
        text_clr = font.render("Delete", True, COLOR_TEXT)
        screen.blit(text_clr, (btn_clr_rect.centerx - text_clr.get_width()//2, btn_clr_rect.centery - text_clr.get_height()//2))

        # Loading 遮罩
        if sd_gen.is_generating:
            overlay = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            
            t_load = int(time.time() * 2) % 4
            dots = "." * t_load
            load_text = big_font.render(f"Generating AI Art{dots}", True, (255, 255, 255))
            screen.blit(load_text, (CANVAS_W//2 - load_text.get_width()//2, CANVAS_H//2))

        pygame.display.flip()
        
        # 預覽視窗
        preview_w = 480
        preview_h = int(preview_w * (CAM_H / CAM_W))
        preview_img = cv2.resize(img, (preview_w, preview_h)) 
        cv2.imshow("Camera Preview (Press 'q' to quit)", preview_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
        clock.tick(60)

except KeyboardInterrupt:
    print("程式被使用者中斷")
except Exception as e:
    print(f"執行時發生錯誤: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("程式已關閉")
