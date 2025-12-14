import replicate
import threading
import pygame
import io
import requests
import os

class SDGenerator:
    def __init__(self, api_key=None):
        # Replicate 套件會自動讀取 os.environ["REPLICATE_API_TOKEN"]
        # 所以這裡不需要特別存 api_key，除非你想手動設定
        self.is_generating = False
        self.generated_image = None
        self.error_message = None

    def generate(self, image_bytes, canvas_w, canvas_h):
        if self.is_generating:
            return
        
        self.is_generating = True
        self.error_message = None
        
        thread = threading.Thread(
            target=self._run_api_request, 
            args=(image_bytes, canvas_w, canvas_h)
        )
        thread.start()

    def _run_api_request(self, image_bytes, canvas_w, canvas_h):
        print("🚀 [Replicate] 開始傳送 API 請求...")
        try:
            # 1. Replicate 需要圖片是一個檔案物件或 URL
            # image_bytes 已經是 BytesIO，可以直接用
            
            # 2. 設定參數
            # 使用 SDXL 模型
            model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
            
            input_data = {
                "image": image_bytes,
                "prompt": "traditional Chinese landscape painting, convert black ink strokes into realistic majestic mountains, detailed rock textures, waterfalls cascading from peaks, pine trees growing on rocks, misty clouds, masterpiece, 8k resolution, cinematic lighting, sharp focus",
                "negative_prompt": "flat, blurry, abstract, messy, low quality, cartoon, simple lines",
                "strength": 0.45, # 控制重繪幅度 (img2img)
                "guidance_scale": 7.5,
                "num_inference_steps": 25, # Replicate 通常可以設稍高一點
                "refine": "expert_ensemble_refiner", # SDXL Refiner 增強細節
                "high_noise_frac": 0.8
            }

            # 3. 呼叫 API (這會阻塞直到完成)
            output = replicate.run(
                model,
                input=input_data
            )
            
            # output 通常是一個圖片 URL 列表 ['https://...']
            if output and len(output) > 0:
                image_url = output[0]
                print(f"✅ 生成成功！下載圖片中... ({image_url})")
                
                # 4. 下載圖片
                resp = requests.get(image_url)
                if resp.status_code == 200:
                    img_data = io.BytesIO(resp.content)
                    loaded_img = pygame.image.load(img_data)
                    self.generated_image = pygame.transform.smoothscale(loaded_img, (canvas_w, canvas_h))
                else:
                    raise Exception("無法下載生成的圖片")
            else:
                raise Exception("Replicate 沒有回傳圖片")

        except Exception as e:
            print(f"❌ [Replicate] 錯誤: {e}")
            self.error_message = str(e)
        finally:
            self.is_generating = False

    def get_result(self):
        if self.generated_image:
            img = self.generated_image
            self.generated_image = None
            return img
        return None

    def reset(self):
        self.generated_image = None
        self.is_generating = False
        self.error_message = None
