import requests
import base64
import io
import threading
import pygame

class SDGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_generating = False
        self.generated_image = None
        self.error_message = None

    def generate(self, image_bytes, canvas_w, canvas_h):
        """
        啟動一個執行緒來執行生成任務，避免卡住主程式
        """
        if self.is_generating:
            return
        
        self.is_generating = True
        self.error_message = None
        
        # 啟動執行緒
        thread = threading.Thread(
            target=self._run_api_request, 
            args=(image_bytes, canvas_w, canvas_h)
        )
        thread.start()

    def _run_api_request(self, image_bytes, canvas_w, canvas_h):
        print("🚀 [SDGenerator] 開始傳送 API 請求...")
        try:
            
            files = {
                'init_image': ('image.png', image_bytes, 'image/png'),
            }

            data = {
                'init_image_mode': 'IMAGE_STRENGTH',
                'image_strength': 0.45,
                'text_prompts[0][text]': "traditional Chinese landscape painting, convert black ink strokes into realistic majestic mountains, detailed rock textures, waterfalls cascading from peaks, pine trees growing on rocks, misty clouds, masterpiece, 8k resolution, cinematic lighting, sharp focus",
                'text_prompts[0][weight]': 1,
                'text_prompts[1][text]': "flat, blurry, abstract, messy, low quality, cartoon, simple lines",
                'text_prompts[1][weight]': -1,
                'cfg_scale': 8,
                'samples': 1,
                'steps': 35,
            }

            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }

            response = requests.post(
                'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image',
                headers=headers,
                files=files,
                data=data
            )

            if response.status_code != 200:
                raise Exception(f"API Error ({response.status_code}): {response.text}")

            result = response.json()
            base64_img = result['artifacts'][0]['base64']
            
            # 解碼圖片
            img_bytes_data = base64.b64decode(base64_img)
            img_file = io.BytesIO(img_bytes_data)
            
            # 載入並轉換為 Pygame Surface (這步必須在主執行緒使用前完成轉換)
            # 但 Pygame 的 image.load 可以在執行緒中跑，只要不操作 screen 即可
            loaded_img = pygame.image.load(img_file)
            self.generated_image = pygame.transform.smoothscale(loaded_img, (canvas_w, canvas_h))
            
            print("✅ [SDGenerator] 生成成功！")

        except Exception as e:
            print(f"❌ [SDGenerator] 錯誤: {e}")
            self.error_message = str(e)
        finally:
            self.is_generating = False

    def get_result(self):
        """
        獲取生成結果。如果還沒好，返回 None。
        如果生成完畢，返回 Surface 並清空緩存，避免重複獲取。
        """
        if self.generated_image:
            img = self.generated_image
            self.generated_image = None # 取出後清空
            return img
        return None

    def reset(self):
        self.generated_image = None
        self.is_generating = False
        self.error_message = None

