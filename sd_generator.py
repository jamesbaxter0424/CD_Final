import replicate
import threading
import pygame
import io
import requests
import os

class SDGenerator:
    def __init__(self, api_key=None):
        # Replicate package automatically reads os.environ["REPLICATE_API_TOKEN"]
        # So we don't need to store api_key explicitly, unless manually setting it
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
        print("🚀 [Replicate] Starting API request...")
        try:
            # 1. Replicate requires image as file object or URL
            # image_bytes is already BytesIO, can be used directly
            
            # 2. Set parameters
            # Use SDXL model
            model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
            
            input_data = {
                "image": image_bytes,
                "prompt": "traditional Chinese landscape painting, convert black ink strokes into realistic majestic mountains, detailed rock textures, waterfalls cascading from peaks, pine trees growing on rocks, misty clouds, masterpiece, 8k resolution, cinematic lighting, sharp focus",
                "negative_prompt": "flat, blurry, abstract, messy, low quality, cartoon, simple lines",
                "strength": 0.45, # Control redrawing strength (img2img)
                "guidance_scale": 7.5,
                "num_inference_steps": 25, # Replicate usually allows slightly higher
                "refine": "expert_ensemble_refiner", # SDXL Refiner enhances details
                "high_noise_frac": 0.8
            }

            # 3. Call API (This blocks until completion)
            output = replicate.run(
                model,
                input=input_data
            )
            
            # output is usually a list of image URLs ['https://...']
            if output and len(output) > 0:
                image_url = output[0]
                print(f"✅ Generation successful! Downloading image... ({image_url})")
                
                # 4. Download image
                resp = requests.get(image_url)
                if resp.status_code == 200:
                    img_data = io.BytesIO(resp.content)
                    loaded_img = pygame.image.load(img_data)
                    self.generated_image = pygame.transform.smoothscale(loaded_img, (canvas_w, canvas_h))
                else:
                    raise Exception("Failed to download generated image")
            else:
                raise Exception("Replicate did not return any image")

        except Exception as e:
            print(f"❌ [Replicate] Error: {e}")
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
