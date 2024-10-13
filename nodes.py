import os, sys
package_dir_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_dir_dir)

import cv2
from PIL import Image
import numpy as np
import torch
from .infer import video2images, infer_hamer, images2video, infer_hamer_single

def cv2tensor(img_cv):
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.uint8)
    img_pil = Image.fromarray(img_cv)
    img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)    
    img_tensor = img_tensor.unsqueeze(0)

def tensor2cv(image):
    if image.dim() == 4: image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

def cv3to4(img):
    height, width, channels = img.shape
    if channels < 4:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return new_img
    return img

def cv4to3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    new_img = img[y:y+h, x:x+w]
    return new_img 

def cv2mask(img):   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, img.dtype)

    cv2.fillPoly(mask, roi, (255,)*img.shape[2], )
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

class XX_hamer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "video_path": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "hamer_path",)
    FUNCTION = "process"
    CATEGORY = "XX-hamer"

    def process(self, video_path='', image=None):
        print(video_path)
        if video_path != '':
            out_file = os.path.join(
                os.path.dirname(video_path),
                os.path.splitext(os.path.basename(video_path))[0],
                'hamer.mp4'
                )
            print(out_file)
            if not os.path.exists(out_file):
                video2images(video_path)
                infer_hamer(video_path)
                images2video(video_path)
            return (image, out_file,)
        if image != None:
            img_outs = []
            for image_item in image:
                image_item = image_item.unsqueeze(0)
                # image: torch.Size([1, 1334, 750, 3])
                img_cv = tensor2cv(image_item)   # img_cv: (1334, 750, 3)            
                img_cv = infer_hamer_single(img_cv) #(1334, 750, 4)

                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR) #(1334, 750, 4)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.uint8)

                img_pil = Image.fromarray(img_cv)

                img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)
                img_tensor = img_tensor.unsqueeze(0)
                img_outs.append(img_tensor) 
            return (torch.cat(img_outs, dim=0), '',)

NODE_CLASS_MAPPINGS = {    
    "XX_hamer":XX_hamer,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "XX hamer":"XX_hamer",
}