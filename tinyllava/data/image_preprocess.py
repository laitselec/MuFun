import os

# from PIL import Image, ImageFile
import torch
import ast

from ..utils.data_utils import *

# ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')
import requests
from io import BytesIO
from transformers.pipelines.audio_utils import ffmpeg_read
# import librosa
import mutagen
from torchaudio import functional as taF
import numpy as np
feature_extractor_sampling_rate = 16000
clip_length = 30*feature_extractor_sampling_rate
clip_drop = feature_extractor_sampling_rate//2
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.opus', '.ogg')

def load_audio_single(audio_file, seg=None):
    assert isinstance(audio_file, str), "audio_file should be a string"
    if audio_file.endswith(AUDIO_EXTENSIONS):
        inputs=audio_file
        in_sampling_rate=mutagen.File(inputs).info.sample_rate
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()
        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, in_sampling_rate)
        if seg is not None:
            inputs = inputs[int(seg[0] * in_sampling_rate):int(seg[1] * in_sampling_rate)]
        if in_sampling_rate != feature_extractor_sampling_rate:
            inputs = taF.resample(
                torch.from_numpy(inputs.copy()), in_sampling_rate, feature_extractor_sampling_rate
            ).numpy()
        if len(inputs) <= clip_length:
            return [inputs]
        else:
            audios = []
            for i in range(0, len(inputs), clip_length):
                chunk = inputs[i : i + clip_length]
                chunk_index = len(chunk)
                if chunk_index > clip_drop:
                    audios.append(chunk)
            return audios
    if audio_file.endswith('.npy'):
        return [np.load(audio_file)]
   

def load_audio(audio_preprocess, audio_files, segs=None, audio_folder=None):
    if audio_files is None:
        return None, None
    if isinstance(audio_files, str):
        audio_files = [audio_files]
    if segs:
        if segs and isinstance(segs[0], float):
            segs = [segs]
    else:
        segs = [None for _ in range(len(audio_files))]
    if audio_folder:
        audio_files = [os.path.join(audio_folder, afile) for afile in audio_files]
   
    def get_single_audio(audio_file, seg):
        try:
            if seg:
                audio = load_audio_single(audio_file, seg)
            else:
                audio = load_audio_single(audio_file)
                
            audio = [audio_preprocess(aud) for aud in audio]
            
        except Exception as e:
            print(f"Error loading {audio_file} seg {seg}: {e}")
            audio = None
            
        return audio
    
    audio_size= []    
    audio_list = []
    for ii in range(len(audio_files)):
        audio_file = audio_files[ii]
        seg = segs[ii]
        single_audio_list = get_single_audio(audio_file,seg)
        audio_size.append(len(single_audio_list))
        audio_list.extend(single_audio_list)
    
    return audio_list, audio_size
        
class ImagePreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ratio', None)
        self.image_processor = image_processor
        self.image_grid_pinpoints = getattr(data_args, 'image_grid_pinpoints', None)
    
    def __call__(self, image):
        if self.image_aspect_ratio == 'pad':
            image = self.expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        elif self.image_aspect_ratio == "anyres":
            image = self.process_anyres_image(image, self.image_processor, self.image_grid_pinpoints)
            return image
        elif self.image_aspect_ratio == "audio":
            # if len(image) == 1:
            #     image = image[0]
            #     return self.image_processor(image, sampling_rate=feature_extractor_sampling_rate, return_tensors="pt").input_features
            # else:
            #     return torch.cat([self.image_processor(img, sampling_rate=feature_extractor_sampling_rate, return_tensors="pt").input_features for img in image], dim=-1)
            return self.image_processor(image, sampling_rate=feature_extractor_sampling_rate, return_tensors="pt").input_features
                
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        return image

    @classmethod
    def expand2square(cls, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
        """
        Process an image with variable resolutions.

        Args:
            image (PIL.Image.Image): The input image to be processed.
            processor: The image processor object.
            grid_pinpoints (str): A string representation of a list of possible resolutions.

        Returns:
            torch.Tensor: A tensor containing the processed image patches.
        """
        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)

        patches = divide_to_patches(image_padded, processor.crop_size['height'])

        image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

        image_patches = [image_original_resize] + patches
        image_patches = [processor(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
        return torch.stack(image_patches, dim=0)
    
