"""
Title: Generate an image from a text prompt using StableDiffusion
Author: fchollet
Date created: 2022/09/24
Last modified: 2022/09/24
Description: Use StableDiffusion to generate an image according to a short text
             description.
"""
from typing import List

from PIL import Image
from cog import BasePredictor, Path, Input
from tensorflow import keras

from keras_cv.models import StableDiffusion, StableDiffusionV2


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        keras.mixed_precision.set_global_policy("float32")
        # For initialization purpose
        prompt = "a happy cat on a hat"
        model_v1 = StableDiffusion(img_height=128, img_width=128, jit_compile=True)
        model_v1.text_to_image(prompt)
        model_v2 = StableDiffusionV2(img_height=128, img_width=128, jit_compile=True)
        model_v2.text_to_image(prompt)

        self.model_map = {}

    def predict(self,
                prompt: str = Input(
                    description="Image prompt."),
                image_size: str = Input(
                    description="Image size.",
                    choice=["384x512", "640x960"],
                    default="384x512"),
                num_outputs: int = Input(
                    description="Number of images to output.",
                    ge=1,
                    le=4,
                    default=1,
                ),
                num_inference_steps: int = Input(
                    description="Number of denoising steps", ge=1, le=500, default=50
                ),
                unconditional_guidance_scale: float = Input(
                    description="Scale for unconditional guidance", ge=1, le=20, default=7.5
                ),
                version: int = Input(
                    description="Stable diffusion version.",
                    choice=[1, 2],
                    default=1
                )
                ) -> List[Path]:
        """Run a single prediction on the model"""
        tks = image_size.split("x")
        img_width = int(tks[0])
        img_height = int(tks[1])
        if (img_width, img_height, version) not in self.model_map.keys():
            if version == 1:
                model = StableDiffusion(img_height=img_height, img_width=img_width, jit_compile=True)
            else:
                model = StableDiffusionV2(img_height=img_height, img_width=img_width, jit_compile=True)
            self.model_map[(img_width, img_height, version)] = model

        model = self.model_map[(img_width, img_height, version)]
        imgs = model.text_to_image(prompt, batch_size=num_outputs, num_steps=num_inference_steps,
                                   unconditional_guidance_scale=unconditional_guidance_scale)

        output_paths = []
        for idx, img in enumerate(imgs):
            output_path = f"/tmp/out-{idx}.png"
            Image.fromarray(img[0]).save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
