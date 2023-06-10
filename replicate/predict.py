import os
from typing import List

import profanity_check
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from cog import BasePredictor, Path, Input
from tensorflow import keras

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

from keras_cv.models import StableDiffusion, StableDiffusionV2


def preprocess_image(image):
    """ Loads image from path and preprocesses to make it model ready
        Args:
          image: image numpy value.
    """
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if image.shape[-1] == 4:
        image = image[..., :-1]
    hr_size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def save_image(image, filename):
    """
      Saves unscaled Tensor Images.
      Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s" % filename)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        keras.mixed_precision.set_global_policy("float32")
        # For initialization purpose
        prompt = "a happy cat on a hat"
        model_v1 = StableDiffusion(img_height=128, img_width=128, jit_compile=True)
        model_v1.text_to_image(prompt)
        # TODO: Can create separate service for SDV2 to make the initialization faster.
        model_v2 = StableDiffusionV2(img_height=128, img_width=128, jit_compile=True)
        model_v2.text_to_image(prompt)

        self.model_map = {}
        self.upscale_model = hub.load(SAVED_MODEL_PATH)

    def predict(self,
                prompt: str = Input(
                    description="Image prompt."),
                image_size: str = Input(
                    description="Image size.",
                    choices=["384x512", "640x960"],
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
                    choices=[1, 2],
                    default=1
                ),
                upscale: bool = Input(
                    description="Whether to upscale the image by 4 times",
                    choices=[False, True],
                    default=False
                ),
                ) -> List[Path]:
        """Run a single prediction on the model"""
        is_profane = profanity_check.predict([prompt])
        if is_profane:
            raise Exception("The service detects profanity in your prompt, please check and rewrite your prompt.")

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
            if upscale:
                img = preprocess_image(img)
                upscale_image = self.upscale_model(img)
                save_image(tf.squeeze(upscale_image), output_path)
            else:
                Image.fromarray(img).save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
