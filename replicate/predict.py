import os
from typing import List

import profanity_check
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from cog import BasePredictor, Path, Input
from tensorflow import keras

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
UPSCALE_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
DIFFUSION_MODEL_PATH = "https://huggingface.co/Elvenson/stable_diffusion_weights/resolve/main/fine_tune_exp1.h5"

from keras_cv.models import StableDiffusion


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
        # For initialization purpose
        prompt = "a happy cat on a hat"
        self.model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        diffusion_model_path = keras.utils.get_file(
            origin=DIFFUSION_MODEL_PATH,
        )
        self.model.diffusion_model.load_weights(diffusion_model_path)
        self.model.text_to_image(prompt, num_steps=1)
        self.upscale_model = hub.load(UPSCALE_MODEL_PATH)

    def predict(self,
                prompt: str = Input(
                    description="Image prompt."),
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
                upscale: bool = Input(
                    description="Whether to upscale the image by 4 times",
                    default=False
                ),
                ) -> List[Path]:
        """Run a single prediction on the model"""
        is_profane = profanity_check.predict([prompt])
        if is_profane:
            raise Exception("The service detects profanity in your prompt, please check and rewrite your prompt.")

        imgs = self.model.text_to_image(prompt, batch_size=num_outputs, num_steps=num_inference_steps,
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
