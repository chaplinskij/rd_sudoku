import os
import random
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np


class FileGenerator:
    @classmethod
    def generate(cls, output_dir='out', image_size=(28, 28), num_samples=1000, fonts=None):
        os.makedirs(output_dir, exist_ok=True)

        for digit in range(10):
            digit_dir = os.path.join(output_dir, str(digit))
            os.makedirs(digit_dir, exist_ok=True)

            for i in range(num_samples):
                font_path = random.choice(fonts)
                font = ImageFont.truetype(font_path, size=24)

                img = Image.new('L', image_size, color=255)
                draw = ImageDraw.Draw(img)

                text = str(digit) if digit else ''
                text_x, text_y = (image_size[0] / 2, image_size[1] / 2)

                # Применяем случайные эффекты
                is_bold = random.choice([True, False])
                thickness = random.randint(1, 2)
                if is_bold:
                    for offset in range(thickness):
                        draw.text((text_x + offset, text_y), text, font=font, fill=0, anchor="mm")
                        draw.text((text_x - offset, text_y), text, font=font, fill=0, anchor="mm")
                        draw.text((text_x, text_y + offset), text, font=font, fill=0, anchor="mm")
                        draw.text((text_x, text_y - offset), text, font=font, fill=0, anchor="mm")
                else:
                    draw.text((text_x, text_y), text, font=font, fill=0, anchor="mm")

                # Наклоняем изображение для имитации курсива
                is_affine = random.choice([True, False, True])
                if is_affine:
                    affine_x = random.choice([-0.2, -0.15, -0.1, 0, 0.1, 0.15, 0.2, ])
                    affine_y = random.choice([-0.2, -0.15, -0.1, 0, 0.1, 0.15, 0.2, ])
                    img = img.transform(img.size, Image.AFFINE, (1, affine_x, 0, affine_y, 1, 0), fillcolor=255)

                # добавляем границу
                is_offset = random.choice([True, False])
                if is_offset:
                    offset_x = random.randint(-2, 2)
                    offset_y = random.randint(-2, 2)
                    img = img.transform(img.size, Image.AFFINE, (1, 0, offset_x, 0, 1, offset_y), fillcolor=0)

                file_name = f"{digit}_{i}.png"
                img.save(os.path.join(digit_dir, file_name))
