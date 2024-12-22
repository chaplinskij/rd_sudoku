import os
from train.services import FileGenerator


if __name__ == '__main__':
    output_dir = 'out'
    num_samples = 1000
    image_size = (28, 28)
    fonts = []
    for dir in ["/usr/share/fonts/truetype/ubuntu/", "/usr/share/fonts/truetype/freefont/"]:
        if not os.path.exists(dir):
            continue
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.lower().endswith(".ttf"):
                    fonts.append(os.path.join(root, file))

    runner = FileGenerator.generate(
        output_dir=output_dir,
        num_samples=num_samples,
        image_size=image_size,
        fonts=fonts,
    )
