"""
Following script can be used for compressing images without quality loss.
This script can also be used used as png to jpg converter
"""
from PIL import Image

# get the image
# add the png image path here
image = Image.open('utility\demo_files\mario.png')

# image details
print(f"Images read is in {image.mode} and the size is {image.size}")


rgb_image = image.convert('RGB')
rgb_image.save('utility\demo_files\mario.jpg')  # add the jpg image path here
