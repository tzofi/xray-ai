from PIL import Image
import sys

background = Image.open(sys.argv[1])
overlay = Image.open(sys.argv[2])

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.30)
new_img.save("severe.png","PNG")
