import pickle

from PIL import Image
from img2vec_pytorch import Img2Vec


with open('model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = 'opencv_lessons/lesson_14/data/val/cloudy/cloudy246.jpg'
image = Image.open(image_path)

features = img2vec.get_vec(image)
pred = model.predict([features])
print(pred)
