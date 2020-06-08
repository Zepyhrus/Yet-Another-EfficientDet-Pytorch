
from glob import glob


if __name__ == "__main__":
  images = glob('data/coco/train2017/*')

  print(len(images))

  print([_ for _ in images if not _.endswith('.jpg')])




