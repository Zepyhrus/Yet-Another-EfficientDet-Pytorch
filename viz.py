import shutil
from glob import glob

from tqdm import tqdm

from hie.hie import HIE
from hie.tools import GREEN



if __name__ == "__main__":
  dt = HIE('data/seed/labels/val.json', 'seed')

  for img_id in tqdm(dt.getImgIds()):
    shutil.copy(f'data/seed/images/{img_id}.jpg', f'data/seed/val/{img_id}.jpg')




