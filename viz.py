
from glob import glob

from hie.hie import HIE
from hie.tools import GREEN



if __name__ == "__main__":
  dt = HIE('det/seed-d0-iou-0.2-thersh-0.1.json', 'seed')

  dt.viz(show_bbox=True, color=GREEN, pause=5)




