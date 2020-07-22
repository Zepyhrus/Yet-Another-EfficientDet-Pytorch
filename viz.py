
from glob import glob

from hie.hie import HIE



if __name__ == "__main__":
  dt = HIE('det/origin-d0-iou-0.4-thersh-0.3.json')

  dt.viz(show_bbox=True)




