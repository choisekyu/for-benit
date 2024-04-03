from pathlib import Path
from multiprocessing import Process
import numpy as np
from PIL import Image
from tqdm import tqdm
from imageprocessing import COCO


LABELS = {
    'crosswalk': (8, 23),
    'human': (19, 20, 21, 22),
    #'lane': (24,), # lane marking
    'traffic_light': (48,),
    'traffic_sign': (49,),
    'bicycle': (52,),
    'bus': (54,),
    'car': (55,),
    'motorcycle': (57,),
    'vehicle-unknown': (58, 59, 56, 60, 61),
    }
CATEGORIES = [
    {'supercategory': 'pedestrian', 'id': 1, 'name': 'pedestrian'},
    {'supercategory': 'rider', 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'rider', 'id': 3, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 5, 'name': 'bus'},
    {'supercategory': 'vehicle', 'id': 6, 'name': 'vehicle else'},
    #{'supercategory': 'lane', 'id': 7, 'name': 'lane'},
    {'supercategory': 'traffic light', 'id': 8, 'name': 'traffic light'},
    {'supercategory': 'traffic sign', 'id': 9, 'name': 'traffic sign'},
    {'supercategory': 'road mark', 'id': 10, 'name': 'crosswalk'}
    ]
LABELCAT = {
    'crosswalk': 10,
    'human': 1,
    'lane': 7,
    'traffic_light': 8,
    'traffic_sign': 9,
    'bicycle': 2,
    'bus': 5, 
    'car': 4,
    'motorcycle': 3,
    'vehicle-unknown': 6,
    }
coco = COCO()


def _main(pos: int, path: list, cpu_num: int):
    image_path = list((path / 'images').glob('*.jpg'))
    image_path.sort()
    num = len(image_path) // cpu_num
    image_path = image_path[pos*num:(pos+1)*num if pos+1 < cpu_num else None]
    for cnt, p in enumerate(tqdm(image_path, desc=str(pos), position=pos)):
        name = p.stem
        # load image
        label_image = Image.open(path / 'labels' / f'{name}.png')
        instance_image = Image.open(path / 'instances' / f'{name}.png')

        # convert labeled data to numpy arrays for better handling
        label_image = np.array(label_image)
        instance_image = np.array(instance_image, dtype=np.uint16)

        # now we split the instance_image into labels and instatnce ids
        instance_label = np.array(instance_image / 256, dtype=np.uint8)
        instance_id = np.array(instance_image % 256, dtype=np.uint8)

        coco.add_image({
            'license': 0,
            'file_name': p.name,
            'coco_url': '',
            'height': label_image.shape[0],
            'width': label_image.shape[1],
            'date_captured': '', 'flickr_url': '', 'id': pos*num+cnt})
        results = []
        for name, label in LABELS.items():
            for l in label:
                masks = np.where(instance_label == l)
                etc = np.where(instance_label != l)
                if masks[0].size != 0:
                    temp = instance_id.copy()
                    temp[etc] = -1
                    for id_ in np.unique(instance_id[masks]):
                        results.append(
                            (LABELCAT[name], np.where(temp == id_, 1, 0)))
        coco.add_annotation(results, pos*num+cnt)


def main(path: Path, cpu_num: int = 4):
    if isinstance(path, str):
        path = Path(path)
    if cpu_num == 1:
        _main(0, path, 1)
    else:
        procs = []
        for pos in range(cpu_num):
            p = Process(target=_main, args=(pos, path, cpu_num))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()


if __name__ == '__main__':
    coco.set_info({
        'description': 'Mapillary Vistas Dataset',
        'url': 'https://www.mapillary.com/',
        'version': '1.2',
        'year': 2020,
        'contributor': 'Mapillary AB',
        'date_created': '2020'})
    coco.add_license({'url': '', 'id': 0, 'name': ''})
    [coco.add_category(cat) for cat in CATEGORIES]
    for name in ['validation', 'training']:
        path = Path(f'/repo/data/mapillary-vistas-dataset_public_v1.2/{name}')
        main(path, cpu_num=1)
        coco.save((path.parent/'annotations'/f'{name}.json').as_posix())
        coco.reset()
