import os
import glob
from itertools import chain
import cv2
import math
import torch
import torch.utils.data as data
import umsgpack
import json
from panoptic_bev.data.transform import *
from panoptic_bev.helpers.more_util import ex_box_jaccard
from panoptic_bev.helpers.draw_gaussian import gaussian_radius, draw_umich_gaussian
from panoptic_bev.helpers.file_io import read_image, read_lines, read_panoptic_dataset_binary

def read_split(filename):
    """
    Read a list of NuScenes sample tokens
    """
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        return [val for val in lines if val != ""]

class BEVKitti360Dataset(data.Dataset):
    _IMG_DIR         = "img"
    _BEV_MSK_DIR     = "bev_msk"
    _FRONT_MSK_DIR   = "front_msk_trainid"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR         = "bev_ortho"
    _LST_DIR         = "split"
    _METADATA_FILE   = "metadata_ortho.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform, det_config= None):
        super(BEVKitti360Dataset, self).__init__()
        self.seam_root_dir = seam_root_dir
        self.kitti_root_dir = dataset_root_dir
        self.split_name = split_name
        self.transform = transform
        self.rgb_cameras = ['front']

        # Folders
        self._img_dir         = os.path.join(seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_msk_dir     = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_MSK_DIR, BEVKitti360Dataset._BEV_DIR)
        self._front_msk_dir   = os.path.join(seam_root_dir, BEVKitti360Dataset._FRONT_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._WEIGHTS_MSK_DIR)
        self._lst_dir         = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)

        # Load meta-data and split
        # self._meta, self._images, self._img_map = self._load_split()

        if "test" in split_name:
            split_folder_rel = "testing"
        else:
            split_folder_rel = "train_val"
        root = "data"

        self.images_db_path     = os.path.join(root, "kitti_360", split_folder_rel, "image")
        self.gtmaps_db_path     = os.path.join(root, "kitti_360", split_folder_rel, "seman")
        self.calib_db_path      = os.path.join(root, "kitti_360", split_folder_rel, "calib")
        self.panop_db_path      = os.path.join(root, "kitti_360", split_folder_rel, "panop")
        self.label_db_path      = os.path.join(root, "kitti_360", split_folder_rel, "label")
        self.label_dota_db_path = os.path.join(root, "kitti_360", split_folder_rel, "label_dota")
        self._front_msk_dir     = os.path.join(root, "kitti_360", split_folder_rel, "front")
        self._weights_msk_dir   = os.path.join(root, "kitti_360", split_folder_rel, "weght")
        self.split_path         = os.path.join(root, "kitti_360", "ImageSets", "{}.txt".format(split_name))

        self._images = read_split(self.split_path)
        metadata     = read_panoptic_dataset_binary(os.path.join(root,  "kitti_360", BEVKitti360Dataset._METADATA_FILE))
        self._meta = metadata["meta"]

        print("Number of images = {}".format(len(self._images)))
        if det_config is None:
            self.detector = False
        else:
            self.detector    = True
            self.det_config  = det_config
            self.det_classes = det_config.getstruct('det_classes')
            self.down_ratio  = det_config.getint('det_down_ratio')
            self.det_cat_ids = {cat:i for i,cat in enumerate(self.det_classes)}
            self.half_dim    = 0
            self.h_seman     = 704
            self.w_seman     = 768
            self.max_objs    = 50
            self.num_det_classes = len(self.det_classes)
        self.debug = False

    # Load the train or the validation split
    def _load_split(self):
        metadata = read_panoptic_dataset_binary(os.path.join(self.seam_root_dir, BEVKitti360Dataset._METADATA_FILE))
        lst      = read_lines(os.path.join(self._lst_dir, self.split_name + ".txt"))

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        front_msk_frames = os.listdir(self._front_msk_dir)
        front_msk_frames = [frame.split(".")[0] for frame in front_msk_frames]
        lst = [entry for entry in lst if entry in front_msk_frames]
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images, img_map

    def _load_item(self, item_idx):
        img_desc = self._images[item_idx]
        # scene, frame_id = img_desc["id"].split(";")
        #
        # # Get the RGB file names
        # img_file = [os.path.join(self.kitti_root_dir, self._img_map[camera]["{}.png".format(img_desc['id'])])
        #             for camera in self.rgb_cameras]
        # if all([(not os.path.exists(img)) for img in img_file]):
        #     raise IOError("RGB image not found! Scene: {}, Frame: {}".format(scene, frame_id))
        #
        # # Load the images
        # img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]
        #
        # # Load the BEV mask
        # bev_msk_file = os.path.join(self._bev_msk_dir, "{}.png".format(img_desc['id']))
        # bev_msk = [Image.open(bev_msk_file)]
        #
        # # Load the front mask
        # front_msk_file = os.path.join(self._front_msk_dir, "{}.png".format(img_desc['id']))
        # front_msk = [Image.open(front_msk_file)]
        #
        # # Load the weight map
        # weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(img_desc['id']))
        # weights_msk = cv2.imread(weights_msk_file, cv2.IMREAD_UNCHANGED).astype(float)
        # if weights_msk is not None:
        #     weights_msk_combined = (weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000
        #     weights_msk_combined = [Image.fromarray(weights_msk_combined.astype(np.int32))]
        # else:
        #     weights_msk_combined = None
        #
        # cat = img_desc["cat"]
        # iscrowd = img_desc["iscrowd"]
        # calib = img_desc['cam_intrinsic']
        # id    = img_desc["id"]
        # seman = None

        img_name        = img_desc
        calib_path      = os.path.join(self.calib_db_path , img_name + ".txt")
        image_path      = os.path.join(self.images_db_path, img_name + ".png")
        sem_seg_path    = os.path.join(self.gtmaps_db_path, img_name + ".png")
        pan_seg_path    = os.path.join(self.panop_db_path , img_name + ".png")
        label_dota_path = os.path.join(self.label_dota_db_path, img_name + ".txt")
        weght_msk_path  = os.path.join(self._weights_msk_dir, img_name + ".png")

        # Load input images
        img       = [Image.open(image_path).convert(mode='RGB')]

        # Load ground truth maps
        # White (255 or 1) is unmasked, black (0) is masked
        bev_msk   = [Image.open(pan_seg_path)]
        seman     = [Image.fromarray(cv2.rotate(read_image(sem_seg_path), cv2.ROTATE_90_CLOCKWISE)[:,:,0])]
        front_msk = None
        weights_msk_combined = None

        # weights_msk = cv2.imread(weght_msk_path, cv2.IMREAD_UNCHANGED).astype(float)
        # if weights_msk is not None:
        #     weights_msk_combined = (weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000
        #     weights_msk_combined = [Image.fromarray(weights_msk_combined.astype(np.int32))]

        bev_msk_arr    = np.array(bev_msk[0], dtype=np.int32, copy=False)
        num_objects    = np.unique(bev_msk_arr)
        cat            = [0] * num_objects
        iscrowd        = [0] * num_objects
        id             = str(img_name)

        # Load intrinsincs as shape of 3x3
        lines = read_lines(calib_path)
        obj   = lines[0].strip().split(' ')[1:]
        calib = np.array(obj, dtype=np.float32).reshape(3,4)[:, :3]

        if  self.detector:
            valid_pts = []
            valid_cat = []
            valid_dif = []
            valid_h3d = []
            valid_x3d = []
            valid_y3d = []
            valid_yaw = []
            with open(label_dota_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    obj = line.split(' ')  # list object
                    if len(obj)>8:
                        # Format: x1, z1, x2, z2, x3, z3, x4, z4, category, difficulty, alpha, x2d1, y2d1, x2d2, y2d2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d
                        #         0   1    2   3   4   5   6  7    8          9           10   11     12    13    14   15   16    17  18   19   20    21
                        points = [float(t) for t in obj[:8]]
                        points = np.array(points).reshape(4, 2)
                        # Add half_dim for z-axis points
                        points[:, 1] += self.half_dim
                        points = points.flatten()
                        x1, z1, x2, z2, x3, z3, x4, z4 = points
                        xc, yc = np.mean(points.reshape(4,2), axis= 0)
                        # We do not crop points
                        if (xc < 0 or xc >= self.w_seman  or yc < 0 or yc >= self.h_seman or obj[8] not in self.det_cat_ids):
                            continue
                        else:
                            valid_pts.append([[x1,z1], [x2,z2], [x3,z3], [x4,z4]])
                            valid_cat.append(self.det_cat_ids[obj[8]])
                            valid_dif.append(int(obj[9]))
                            h3d = float(obj[15])
                            valid_h3d.append(h3d)

                            x3d = float(obj[18])
                            y3d = float(obj[19])
                            z3d = float(obj[20])

                            valid_x3d.append(x3d)
                            valid_y3d.append(y3d)

                            valid_yaw.append(float(obj[21]))

            img_det_gt = {}
            img_det_gt['cat'] = np.asarray(valid_cat, np.int32)
            img_det_gt['dif'] = np.asarray(valid_dif, np.int32)
            img_det_gt['pts'] = np.asarray(valid_pts, np.float32)
            img_det_gt['yaw'] = np.asarray(valid_yaw, np.float32)
            img_det_gt['h3d'] = np.asarray(valid_h3d, np.float32)
            img_det_gt['y3d'] = np.asarray(valid_y3d, np.float32)
        else:
            data_dict  = {}
            img_det_gt = None

        return img, bev_msk, front_msk, weights_msk_combined, cat, iscrowd, calib.tolist(), id, seman, img_det_gt

    def generate_det_ground_truth(self, input_annotation):
        # Filter out objects first
        size_thresh = 3
        annotation  = {}
        out_rects = []
        out_cat = []
        out_h3d = []
        out_y3d = []
        out_yaw = []
        for pt_old, cat, h3d, y3d, yaw in zip(input_annotation['pts'] , input_annotation['cat'], input_annotation['h3d'], input_annotation['y3d'], input_annotation['yaw']):
            rect = cv2.minAreaRect(pt_old/self.down_ratio)
            if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                continue
            out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            out_cat.append(cat)
            out_h3d.append(h3d)
            out_y3d.append(y3d)
            out_yaw.append(yaw)
        annotation['rect'] = np.asarray(out_rects, np.float32)
        annotation['cat'] = np.asarray(out_cat, np.uint8)
        annotation['h3d'] = np.asarray(out_h3d, np.float32)
        annotation['y3d'] = np.asarray(out_y3d, np.float32)
        annotation['yaw'] = np.asarray(out_yaw, np.float32)

        # Convert to heatmap
        h_det_hm = int(self.h_seman / self.down_ratio)
        w_det_hm = int(self.w_seman / self.down_ratio)
        hm = np.zeros((self.num_det_classes, h_det_hm, w_det_hm), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        h3d = np.zeros((self.max_objs, 1), dtype=np.float32)
        y3d = np.zeros((self.max_objs, 1), dtype=np.float32)
        yaw = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        cat_ind = 255*np.ones((self.max_objs), dtype=np.int64)
        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(image, (w_det_hm, h_det_hm))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            cat_ind[k] = annotation['cat'][k]
            ind[k] = ct_int[1] * w_det_hm + ct_int[0]
            assert ind[k] <= h_det_hm * w_det_hm
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (cen_x, cen_y), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            #####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1

            # h3d, y3d, yaw
            h3d[k, 0] = annotation['h3d'][k]
            y3d[k, 0] = annotation['y3d'][k]
            yaw[k, 0] = annotation['yaw'][k]

        ret = {
           'hm': torch.tensor(hm.copy()),
           'reg_mask': torch.tensor(reg_mask.copy()),
           'ind': torch.tensor(ind.copy()),
           'cat_ind': torch.tensor(cat_ind.copy()),
           'wh': torch.tensor(wh.copy()),
           'reg': torch.tensor(reg.copy()),
           'cls_theta': torch.tensor(cls_theta.copy()),
           'h3d': torch.tensor(h3d.copy()),
           'y3d': torch.tensor(y3d.copy()),
           'yaw': torch.tensor(yaw.copy()),
           }
        return ret

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1

    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    # @property
    # def img_sizes(self):
    #     """Size of each image of the dataset"""
    #     return [img_desc["size"] for img_desc in self._images]
    #
    # @property
    # def img_categories(self):
    #     """Categories present in each image of the dataset"""
    #     return [img_desc["cat"] for img_desc in self._images]

    @property
    def dataset_name(self):
        return "Kitti360"

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img, bev_msk, front_msk, weights_msk,cat, iscrowd, calib, idx, sem_msk, det_annotations = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, front_msk=front_msk, weights_msk=weights_msk, cat=cat,
                             iscrowd=iscrowd, calib=calib, sem_msk= sem_msk, det_annotations= det_annotations)
        if self.detector:
            data_dict = self.generate_det_ground_truth(rec['det_annotations'])
            for k,v in data_dict.items():
                rec[k] = v
            del data_dict
        rec.pop('det_annotations')
        if self.debug:
            from matplotlib import pyplot as plt
            plt.subplot(211)
            plt.imshow(rec['img'].cpu().permute(1,2,0).float().numpy())
            plt.subplot(212)
            plt.imshow(torch.max(rec['hm'], dim=0)[0].cpu().float().numpy())
            plt.show()
            plt.close()

        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        if front_msk is not None:
            for m in front_msk:
                m.close()

        for k,v in rec.copy().items():
            if v is None:
                del rec[k]

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class BEVNuScenesDataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _FRONT_MSK_DIR = "front_msk_trainid"
    _VF_MSK_DIR = "vf_mask"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _METADATA_FILE = "metadata_ortho.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform):
        super(BEVNuScenesDataset, self).__init__()
        self.seam_root_dir = seam_root_dir
        self.nuscenes_root_dir = dataset_root_dir
        self.split_name = split_name
        self.transform = transform
        self.rgb_cameras = ['front']

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._BEV_MSK_DIR, BEVNuScenesDataset._BEV_DIR)
        self._front_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._FRONT_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._LST_DIR)
        self._vf_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._VF_MSK_DIR)

        # Load meta-data and split
        self._meta, self._images, self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVNuScenesDataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        bev_msk_frames = os.listdir(self._bev_msk_dir)
        bev_msk_frames = [frame.split(".")[0] for frame in bev_msk_frames]
        lst = [entry for entry in lst if entry in bev_msk_frames]
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images, img_map

    def _load_item(self, item_idx):
        img_desc = self._images[item_idx]

        # Get the RGB file names
        img_file = [os.path.join(self.nuscenes_root_dir, self._img_map[camera]["{}.png".format(img_desc['id'])])
                    for camera in self.rgb_cameras]
        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Name: {}".format(img_desc['id']))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        bev_msk_file = os.path.join(self._bev_msk_dir, "{}.png".format(img_desc['id']))
        bev_msk = [Image.open(bev_msk_file)]

        # Load the VF mask
        vf_msk_file = os.path.join(self._vf_msk_dir, "{}.png".format(img_desc["id"]))
        vf_msk = [Image.open(vf_msk_file)]

        # Load the weight map
        weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(img_desc['id']))
        weights_msk = cv2.imread(weights_msk_file, cv2.IMREAD_UNCHANGED).astype(float)
        if weights_msk is not None:
            weights_msk_combined = (weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000
            weights_msk_combined = [Image.fromarray(weights_msk_combined.astype(np.int32))]
        else:
            weights_msk_combined = None

        cat = img_desc["cat"]
        iscrowd = img_desc["iscrowd"]
        calib = img_desc['cam_intrinsic']
        return img, bev_msk, vf_msk, weights_msk_combined, cat, iscrowd, calib, img_desc["id"]

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    @property
    def dataset_name(self):
        return "nuScenes"

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img, bev_msk, vf_msk, wt_mask, cat, iscrowd, calib, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, front_msk=vf_msk, weights_msk=wt_mask, cat=cat, iscrowd=iscrowd,
                             calib=calib)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        if vf_msk is not None:
            for m in vf_msk:
                m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)
