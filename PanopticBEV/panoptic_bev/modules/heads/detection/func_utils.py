import os, sys
sys.path.append(os.getcwd())
import torch
import numpy as np
np.set_printoptions   (precision= 2, suppress= True)
from panoptic_bev.data.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast
from panoptic_bev.helpers.seman_helper import box_to_params
from panoptic_bev.helpers.more_util import convertRot2Alpha, project_3d

def decode_prediction(predictions, det_category, calib):
    # predictions = predictions[0, :, :]
    # ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    # h, w, c = ori_image.shape
    h, w = 376, 1408
    half_dim = 0
    down_ratio = 4#args.seg_down_ratio * args.det_down_ratio

    pts0    = {cat: [] for cat in det_category}
    scores0 = {cat: [] for cat in det_category}
    boxes0  = {cat: [] for cat in det_category}
    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tr, br, bl, tl], np.float32)

        rot_pts  = np.asarray([bl, tl, tr, br], np.float32)
        rot_pts *= down_ratio
        rot_pts[:, 1] -= half_dim
        x3d, z3d, l3d, w3d, angle = box_to_params(rot_pts, bev_h= 704, bev_w= 768)

        h3d = pred[12]
        y3d = pred[13]

        yaw = -angle#pred[14]
        alpha = convertRot2Alpha(ry3d= yaw, z3d= z3d, x3d= x3d)
        corners2d = project_3d(calib, x3d, y3d-h3d/2.0, z3d, w3d, h3d, l3d, yaw)
        x1 = np.min(corners2d[:, 0])
        x2 = np.max(corners2d[:, 0])
        y1 = np.min(corners2d[:, 1])
        y2 = np.max(corners2d[:, 1])

        # Update projected boxes to lie within the image bounds
        x1 = np.min([np.max([x1, 0.0]), w])
        x2 = np.min([np.max([x2, 0.0]), w])
        y1 = np.min([np.max([y1, 0.0]), h])
        y2 = np.min([np.max([y2, 0.0]), h])
        boxes = np.asarray([alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, yaw])

        cat = det_category[int(clse)]
        pts0[cat].append(pts)
        scores0[cat].append(score)
        boxes0[cat].append(boxes)
    return pts0, scores0, boxes0


def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.1)
    return nms_item[keep_index], keep_index


def write_results(args,
                  model,
                  dsets,
                  down_ratio,
                  device,
                  decoder,
                  result_path,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].to(device)
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']

        with torch.no_grad():
            pr_decs = model(image)


        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)
        pts0, scores0 = decode_prediction(predictions, dsets, args, img_id, down_ratio)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        # nms
        for cat in dsets.category:
            if cat == 'background':
                continue
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)
            if pts_cat.shape[0]:
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)
        if print_ps:
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))

    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
