"""
    Sample Run:
    python tools/parse_nuscenes_log.py --str "p.."
    python tools/parse_nuscenes_log.py --log file_path
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)
from mmdet3d_plugin.helpers.file_io import read_lines, read_json

def parse_log(data, use_json= False):
    classes       = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    large_classes = np.arange(1, 5)
    small_classes = np.arange(5, 10)
    dist_thresh   = ['0.5','1.0','2.0','4.0']


    if use_json:
        prefix  = 'metrics_summary'
        mid_fix = 'label_aps'
    else:
        prefix  = 'pts_bbox_NuScenes/'
        mid_fix = '_AP_dist_'

        key_list = []
        for cls in classes:
            for dist in dist_thresh:
                key_list.append(prefix + cls + mid_fix + dist + ':')

        # Split by ","
        key_value_pairs = data.split(",")
        data_parsed = {}
        for key_val_pair in key_value_pairs:
            temp = key_val_pair.split(" ")
            if len(temp) == 3:
                key, val = temp[1], temp[2]
            else:
                key, val = temp[0], temp[1]
            if key in key_list or key== "pts_bbox_NuScenes/mAP:" or key== "pts_bbox_NuScenes/NDS:":
                data_parsed[key] = float(val)
                pass


    ap_performance = np.zeros((len(dist_thresh), len(classes)))
    for j, cls in enumerate(classes):
        for i, dist in enumerate(dist_thresh):
            if use_json:
                ap_performance[i, j] = data[prefix][mid_fix][cls][dist]
            else:
                key = prefix + cls + mid_fix + dist + ':'
                ap_performance[i, j] = data_parsed[key]

    print("")
    print(classes)
    # AP of each class
    ap_class = np.mean(ap_performance, axis=0)
    ap_class_list = ["{:f}".format(i) for i in ap_class]
    print(",".join(ap_class_list))

    ap_large = np.mean(ap_class[large_classes])
    ap_small = np.mean(ap_class[small_classes])
    print("APlarge= {:.2f}".format(100.0 * ap_large))
    print("APcar  = {:.2f}".format(100.0 * ap_class[0]))
    print("APsmall= {:.2f}".format(100.0 * ap_small))
    map_calc = np.mean(ap_performance)
    if use_json:
        map = data['result']['mAP']
        nds = data['result']['NDS']
    else:
        map = data_parsed["pts_bbox_NuScenes/mAP:"]
        nds = data_parsed["pts_bbox_NuScenes/NDS:"]
    assert(np.abs(map_calc - map) < 0.01)
    print("mAP    = {:.2f}".format(100.0* map))
    print("NDS    = {:.2f}".format(100.0* nds))

if __name__ == '__main__':
    #================================================================
    # Main starts here
    #================================================================
    import argparse
    parser = argparse.ArgumentParser(description='implementation of GUPNet')
    parser.add_argument('--log', type=str, default= "nusc_log/detr3d.txt", help='one of kitti,nusc_kitti,nuscenes,waymo')
    parser.add_argument('--str', type=str, default= None, help='log to parse')
    args = parser.parse_args()

    use_json= False
    if args.str is not None:
        data = args.str
    else:
        if ".json" in args.log:
            data = read_json(args.log)
            use_json = True
        else:
            data = read_lines(args.log)[0]

    # Parse data
    parse_log(data, use_json= use_json)