
import pickle

info_path = '/home/junfu/data/lidar_image/OpenPCDet/data/vod/lidar/kitti_infos_train.pkl'

with open(info_path, 'rb') as f:
    infos = pickle.load(f)

# 遍历所有样本，统计各类别数量
total_class_counts = {}
for k in range(len(infos)):
    info = infos[k]
    sample_idx = info['point_cloud']['lidar_idx']
    annos = info['annos']
    names = annos['name']

    # 统计各类别数量
    class_counts = {}
    for name in names:
        if name in class_counts:
            class_counts[name] += 1
        else:
            class_counts[name] = 1

        # 更新总的类别数量
        if name in total_class_counts:
            total_class_counts[name] += 1
        else:
            total_class_counts[name] = 1

    print(f"Sample {sample_idx} class counts: {class_counts}")

# 输出所有样本中各类别的总数量
print(f"Total class counts: {total_class_counts}")
