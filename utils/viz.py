import pickle 
import open3d as o3d 
import numpy as np 

def viz_results():
    labels = pickle.load(open('../labels_predicted.pickle','rb'))
    colors = np.ones((len(labels),3))
    colors = colors*labels/6.astype(np.float32)
    points = pickle.load(open('../val_cloud.pickle','rb'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = colors

    o3d.visualization.draw_geometries([pcd])