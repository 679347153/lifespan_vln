from .utility import dict_to_polygon, polygon_to_dict, inflate_region_from_center,judge_overlaps,get_room_by_id
from semantic_map import Object

def get_Robjs(obj_now: Object, obj_region_inflation_rate: float, overlap_threshold: float):
    # 根据位置获取obj_now的邻近的物体的id，邻近圈子于物体的大小有关
    # 如果放大圈子，GMM地图显示的概率则是物体
    # 同时保证获取的obj必须再room以内

    # HACK: 注意python里面是引用传递，如果为了避免引用传递导致的错误，需要深拷贝
    region = obj_now.region
    room_id = obj_now.room_id

    # 膨胀现在物体所再的region，
    # 取来与物体所在的region++重叠的物体的id
    # 计算Rcfd以及Nr

    room = get_room_by_id(room_id)

    region_plus = inflate_region_from_center(region, obj_region_inflation_rate)

    # 获取 R_objs 的id，不计算 Nr 和 Rcfd
    # NOTICE 因为这两个需要结合历史信息判断
    for obj in room.objects:
        if obj.id != obj_now.id and judge_overlaps(region_plus, overlap_threshold, obj.region):
            obj_now.add_R_objs(obj.id)

    # 更新邻近物体的 Rcfd 和 Nr
    # Nr 是相关次数，表示统计学意义上的相关性
    # 存储在 obj 中，所以找到 obj 就是 obj 与这个 Robj 的 Nr，不会乱而不清
    # Rcfd 是 obj 与这个 Robj 物体的相关置信度的值,权重表示最近又没有一起出现

    return obj_now.R_objs

def object_create():
    """
    通过信息输入建立一个新的 Object 对象。
    大概需要pcd点云、以及对应位置的物体的label、region
    关于物体的label
    """
    obj_now = Object()
    obj_now.N = 1 # 由于是新建的物体，所以N=1
