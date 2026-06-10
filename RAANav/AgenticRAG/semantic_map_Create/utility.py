from shapely.geometry import Polygon
from typing import List, Dict, Optional

def get_room_by_id(room_id):
    # TODO: 实现根据 room_id 获取房间信息的逻辑
    pass

# 假设这两个函数已经存在，它们负责字典和shapely.Polygon之间的转换
# 这是一个简单的模拟，以使代码可运行
def dict_to_polygon(region: List[Dict]) -> Polygon:
    return Polygon([(p['x'], p['y']) for p in region])

def polygon_to_dict(polygon: Polygon) -> List[Dict]:
    return [{"x": x, "y": y} for x, y in polygon.exterior.coords]

def inflate_region_from_center(region: List[Dict], inflation_rate: float) -> List[Dict]:
    """
    从中心点向外膨胀一个多边形区域。

    Args:
        region: 包含多边形顶点的字典列表。
        inflation_rate: 膨胀率，1.0为无变化，>1为膨胀，<1为收缩。

    Returns:
        List[Dict]: 膨胀后的新多边形顶点字典列表。
    """
    # 1. 将字典列表转换为 Shapely 多边形对象
    original_polygon = dict_to_polygon(region)
    
    # 2. 获取多边形的中心点（质心）
    # shapely 的 .centroid 属性会自动计算质心
    centroid = original_polygon.centroid

    # 3. 创建一个新的顶点列表，用于存放膨胀后的点
    inflated_coords = []
    
    # 4. 遍历原始多边形的所有顶点
    for x, y in original_polygon.exterior.coords:
        # a. 计算从中心点到当前顶点的向量
        vector_x = x - centroid.x
        vector_y = y - centroid.y
        
        # b. 按膨胀率缩放这个向量
        inflated_vector_x = vector_x * inflation_rate
        inflated_vector_y = vector_y * inflation_rate
        
        # c. 将缩放后的向量加回中心点，得到新的顶点坐标
        new_x = centroid.x + inflated_vector_x
        new_y = centroid.y + inflated_vector_y
        
        inflated_coords.append((new_x, new_y))

    # 5. 使用新的顶点创建膨胀后的多边形
    inflated_polygon = Polygon(inflated_coords)

    # 6. 将新的多边形转换回字典列表格式并返回
    return polygon_to_dict(inflated_polygon)

def judge_overlaps(inflated_region: List[Dict], overlap_threshold: float, obj_region: List[Dict]) -> bool:
    """
    判断obj_region与inflated_region的重叠部分是否占inflated_region的overlap_threshold
    """
    # 1. 将字典列表转换为 Shapely 多边形对象
    inflated_polygon = dict_to_polygon(inflated_region)
    obj_polygon = dict_to_polygon(obj_region)

    # 2. 计算重叠区域
    intersection = inflated_polygon.intersection(obj_polygon)

    # 3. 判断重叠部分是否占据了足够的比例
    if intersection.is_empty:
        return False

    overlap_area = intersection.area
    inflated_area = inflated_polygon.area

    return overlap_area / inflated_area > overlap_threshold
