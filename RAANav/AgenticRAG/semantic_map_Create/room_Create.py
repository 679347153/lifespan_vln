def get_room_name(room):
    # TODO 调用API给room一个名字
    return "Room Name"

def room_create_json(room):
    # 创建房间的逻辑
    # 创建了一个Room对象，它有全局唯一的room_id
    # 假设前文获取了room的图片和obj_list
    # 现在没有/需要补充获取room的名字
    # TODO 调用API给room一个名字
    # HACK 暂时不确定room.objects.label返回的是否是objects的label列表
    return room.to_json() 
    # room的to_dict返回存储的不是json格式的文件，需额外写utility函数调用
