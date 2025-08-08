import rasterio
from rasterio.transform import Affine
import numpy as np
import torch

def writeTiff(path, data, im_geotrans, im_proj):
    """
    保存numpy或tensor数据为GeoTIFF格式文件

    参数:
    - path: str, 保存路径
    - data: numpy数组 或 torch tensor (二维)
    - im_geotrans: 6元素tuple，仿射变换参数 (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)
    - im_proj: 投影字符串，WKT格式或EPSG代码等

    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # data必须是二维数组
    if data.ndim != 2:
        raise ValueError("只支持二维栅格数据保存")

    height, width = data.shape

    # 构造仿射变换矩阵
    transform = Affine(
        im_geotrans[1],  # pixel width
        im_geotrans[2],  # rotation x
        im_geotrans[0],  # origin x
        im_geotrans[4],  # rotation y
        im_geotrans[5],  # pixel height (通常为负数)
        im_geotrans[3],  # origin y
    )

    # 如果im_proj为空，则默认WGS84
    if not im_proj:
        im_proj = 'EPSG:4326'

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=im_proj,
        transform=transform
    ) as dst:
        dst.write(data, 1)
