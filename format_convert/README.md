# AI平台格式转换模块

- [x] 和Anydraw格式的转换: rect/pen/polygon/quad
- [x] 和pts格式转化: pts
- [x] 平台上4个点的多边形转换成anydraw的quad
- [x] 支持RLE格式的分割数据集(仿VOC分割格式)


## sample codes

### 和anydraw之间的转换
testbed_cvt_anydraw_platform

## 和pts之间的转换
testbed_cvt_pts_platform.py

## 和分割数据之间的转换
testbed_cvt_pts_platform.py,该脚本需要的配置文件参见 ./examples_files/segment_colors.json