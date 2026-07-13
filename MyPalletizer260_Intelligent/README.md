# 手写公式识别计算系统 - MyPalletizer 260 版本

基于 MyPalletizer 260 机械臂和 PaddleOCR 的手写数学公式识别与自动求解系统。系统能够识别手写算式、自动计算结果，并使用机械臂抓取对应的答案卡片。

> 本项目是从 MyCobot 280 版本移植而来。MyPalletizer 260 使用 **4 个关节** (J1, J2, J3, J4) 和 **4 元素坐标** [x, y, z, θ]（不同于 280 的 6 关节 / 6 元素坐标）。

## 功能特点

- ✅ 使用 PaddleOCR 识别手写数字和运算符（+、-、×、÷）
- ✅ 支持运算符优先级（先乘除后加减）
- ✅ 支持多位数识别（如 16、25 等）
- ✅ 平面映射标定，实现精确的像素到机械臂坐标转换
- ✅ 自动抓取答案卡片并放置到指定位置
- ✅ 双区域工作模式（A区域识别算式，B区域抓取答案）

## 系统要求

### 硬件
- MyPalletizer 260 机械臂
- USB 摄像头
- 吸泵套件（用于抓取卡片）
- 数字卡片（0-9）

### 软件
- Python 3.8+
- Windows/Linux 操作系统

## 安装依赖

```bash
pip install opencv-python numpy pymycobot paddlepaddle paddleocr
```

## 文件说明

```
.
├── math_solver_handwriting.py      # 主程序
├── uvc_camera.py                   # 相机模块
├── calibrate_plane_mapping.py      # A区域标定脚本
├── calibrate_plane_mapping_B.py    # B区域标定脚本
├── camera_params.npz               # 相机标定参数
├── plane_mapping.json              # A区域平面映射数据（运行标定生成）
├── plane_mapping_B.json            # B区域平面映射数据（运行标定生成）
└── math_solver_config.json         # 位置配置文件
```

## 使用步骤

### 1. 相机标定（首次使用）

如果没有 `camera_params.npz` 文件，需要先进行相机标定。

### 2. A区域平面映射标定

A区域用于识别手写算式。

```bash
python calibrate_plane_mapping.py
```

**标定步骤：**
1. 将棋盘格（8列×6行，内角点7×5）放置在A区域
2. 手动移动机械臂到观测位置
3. 按提示记录4个角点的像素坐标和机械臂坐标
4. 标定数据保存到 `plane_mapping.json`

### 3. B区域平面映射标定

B区域用于抓取答案卡片。

```bash
python calibrate_plane_mapping_B.py
```

标定步骤与A区域相同，数据保存到 `plane_mapping_B.json`。

### 4. 配置位置参数

编辑 `math_solver_config.json`，设置机械臂的关键位置。**注意：MyPalletizer 260 有 4 个关节，所以角度列表长度为 4**：

```json
{
  "POSITION_A": [j1, j2, j3, j4],
  "POSITION_B": [j1, j2, j3, j4],
  "PLACE_ABOVE": [x, y, z, θ],
  "PLACE_DOWN":  [x, y, z, θ]
}
```

- `POSITION_A`: A区域观测位置（角度模式）
- `POSITION_B`: B区域观测位置（角度模式）
- `PLACE_ABOVE`: 放置点上方（坐标模式）
- `PLACE_DOWN`: 放置点（坐标模式）

### 5. 运行主程序

```bash
python math_solver_handwriting.py
```

**工作流程：**
1. 机械臂移动到A区域，识别手写算式
2. 自动计算结果（支持运算符优先级）
3. 移动到B区域，搜索答案数字卡片
4. 抓取答案卡片
5. 放置到指定位置
6. 返回A区域

## 配置说明

### 串口配置

在 `math_solver_handwriting.py` 中修改串口：

```python
mc = MyPalletizer260("COM6")           # Windows
# mc = MyPalletizer260("/dev/ttyUSB0")  # Linux
```

### 吸泵引脚

- 引脚 5: 吸泵阀门
- 引脚 2: 气体释放

### 识别参数

```python
confidence_threshold = 0.5   # 置信度阈值（0.0-1.0）
max_attempts = 15            # B区域最大搜索次数
pick_height = 70             # mm，抓取高度
observe_height = 180         # mm，观测高度
```

## 运算符支持

| 符号 | 识别 | 说明 |
|------|------|------|
| + | ✅ | 加法 |
| - | ✅ | 减法 |
| × | ✅ | 乘法（也识别 x, X, *） |
| ÷ | ✅ | 除法 |

**运算优先级：** 先乘除，后加减（符合数学规则）

**示例：**
- `2+3*4` = 14
- `10-6/2` = 7

## 与 280 版本的区别

| 项目 | MyCobot 280 | MyPalletizer 260 |
|------|-------------|------------------|
| 关节数 | 6 (J1–J6) | 4 (J1–J4) |
| 坐标模式 | [x, y, z, rx, ry, rz] | [x, y, z, θ] |
| `send_coords` 参数 | `(coords, speed, mode)` | `(coords, speed)`（无 mode） |
| 类名 | `MyCobot280` | `MyPalletizer260` |

## 故障排除

### 1. 机械臂不移动或卡住

**原因：** 姿态角不兼容导致逆运动学失败

**解决：** 代码已添加 `wait_move_safe()` 超时保护，10秒后自动继续

### 2. 识别不到字符

**检查：**
- 光照是否充足
- 手写是否清晰
- 相机焦距是否对准
- 降低 `confidence_threshold` 阈值

### 3. 抓取位置不准

**解决：**
- 重新进行平面映射标定
- 检查标定时的观测高度是否一致
- 调整偏移参数（在 `pick_digit()` 函数中）

### 4. 下降不到位

**检查：**
- 检查目标高度是否在机械臂工作范围内（260 的 Z 范围比 280 小）
- 查看终端输出的实际高度

## 技术细节

### 平面映射原理

使用单应性矩阵（Homography Matrix）将相机像素坐标转换为机械臂坐标：

```
[x_robot]       [pixel_x]
[y_robot] = H × [pixel_y]
[   1   ]       [   1   ]
```

### PaddleOCR 配置

```python
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=False,
    show_log=False
)
```

### 坐标系说明

- **像素坐标系：** 原点在图像左上角，X向右，Y向下
- **机械臂坐标系：** 原点在机械臂基座，X/Y/Z为笛卡尔坐标
- **角度模式：** 4个关节角度 [J1, J2, J3, J4]
- **坐标模式：** [X, Y, Z, θ]

## 许可证

MIT License

---

**注意：** 使用机械臂时请注意安全，确保工作区域内无障碍物和人员。
