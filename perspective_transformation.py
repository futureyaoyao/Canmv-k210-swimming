"""
PerspectiveTransformation 模块

提供 PerspectiveTransformation 类，用于实现坐标变换相关功能：
- 计算单应性变换矩阵
- 应用单应性变换计算变换后的坐标
"""

class PerspectiveTransformation:
    """单应性变换工具类"""

    @staticmethod
    def solve_linear_system(A, B):
        """求解线性方程组 Ax = B"""
        n = len(A)
        augmented = [row[:] + [B[i]] for i, row in enumerate(A)]
        # 高斯消元
        for col in range(n):
            # 寻找主元行
            max_row = col
            for row in range(col, n):
                if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                    max_row = row
            # 交换行
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
            # 归一化主元行
            pivot = augmented[col][col]
            if pivot == 0:
                raise ValueError("矩阵不可逆，请检查输入点对是否有效")
            for j in range(col, n + 1):
                augmented[col][j] /= pivot
            # 消元下方行
            for row in range(col + 1, n):
                factor = augmented[row][col]
                for j in range(col, n + 1):
                    augmented[row][j] -= factor * augmented[col][j]

        # 回代
        x = [0] * n
        for row in reversed(range(n)):
            x[row] = augmented[row][n]
            for col in range(row + 1, n):
                x[row] -= augmented[row][col] * x[col]
        return x

    @staticmethod
    def compute_homography(src_points, dst_points):
        """
        计算单应性变换矩阵
        :param src_points: 源点坐标列表 [(x1, y1), (x2, y2), ...]
        :param dst_points: 目标点坐标列表 [(x1', y1'), (x2', y2'), ...]
        :return: 单应性变换矩阵 H
        """
        A = []
        B = []
        for (src_x, src_y), (dst_x, dst_y) in zip(src_points, dst_points):
            A.append([src_x, src_y, 1, 0, 0, 0, -src_x * dst_x, -src_y * dst_x])
            B.append(dst_x)
            A.append([0, 0, 0, src_x, src_y, 1, -src_x * dst_y, -src_y * dst_y])
            B.append(dst_y)

        params = PerspectiveTransformation.solve_linear_system(A, B)
        a, b, c, d, e, f, g, h = params
        return [
            [a, b, c],
            [d, e, f],
            [g, h, 1]
        ]

    @staticmethod
    def apply_homography(H, x, y):
        """
        应用单应性变换计算变换后的坐标
        :param H: 单应性变换矩阵
        :param x: 源点的 x 坐标
        :param y: 源点的 y 坐标
        :return: 变换后的坐标 (x', y')
        """
        denominator = H[2][0] * x + H[2][1] * y + 1
        x_transformed = (H[0][0] * x + H[0][1] * y + H[0][2]) / denominator
        y_transformed = (H[1][0] * x + H[1][1] * y + H[1][2]) / denominator
        return x_transformed, y_transformed

