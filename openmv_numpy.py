import math  # 导入数学库，用于数学运算

# 定义一个类 array，用于模拟 NumPy 的多维数组功能
class array:
    def __init__(self, M: list):
        """
        初始化 array 对象
        :param M: 输入的多维列表
        """
        self.M = M  # 存储数组数据
        self.shape = self.get_shape()  # 获取数组的形状
        self.ndim = len(self.shape)  # 数组的维度

    def __len__(self):
        """
        返回数组的长度（第一维的大小）
        """
        return len(self.M)

    def __getitem__(self, *args):
        """
        支持数组的索引操作
        :param args: 索引参数，可以是单个整数或元组
        :return: 索引对应的值
        """
        if isinstance(args[0], tuple):  # 如果索引是元组
            assert len(args[0]) <= self.ndim, '索引超出范围'
            indexs = list(args[0])  # 将元组转换为列表
            def get_value(a, num):
                if len(indexs) - 1 == num:  # 如果到达最后一维
                    return a[indexs[num]]
                return get_value(a[indexs[num]], num + 1)  # 递归获取值
            return get_value(self.M, 0)
        elif isinstance(args[0], int):  # 如果索引是整数
            return self.M[args[0]]

    def get_shape(self):
        """
        获取数组的形状
        :return: 数组的形状（元组）
        """
        shape = []
        def get_len(a):
            try:
                shape.append(len(a))  # 获取当前维度的长度
                get_len(a[0])  # 递归获取下一维度
            except:
                pass
        get_len(self.M)
        return tuple(shape)

    def __add__(self, other):
        """
        实现矩阵的加法
        :param other: 另一个 array 对象
        :return: 相加后的新 array 对象
        """
        assert self.ndim == 2 and other.ndim == 2 and self.shape == other.shape, '矩阵形状不匹配'
        r, w = self.shape
        return array([[self[i][j] + other[i][j] for j in range(w)] for i in range(r)])

    def __sub__(self, other):
        """
        实现矩阵的减法
        :param other: 另一个 array 对象
        :return: 相减后的新 array 对象
        """
        assert self.ndim == 2 and other.ndim == 2 and self.shape == other.shape, '矩阵形状不匹配'
        r, w = self.shape
        return array([[self[i][j] - other[i][j] for j in range(w)] for i in range(r)])

    def __mul__(self, other):
        """
        实现矩阵的乘法
        :param other: 另一个 array 对象或标量
        :return: 相乘后的新 array 对象
        """
        if isinstance(other, (int, float)):  # 如果是标量，创建对角矩阵
            other = eye(self.shape[1], other)
        assert self.ndim == 2 and other.ndim == 2, '仅支持二维矩阵相乘'
        r_a, w_a = self.shape
        r_b, w_b = other.shape
        assert w_a == r_b, '矩阵无法相乘'
        def l(i, j):
            return sum([self[i][t] * other[t][j] for t in range(w_a)])  # 计算矩阵乘积
        return array([[l(i, j) for j in range(w_b)] for i in range(r_a)])

    @property
    def T(self):
        """
        获取矩阵的转置
        :return: 转置后的新 array 对象
        """
        assert self.ndim == 2, '仅支持二维矩阵转置'
        r, w = self.shape
        return array([[self[j][i] for j in range(r)] for i in range(w)])

    def det(self):
        """
        计算矩阵的行列式
        :return: 行列式的值
        """
        shape = self.shape
        assert self.ndim == 2 and shape[0] == shape[1], '非方阵'
        r, c = shape
        m = [[self.M[i][j] for j in range(c)] for i in range(r)]
        ans = 1
        for col in range(c):
            v = [math.fabs(row[col]) for row in m[col:]]  # 获取列的绝对值
            pivot = max(v)  # 找到主元
            if pivot == 0:
                return 0
            pivot_index = v.index(pivot) + col
            pivot = m[pivot_index][col]
            pivot_row = [x / pivot for x in m[pivot_index]]
            ans *= pivot
            if pivot_index != col:  # 如果需要交换行
                m[col], m[pivot_index] = pivot_row, m[col]
                ans *= -1  # 交换行需要改变符号
            else:
                m[col] = pivot_row
            for i in range(col + 1, r):
                k = m[i][col]
                m[i] = [m[i][j] - k * pivot_row[j] for j in range(c)]
        return ans

    def inv(self):
        """
        计算矩阵的逆
        :return: 逆矩阵
        """
        shape = self.shape
        assert self.det() != 0, '方阵不可逆'
        r, c = shape
        m = [[self.M[i][j] for j in range(c)] for i in range(r)]
        I = [[1 if i == j else 0 for i in range(r)] for j in range(r)]
        for col in range(c):
            v = [math.fabs(row[col]) for row in m[col:]]
            pivot = max(v)
            if pivot == 0:
                return 0
            pivot_index = v.index(pivot) + col
            pivot = m[pivot_index][col]
            pivot_row = [x / pivot for x in m[pivot_index]]
            I_pivot_row = [x / pivot for x in I[pivot_index]]
            if pivot_index != col:
                m[col], m[pivot_index] = pivot_row, m[col]
                I[col], I[pivot_index] = I_pivot_row, I[col]
            else:
                m[col] = pivot_row
                I[col] = I_pivot_row
            for i in range(r):
                if i != col:
                    k = m[i][col]
                    m[i] = [m[i][j] - k * pivot_row[j] for j in range(c)]
                    I[i] = [I[i][j] - k * I[col][j] for j in range(c)]
        return array(I)

    @staticmethod
    def A_yu(A, I, J):
        """
        计算矩阵的余子式
        :param A: 输入矩阵
        :param I: 行索引
        :param J: 列索引
        :return: 余子式矩阵
        """
        r = len(A[0])
        M = []
        for i in range(r):
            if i != I:
                row = []
                for j in range(r):
                    if j != J:
                        row.append(A[i][j])
                M.append(row)
        return array(M)

    def __str__(self):
        """
        返回矩阵的字符串表示
        """
        return str(self.M)


# 创建单位矩阵
def eye(size, value=1):
    """
    创建单位矩阵
    :param size: 矩阵大小
    :param value: 对角线元素的值
    :return: 单位矩阵
    """
    M = [[value if i == j else 0 for i in range(size)] for j in range(size)]
    return array(M)

# 创建填充矩阵
def full(shape: tuple, value):
    """
    创建指定形状的填充矩阵
    :param shape: 矩阵形状
    :param value: 填充值
    :return: 填充矩阵
    """
    def add(m, index):
        if index < 0:
            return m
        M = []
        for _ in range(shape[index]):
            M.append(m)
        return add(M, index - 1)

    M = add([value for _ in range(shape[-1])], len(shape) - 2)
    return array(M)

# 创建零矩阵
def zeros(shape: tuple):
    """
    创建零矩阵
    :param shape: 矩阵形状
    :return: 零矩阵
    """
    return full(shape, 0)

# 创建全 1 矩阵
def ones(shape: tuple):
    """
    创建全 1 矩阵
    :param shape: 矩阵形状
    :return: 全 1 矩阵
    """
    return full(shape, 1)

# 解线性方程组
def solve(A: array, B: array) -> array:
    """
    解线性方程组 AX = B
    :param A: 系数矩阵
    :param B: 常数矩阵
    :return: 解矩阵
    """
    if A.det() == 0:
        raise ValueError("无解")
    assert B.ndim == 2 and B.shape[0] == A.shape[0] and B.shape[1] == 1, '矩阵形状不匹配'
    r, c = A.shape
    m = [[A.M[i][j] for j in range(c)] for i in range(r)]
    b = [[B.M[i][0]] for i in range(r)]
    for col in range(c):
        v = [math.fabs(row[col]) for row in m[col:]]
        pivot = max(v)
        if pivot == 0:
            return 0
        pivot_index = v.index(pivot) + col
        pivot = m[pivot_index][col]
        pivot_row = [x / pivot for x in m[pivot_index]]
        b_pivot_row = [x / pivot for x in b[pivot_index]]
        if pivot_index != col:
            m[col], m[pivot_index] = pivot_row, m[col]
            b[col], b[pivot_index] = b_pivot_row, b[col]
        else:
            m[col] = pivot_row
            b[col] = b_pivot_row
        for i in range(r):
            if i != col:
                k = m[i][col]
                m[i] = [m[i][j] - k * pivot_row[j] for j in range(c)]
                b[i] = [b[i][0] - k * b[col][0]]
    return array(b)

if __name__ == '__main__':
    print("")

