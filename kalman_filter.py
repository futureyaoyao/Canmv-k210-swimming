import openmv_numpy as np  # 导入用于矩阵运算的库，类似于 NumPy，但适用于 OpenMV 环境

# 定义追踪器类，用于实现卡尔曼滤波器的目标跟踪
class Tracker:
    def __init__(self, A: np.array, H: np.array, Q: np.array, R: np.array, ID, lose_threshold=20, motion_trail_len=10,
                 cls=None):
        """
        初始化追踪器
        :param A: 状态转移矩阵
        :param H: 观测矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差矩阵
        :param ID: 追踪器的唯一标识符
        :param lose_threshold: 丢失目标的阈值（帧数）
        :param motion_trail_len: 运动轨迹的最大长度
        :param cls: 目标类别（可选）
        """
        self.A = A  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差矩阵
        self.P = np.eye(4)  # 初始化误差协方差矩阵为单位矩阵
        self.active = 0  # 活跃状态计数器，用于判断目标是否丢失
        self.lose_threshold = lose_threshold  # 丢失目标的帧数阈值
        self.updated = False  # 标志位，表示当前帧是否已更新
        self.last_X_posterior = None  # 上一帧的后验状态
        self.last_position = [None, None]  # 上一帧的位置
        self.motion_trail_measure = []  # 测量值的运动轨迹
        self.motion_trail_pre = []  # 预测值的运动轨迹
        self.motion_trail_len = motion_trail_len  # 运动轨迹的最大长度
        self.ID = ID  # 追踪器的唯一标识符
        self.cls = cls  # 目标类别

    def __call__(self, x, y, find):
        """
        更新追踪器状态
        :param x: 当前帧检测到的目标 x 坐标
        :param y: 当前帧检测到的目标 y 坐标
        :param find: 是否检测到目标
        :return: 预测的目标位置
        """
        if find:  # 如果检测到目标
            self.add_motion_trail_measure(x, y)  # 添加测量值到运动轨迹
            self.last_position = [x, y]  # 更新最后位置
            if self.active == 0:  # 如果追踪器是第一次激活
                # 初始化后验状态
                self.last_X_posterior = np.array([[x],
                                                  [y],
                                                  [0],
                                                  [0]])
                self.active = self.lose_threshold  # 设置活跃状态计数器
                self.updated = True  # 标记为已更新
                return x, y
            else:
                # 使用卡尔曼滤波器更新状态
                dt = 1  # 时间间隔
                d_x = x - self.last_X_posterior[0][0]  # x 方向速度
                d_y = y - self.last_X_posterior[1][0]  # y 方向速度
                Z_measure = np.array([[x], [y], [d_x / dt], [d_y / dt]])  # 测量值

                # 预测步骤
                self.A[0][2], self.A[1][3] = dt, dt  # 更新状态转移矩阵
                X_prior = self.A * self.last_X_posterior  # 预测状态
                P_k_prior = self.A * self.P * self.A.T + self.Q  # 预测误差协方差矩阵

                # 更新步骤
                K_k = (P_k_prior * self.H.T) * ((self.H * P_k_prior * self.H.T + self.R).inv())  # 卡尔曼增益
                self.last_X_posterior = X_prior + K_k * (Z_measure - self.H * X_prior)  # 后验状态
                self.P = (np.eye(4) - K_k * self.H) * P_k_prior  # 更新误差协方差矩阵

                # 添加预测值到运动轨迹
                x_pre = int(self.last_X_posterior[0][0])
                y_pre = int(self.last_X_posterior[1][0])
                self.add_motion_trail_pre(x_pre, y_pre)
                self.active = self.lose_threshold  # 重置活跃状态计数器
                self.updated = True  # 标记为已更新
                return x_pre, y_pre
        else:  # 如果未检测到目标
            self.active -= 1  # 减少活跃状态计数器
            self.last_X_posterior = self.A * self.last_X_posterior  # 仅预测，不更新
            x_pre = int(self.last_X_posterior[0][0])
            y_pre = int(self.last_X_posterior[1][0])
            self.add_motion_trail_pre(x_pre, y_pre)  # 添加预测值到运动轨迹
            return x_pre, y_pre

    def add_motion_trail_measure(self, x, y):
        """
        添加测量值到运动轨迹
        :param x: 测量的 x 坐标
        :param y: 测量的 y 坐标
        """
        self.motion_trail_measure.append([int(x), int(y)])
        if len(self.motion_trail_measure) >= self.motion_trail_len:
            self.motion_trail_measure.pop(0)

    def add_motion_trail_pre(self, x, y):
        """
        添加预测值到运动轨迹
        :param x: 预测的 x 坐标
        :param y: 预测的 y 坐标
        """
        self.motion_trail_pre.append([int(x), int(y)])
        if len(self.motion_trail_pre) >= self.motion_trail_len:
            self.motion_trail_pre.pop(0)

    def get_pre(self):
        """
        获取预测值（不更新状态）
        :return: 预测的 x 和 y 坐标
        """
        pres = self.A * self.last_X_posterior
        return [int(pres[0][0]), int(pres[1][0])]


# 定义追踪器管理器类，用于管理多个追踪器
class Tracker_Manager:
    def __init__(self, match_threshold=50):
        """
        初始化追踪器管理器
        :param match_threshold: 匹配距离的阈值
        """
        self.trackers = []  # 存储所有追踪器
        self.match_threshold = match_threshold  # 匹配距离阈值
        self.amount = 0  # 追踪器数量计数器

    def __len__(self):
        """
        获取当前追踪器数量
        :return: 追踪器数量
        """
        return len(self.trackers)

    def append(self, tracker: Tracker):
        """
        添加新的追踪器
        :param tracker: 追踪器对象
        """
        self.trackers.append(tracker)

    def match(self, x, y, A: np.array, H: np.array, Q: np.array, R: np.array, lose_threshold=20, motion_trail_len=20):
        """
        匹配目标到现有追踪器或创建新的追踪器
        :param x: 目标的 x 坐标
        :param y: 目标的 y 坐标
        :param A, H, Q, R: 卡尔曼滤波器参数
        :param lose_threshold: 丢失目标的帧数阈值
        :param motion_trail_len: 运动轨迹的最大长度
        """
        def get_dist(tracker, x, y):
            return ((tracker.get_pre()[0] - x) ** 2 + (tracker.get_pre()[1] - y) ** 2) ** 0.5

        dist = [get_dist(tracker, x, y) for tracker in self.trackers]  # 计算所有追踪器与目标的距离
        if len(dist):
            min_dist = min(dist)  # 找到最小距离
        else:
            min_dist = self.match_threshold + 1
        if min_dist <= self.match_threshold:  # 如果最小距离小于阈值，匹配成功
            self.trackers[dist.index(min_dist)](x, y, True)
        else:  # 匹配失败，创建新的追踪器
            print("匹配失败")
            self.amount += 1
            new_trackers = Tracker(A, H, Q, R, self.amount, lose_threshold, motion_trail_len)
            new_trackers(x, y, True)
            self.append(new_trackers)

    def update(self):
        """
        更新所有追踪器状态，并删除失效的追踪器
        """
        delate_indexs = []
        for i, tracker in enumerate(self.trackers):
            if tracker.updated:
                tracker.updated = False
            else:
                tracker(0, 0, False)
            if tracker.active == 0:  # 查找失效追踪器
                delate_indexs.append(i)
        del_num = 0
        for index in delate_indexs:  # 删除失效追踪器
            self.trackers.pop(index - del_num)
            del_num += 1

    def get_positions(self):
        """
        获取所有追踪器的后验坐标
        :return: 包含追踪器 ID 和坐标的列表
        """
        positions = []
        for tracker in self.trackers:
            x, y = tracker.last_X_posterior[0][0], tracker.last_X_posterior[1][0]
            positions.append((tracker.ID, [int(x), int(y)]))
        return positions

    def get_motion_trail_measure(self):
        """
        获取所有追踪器的测量值运动轨迹
        :return: 包含追踪器 ID 和测量轨迹的列表
        """
        return [(tracker.ID, tracker.motion_trail_measure) for tracker in self.trackers]

    def get_motion_trail_pre(self):
        """
        获取所有追踪器的预测值运动轨迹
        :return: 包含追踪器 ID 和预测轨迹的列表
        """
        return [(tracker.ID, tracker.motion_trail_pre) for tracker in self.trackers]
