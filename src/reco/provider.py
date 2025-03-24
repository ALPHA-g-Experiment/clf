import numpy as np


def normalize_spacepoints(
    batch_data, use_wireamp=False, use_cpu=True, zero_mean=True, event_type=None
):
    B, N, C = batch_data.shape
    if use_wireamp:
        assert C == 4
    else:
        assert C == 3

    l_limit_z = -1150
    u_limit_z = 1150

    l_limit_xy = -30
    u_limit_xy = 30

    l_limit_xy2 = -181
    u_limit_xy2 = 181

    # print(event_type)
    normal_data = np.zeros((B, N, C))

    if not use_cpu:
        normal_data = normal_data.cuda()

    if event_type is None:
        if not zero_mean:
            normal_data[:, :, 0] = (batch_data[:, :, 0] - l_limit_xy) / (
                u_limit_xy - l_limit_xy
            )
            normal_data[:, :, 1] = (batch_data[:, :, 1] - l_limit_xy) / (
                u_limit_xy - l_limit_xy
            )
            normal_data[:, :, 2] = (batch_data[:, :, 2] - l_limit_z) / (
                u_limit_z - l_limit_z
            )
        else:
            normal_data[:, :, 0] = (batch_data[:, :, 0]) / (u_limit_xy - l_limit_xy)
            normal_data[:, :, 1] = (batch_data[:, :, 1]) / (u_limit_xy - l_limit_xy)
            normal_data[:, :, 2] = (batch_data[:, :, 2]) / (u_limit_z - l_limit_z)

    elif event_type == "shift and scale globally":
        # Idea #2 -> Shift per event (mean) and scale globally
        # (a) Scale globally : /2300 mm
        mean_data = np.mean(batch_data, axis=1, keepdims=True)
        mean_z = mean_data[:, :, 2]
        centered_data_wrt_mean = batch_data - mean_data
        normal_data[:, :, 0] = centered_data_wrt_mean[:, :, 0] / (
            u_limit_xy - l_limit_xy
        )
        normal_data[:, :, 1] = centered_data_wrt_mean[:, :, 1] / (
            u_limit_xy - l_limit_xy
        )
        normal_data[:, :, 2] = centered_data_wrt_mean[:, :, 2] / (u_limit_z - l_limit_z)
        return normal_data, mean_z
    elif event_type == "shift only":
        mean_data = np.mean(batch_data, axis=1, keepdims=True)
        mean_z = mean_data[:, :, 2]
        centered_data_wrt_mean = batch_data - mean_data
        normal_data[:, :, 0] = centered_data_wrt_mean[:, :, 0]
        normal_data[:, :, 1] = centered_data_wrt_mean[:, :, 1]
        normal_data[:, :, 2] = centered_data_wrt_mean[:, :, 2]
        return normal_data, mean_z
    elif event_type == "scale and shift2":  # scale based on detector and shift by mean
        # Shift only z
        mean_data = np.mean(batch_data, axis=1, keepdims=True)
        mean_z = mean_data[:, :, 2]
        normal_data[:, :, 0] = batch_data[:, :, 0]
        normal_data[:, :, 1] = batch_data[:, :, 1]
        normal_data[:, :, 2] = batch_data[:, :, 2] - mean_z

        # Now scale
        normal_data[:, :, 0] = normal_data[:, :, 0] / (u_limit_xy2 - l_limit_xy2)
        normal_data[:, :, 1] = normal_data[:, :, 1] / (u_limit_xy2 - l_limit_xy2)
        normal_data[:, :, 2] = normal_data[:, :, 2] / (u_limit_z - l_limit_z)

        return normal_data, mean_z
    elif event_type == "shift only z":
        mean_data = np.mean(batch_data, axis=1, keepdims=True)
        mean_z = mean_data[:, :, 2]
        normal_data[:, :, 0] = batch_data[:, :, 0]
        normal_data[:, :, 1] = batch_data[:, :, 1]
        normal_data[:, :, 2] = batch_data[:, :, 2] - mean_z
        return normal_data, mean_z

    if use_wireamp:
        normal_data[:, :, 3] = (np.log(batch_data[:, :, 3]) - np.log(120)) / (
            np.log(5700) - np.log(120)
        )

    return normal_data


def normalize_spacepoints_target(
    target,
    use_cpu=True,
    zero_mean=True,
    event_type=None,
    mean_z=None,
    center_z_std=None,
    min_z_std=None,
    max_z_std=None,
):
    l_limit_z = -1150
    u_limit_z = 1150

    B, C = target.shape
    normal_target = np.zeros((B, C))
    # print(event_type)

    if not use_cpu:
        normal_target = normal_target.cuda()

    if event_type is None:
        if not zero_mean:
            normal_target[:, 0] = (target[:, 0] - l_limit_z) / (u_limit_z - l_limit_z)
        else:
            normal_target[:, 0] = (target[:, 0]) / (u_limit_z - l_limit_z)
    if event_type == "shift and scale globally":
        assert mean_z is not None
        normal_target = target - mean_z
        return normal_target
    elif event_type == "shift only":
        assert mean_z is not None
        normal_target = target - mean_z
    elif (
        event_type == "scale and shift2"
    ):  # scale based on detector # Rename this to shift and scale
        normal_target = target - mean_z
        normal_target[:, 0] = (normal_target[:, 0]) / (u_limit_z - l_limit_z)
        return normal_target
    elif event_type == "shift only z":
        assert mean_z is not None
        normal_target = target - mean_z

    return normal_target


def unnormalize_spacepoints_target(
    normal_target,
    use_cpu=True,
    zero_mean=True,
    event_type=None,
    mean_z=None,
    center_z_std=None,
    min_z_std=None,
    max_z_std=None,
):
    l_limit_z = -1150
    u_limit_z = 1150

    B, C = normal_target.shape
    unnormal_target = np.zeros((B, C))
    # print(event_type)
    if not use_cpu:
        unnormal_target = unnormal_target.cuda()

    if event_type is None:
        if not zero_mean:
            unnormal_target[:, 0] = (u_limit_z - l_limit_z) * (
                normal_target[:, 0]
            ) + l_limit_z
        else:
            unnormal_target[:, 0] = (u_limit_z - l_limit_z) * (normal_target[:, 0])
    if event_type == "shift and scale globally":
        assert mean_z is not None
        unnormal_target = normal_target + mean_z
    elif event_type == "shift only":
        assert mean_z is not None
        unnormal_target = normal_target + mean_z
    elif event_type == "scale and shift2":
        unnormal_target[:, 0] = (u_limit_z - l_limit_z) * (normal_target[:, 0])
        unnormal_target = unnormal_target + mean_z
    elif event_type == "shift only z":
        assert mean_z is not None
        unnormal_target = normal_target + mean_z

    return unnormal_target


def uniform_augmentation(spacepoints, target):
    assert len(spacepoints) == len(target)
    batch_size = len(spacepoints)
    random = np.random.rand(batch_size)

    l_limit_z = -1200
    u_limit_z = 1200

    uniform_spacepoints = np.copy(spacepoints)
    uniform_target = (random * (u_limit_z - l_limit_z)) + l_limit_z

    for i in range(len(spacepoints)):
        uniform_spacepoints[i, :, 2] = uniform_spacepoints[i, :, 2] + (
            uniform_target[i] - target[i]
        )

    uniform_target = uniform_target.reshape(len(uniform_target), 1)

    return (uniform_spacepoints, uniform_target)


def range_augmentation(spacepoints, target, limit_z):
    assert limit_z > 0, "limit_z must be positive!"

    aug_spacepoints = np.copy(spacepoints)
    aug_spacepoints[:, :, 2] = aug_spacepoints[:, :, 2] * (limit_z / 1000)

    aug_target = np.copy(target)
    aug_target = aug_target * (limit_z / 1000)

    return (aug_spacepoints, aug_target)


def normalize_data(batch_data):
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(
            shape_normal.reshape((-1, 3)), rotation_matrix
        )
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(
    batch_data, angle_sigma=0.06, angle_clip=0.18
):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc
