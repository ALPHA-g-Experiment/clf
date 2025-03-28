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
