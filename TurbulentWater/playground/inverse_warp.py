import numpy as np
import torch
import torch.sparse


def inverse_warp(im, di, dj):
    """
    Get observation from original/predicted image

    :param im: [num_batch, channels, height, width]
    :param di: [num_batch, height, width]
    :param dj: [num_batch, height, width]
    :return: [num_batch, channels, height, width]
    """

    num_batch, channels, height, width = im.shape
    i, j = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing='ij')
    i = torch.from_numpy(i).unsqueeze(0).expand_as(im[:, 0, :, :])
    j = torch.from_numpy(j).unsqueeze(0).expand_as(im[:, 0, :, :])
    ii = (i.float() + di).round().clamp(0, height-1).long()
    jj = (j.float() + dj).round().clamp(0, width - 1).long()

    jm = torch.zeros_like(im)
    jm[:, :, ii, jj] = im[:, :, i, j]

    return jm


def direct_warp(jm, di, dj):
    im = np.zeros_like(jm, dtype=np.float)
    count = np.zeros_like(di)

    for k in range(di.shape[0]):
        for i in range(di.shape[1]):
            for j in range(di.shape[2]):
                im[k, :, i, j] += jm[k, :, i + di[k, i, j], j + dj[k, i, j]]
                count[k, i, j] += 1

    np.divide(im, count, out=im, where=count != 0)

    return im


if __name__ == '__main__':
    jm_expected = np.arange(1 * 1 * 2 * 3).reshape((1, 1, 2, 3))
    dj = [[
        [+2, 0, -2],
        [+2, 0, -2],
    ]]
    di = [[
        [0, 1, 1],
        [0, -1, -1],
    ]]
    di = np.array(di)
    dj = np.array(dj)
    im = direct_warp(jm_expected, di, dj)
    im = torch.from_numpy(im)
    di = torch.from_numpy(di).float()
    dj = torch.from_numpy(dj).float()
    jm_actual = inverse_warp(im, di, dj)

    torch.testing.assert_allclose(jm_actual, jm_expected)
