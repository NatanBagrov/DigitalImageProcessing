import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

METRICS_TO_MINIMIZE = {'mse'}
METRICS_TO_MAXIMIZE = {'psnr', 'ssim'}
METRICS = METRICS_TO_MAXIMIZE | METRICS_TO_MINIMIZE


def main():
    outroot = 'results'
    dataroot = 'data'
    true_validation_directory_path = os.path.join(dataroot, 'ImageNet', 'test')
    input_validation_directory_path = os.path.join(dataroot, 'Water', 'test')

    # exp_name = 'both_L1VGGAdv'
    # exp_name = 'diff_both_L1VGGAdvDiff'
    # exp_name = 'both_L1VGGAdvInv'
    exp_name = 'both_L1VGGAdvMy'

    predicted_validation_directory_path = os.path.join(outroot, '{}_val'.format(exp_name))
    train_file_path = os.path.join(outroot, '{}_training.npz'.format(exp_name))
    evaluation_file_path = os.path.join(outroot, '{}_evaluation.npz'.format(exp_name))

    # if os.path.isfile(train_file_path):
    #     npz = dict(np.load(train_file_path))
    #
    #     for key, values in npz.items():
    #         plt.plot(values, label=key)
    #
    #     plt.show()


    if os.path.isfile(evaluation_file_path):
        npz = dict(np.load(evaluation_file_path))
        is_file_exists = np.fromiter(map(
            lambda name: os.path.isfile(os.path.join(true_validation_directory_path, name)),
            npz['paths']
        ), dtype=np.bool)

        for metric in METRICS:
            npz[metric][np.logical_not(is_file_exists)] = np.nan

        header = ['Method', ] + list(METRICS)
        row = [exp_name, ] + ['{:.3f}'.format(np.nanmean(npz[key])) for key in METRICS]
        print('\t&\t'.join(header))
        print('\t&\t'.join(row))

        for metric in METRICS:
            if metric in METRICS_TO_MINIMIZE:
                sign = -1
            else:
                sign = +1

            # sorts nan to the end
            worst_indices = np.argsort(sign * npz[metric])[:3]
            best_indices = np.argsort(-sign * npz[metric])[:3]

            if os.path.isdir(predicted_validation_directory_path):
                figure, axes = plt.subplots(3, len(worst_indices))

                for column, index in enumerate(worst_indices):
                    true_path = os.path.join(true_validation_directory_path, npz['paths'][index])
                    predicted_path = os.path.join(predicted_validation_directory_path, npz['paths'][index])
                    input_path = os.path.join(input_validation_directory_path, npz['paths'][index])
                    title = '{}\n{}: {:.3f}'.format(npz['paths'][index][len('test/'):], metric, npz[metric][index])

                    axes[0][column].set_title(title, size='small')
                    axes[0][column].imshow(Image.open(input_path).convert('RGB'))
                    axes[1][column].imshow(Image.open(predicted_path).convert('RGB'))
                    axes[2][column].imshow(Image.open(true_path).convert('RGB'))

                    for row in range(len(axes)):
                        axes[row][column].tick_params(
                            axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelbottom=False,
                            labelleft=False,
                        )

                axes[0][0].set_ylabel('Input', size='small')
                axes[1][0].set_ylabel('Output', size='small')
                axes[2][0].set_ylabel('Truth', size='small')

                file_path_to_save = os.path.join(outroot, '{}_worst_{}.png'.format(exp_name, metric))
                figure.savefig(file_path_to_save, format='png', dpi=1000)
                plt.show()


if __name__ == '__main__':
    main()
