import argparse
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim, compare_mse

import synthdata
import networks
import pairedtransforms
from datasets import ImageFolder, PairedImageFolder
from globals import ROOT_PATH, cuda_if_available, map_location
import Interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default=os.path.join(ROOT_PATH, 'data'), help='path to images')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
    parser.add_argument('--outroot', default='./results', help='path to save the results')
    parser.add_argument('--exp-name', default='test', help='name of expirement')
    parser.add_argument('--load', default='', help='name of pth to load weights from')
    parser.add_argument('--freeze-cc-net', dest='freeze_cc_net', action='store_true',
                        help='dont train the color corrector net')
    parser.add_argument('--freeze-warp-net', dest='freeze_warp_net', action='store_true',
                        help='dont train the warp net')
    parser.add_argument('--test', dest='test', action='store_true', help='only test the network')
    parser.add_argument('--synth-data', dest='synth_data', action='store_true',
                        help='use synthetic data instead of tank data for training')
    parser.add_argument('--epochs', default=3, type=int, help='number of epochs to train for')
    parser.add_argument('--no-warp-net', dest='warp_net', action='store_false',
                        help='do not include warp net in the model')
    parser.add_argument('--warp-net-downsample', default=3, type=int, help='number of downsampling layers in warp net')
    parser.add_argument('--no-color-net', dest='color_net', action='store_false',
                        help='do not include color net in the model')
    parser.add_argument('--color-net-downsample', default=3, type=int,
                        help='number of downsampling layers in color net')
    parser.add_argument('--no-color-net-skip', dest='color_net_skip', action='store_false',
                        help='dont use u-net skip connections in the color net')
    parser.add_argument('--dim', default=32, type=int,
                        help='initial feature dimension (doubled at each downsampling layer)')
    parser.add_argument('--n-res', default=8, type=int, help='number of residual blocks')
    parser.add_argument('--norm', default='gn', type=str, help='type of normalization layer')
    parser.add_argument('--denormalize', dest='denormalize', action='store_true',
                        help='denormalize output image by input image mean/var')
    parser.add_argument('--weight-X-L1', default=1., type=float,
                        help='weight of L1 reconstruction loss after color corrector net')
    parser.add_argument('--weight-Y-L1', default=1., type=float, help='weight of L1 reconstruction loss after warp net')
    parser.add_argument('--weight-Y-VGG', default=1., type=float, help='weight of perceptual loss after warp net')
    parser.add_argument('--weight-Z-L1', default=1., type=float,
                        help='weight of L1 reconstruction loss after color net')
    parser.add_argument('--weight-Z-VGG', default=.5, type=float, help='weight of perceptual loss after color net')
    parser.add_argument('--weight-Z-Adv', default=0.2, type=float, help='weight of adversarial loss after color net')
    parser.add_argument('--weight-Z-invertability', default=0.0, type=float, help='weight of invertability loss')
    parser.add_argument('--weight-diff', default=0.0, type=float, help='weight of differentiability loss')
    args = parser.parse_args()

    # set random seed for consistent fixed batch
    torch.manual_seed(8)

    # set weights of losses of intermediate outputs to zero if not necessary
    if not args.warp_net:
        args.weight_Y_L1 = 0
        args.weight_Y_VGG = 0
    if not args.color_net:
        args.weight_Z_L1 = 0
        args.weight_Z_VGG = 0
        args.weight_Z_Adv = 0

    # datasets
    train_dir_1 = os.path.join(args.dataroot, 'Water', 'train')
    train_dir_2 = os.path.join(args.dataroot, 'ImageNet', 'train')
    val_dir_1 = os.path.join(args.dataroot, 'Water', 'test')
    val_dir_2 = os.path.join(args.dataroot, 'ImageNet', 'test')
    test_dir = os.path.join(args.dataroot, 'Water_Real')

    if args.synth_data:
        train_data = ImageFolder(train_dir_2, transform=
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            synthdata.SynthData(224, n=args.batch_size),
        ]))
    else:
        train_data = PairedImageFolder(train_dir_1, train_dir_2, transform=
        transforms.Compose([
            pairedtransforms.RandomResizedCrop(224),
            pairedtransforms.RandomHorizontalFlip(),
            pairedtransforms.ToTensor(),
            pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))
    val_data = PairedImageFolder(val_dir_1, val_dir_2, transform=
    transforms.Compose([
        pairedtransforms.Resize(256),
        pairedtransforms.CenterCrop(256),
        pairedtransforms.ToTensor(),
        pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]))
    test_data = ImageFolder(test_dir, transform=
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]), return_path=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers,
                                              pin_memory=True, shuffle=False)

    metric_name_to_function = {
        'mse': compare_mse,
        'psnr': compare_psnr,
        'ssim': lambda x, y: compare_ssim(x, y, multichannel=True),
    }

    # fixed test batch for visualization during training
    fixed_batch = iter(val_loader).next()[0]

    # model
    model = networks.Model(args)
    cuda_if_available(model)

    # load weights from checkpoint
    if args.test and not args.load:
        args.load = args.exp_name
    if args.load:
        model.load_state_dict(torch.load(
            os.path.join(args.outroot, '%s_net.pth' % args.load),
            map_location=map_location(),
        ), strict=args.test)

    # create outroot if necessary
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)

    # if args.test only run test script
    if args.test:
        evaluate(val_loader, model, args, metric_name_to_function)
        test(test_loader, model, args)
        return

    # main training loop
    for epoch in range(args.epochs):
        train(train_loader, model, fixed_batch, epoch, args)
        torch.save(model.state_dict(), os.path.join(args.outroot, '%s_net.pth' % args.exp_name))
        evaluate(val_loader, model, args, metric_name_to_function)
        test(test_loader, model, args)


def train(loader, model, fixed_batch, epoch, args):
    model.train()

    end_time = time.time()
    for i, ((input, target), _) in enumerate(loader):
        input = cuda_if_available(input)
        target = cuda_if_available(target)
        data_time = time.time() - end_time

        # take an optimization step
        losses = model.optimize_parameters(input, target)
        batch_time = time.time() - end_time

        # display progress
        print('%s Epoch: %02d/%02d %04d/%04d time: %.3f %.3f ' % (args.exp_name, epoch, args.epochs,
                                                                  i, len(loader), data_time,
                                                                  batch_time) + model.print_losses(losses))

        # visualize progress
        if i % 100 == 0:
            visualize(input, target, model, os.path.join(args.outroot, '%s_train.jpg' % args.exp_name))
            visualize(cuda_if_available(fixed_batch[0]), fixed_batch[1], model,
                      os.path.join(args.outroot, '%s_val.jpg' % args.exp_name))
            model.train()

        end_time = time.time()

        # if i==10:
        #    break


def visualize(input, target, model, name):
    model.eval()
    with torch.no_grad():
        x, warp, y, z = model(input)
        # TODO: add to model output
        x_ = Interp.inverse_warp(z, warp[:, 0, :, :], warp[:, 1, :, :])
        warp = (warp + 5) / 10
        warp = torch.cat((warp, torch.ones_like(y[:, :1, :, :])), dim=1)
        visuals = torch.cat([input.cpu(), x.cpu(), warp.cpu(), y.cpu(), z.cpu(), target.cpu(), x_.cpu()], dim=2)
        torchvision.utils.save_image(visuals, name, nrow=16, normalize=True, range=(-1, 1), pad_value=1)


def test(loader, model, args):
    model.eval()
    with torch.no_grad():
        end_time = time.time()
        for i, data in enumerate(loader):
            input = cuda_if_available(data[0])
            data_time = time.time() - end_time
            x, warp, y, z = model(input, cc=False)
            # To visualize field
            # X, Y = np.mgrid[0:warp.size()[2], 0:warp.size()[3]]
            # plt.quiver(X[::16, ::16], Y[::16, ::16], warp.detach().numpy().transpose((0, 2, 3, 1))[0, ::16, ::16, 0], warp.detach().numpy().transpose((0, 2, 3, 1))[0, ::16, ::16, 1], edgecolor='k', facecolor='None', linewidth=.1)

            # save the output for each image by name
            for out, name in zip(z, data[-1]):
                if not os.path.exists(os.path.join(args.outroot, '%s_test' % args.exp_name, os.path.dirname(name))):
                    os.makedirs(os.path.join(args.outroot, '%s_test' % args.exp_name, os.path.dirname(name)))

                im = Image.fromarray((out * .5 + .5).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
                im.save(os.path.join(args.outroot, '%s_test' % args.exp_name, name))

            batch_time = time.time() - end_time
            print('%s Test: %04d/%04d time: %.3f %.3f' % (
                args.exp_name,
                i, len(loader),
                data_time, batch_time,
            ))
            end_time = time.time()

            # if i==10:
            #    break


def evaluate(loader, model, args, metric_name_to_function):
    metric_name_to_values = {
        metric_name: list()
        for metric_name in metric_name_to_function.keys()
    }
    paths = list()
    model.eval()
    with torch.no_grad():
        end_time = time.time()

        for i, ((input, target), names) in enumerate(loader):
            input = cuda_if_available(input)
            data_time = time.time() - end_time
            x, warp, y, z = model(input, cc=False)

            for image_true, image_predicted, name in zip(target, z, names):
                paths.append(name)
                image_true = image_true.cpu().numpy().transpose((1, 2, 0))
                image_predicted = image_predicted.cpu().numpy().transpose((1, 2, 0))

                for metric_name, metric_function in metric_name_to_function.items():
                    metric_name_to_values[metric_name].append(metric_function(
                        image_true,
                        image_predicted,
                    ))

            batch_time = time.time() - end_time
            print('%s Evaluation: %04d/%04d time: %.3f %.3f %s' % (
                args.exp_name,
                i, len(loader),
                data_time, batch_time,
                ' '.join(map(lambda name_values:
                             '{}: {:.3f}'.format(name_values[0], np.mean(name_values[1])),
                             metric_name_to_values.items()))
            ))
            end_time = time.time()

    np.savez_compressed(
        os.path.join(args.outroot, '{}_evaluation'.format(args.exp_name)),
        paths=paths,
        **metric_name_to_values,
    )


def cuda_if_available(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return x


def map_location():
    if torch.cuda.is_available():
        return None

    return 'cpu'


if __name__ == '__main__':
    print(f'Using PyTorch version {torch.__version__} on {"gpu" if torch.cuda.is_available() else "cpu"}')
    main()
