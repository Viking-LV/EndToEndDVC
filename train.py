import time
import argparse
from calculate import *
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sideInfoEnhancement import *

import siNet1201

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=16, help='batch size')
parser.add_argument(
    '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--trainH265', '-H', required=True, type=str, help='folder of training imagesH265')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=15, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument(
    '--iterations', type=int, default=1, help='unroll iterations')
parser.add_argument(
    '--siIterations', type=int, default=1, help='unroll iterations')
parser.add_argument(
    '--level', type=int, default=5, help='Decoding layer that controls the addition of edge information')
# 1 rnn1、2 rnn2、3 rnn3、4 rnn4、5 conv2_2
parser.add_argument(
    '--bits', type=int, default=32, help='unroll bits')
parser.add_argument(
    '--flag', type=int, default=5, help='unroll side frame')
# 1 前；2 中；3 后；4 噪声；5 前后两帧
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping.')  # 1211 添加 模仿插值
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1209 指定GPU

## load 32x32 patches from images
import dataset1123, network1123

train_transform = transforms.Compose([
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
])

train_set = dataset1123.ImageFolder(is_train=True, root=args.train, rootH265=args.trainH265)

train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

## load networks on GPU
while 1:
    try:
        print("test")
        outpath = 'flag_{}/bits_{}/siIter_{}_iter_{}/encoder_epoch_{:05d}.pth'.format(
            args.flag, args.bits, args.siIterations, args.iterations, args.max_epochs)
        outpath = "/home/user333/vvvvv/distrubute_video_coding" + "/" + outpath
        encoder = network1123.EncoderCell().cuda()
        binarizer = network1123.Binarizer(args.bits).cuda()
        decoder = network1123.DecoderCell(args.flag, args.bits).cuda()
        sideProduce = siNet1201.SiNet().cuda()  # 1124 边信息生成网络

        solver = optim.Adam(
            [
                {
                    'params': encoder.parameters()
                },
                {
                    'params': binarizer.parameters()
                },
                {
                    'params': decoder.parameters()
                },
                {
                    'params': sideProduce.parameters()
                },
            ],
            lr=args.lr)


        def resume(epoch=None):
            if epoch is None:
                s = 'iter'
                epoch = 0
            else:
                s = 'epoch'

            encoder.load_state_dict(
                torch.load('checkpoint_{}/encoder_{}_{:08d}.pth'.format(args.bits, s, epoch)))
            binarizer.load_state_dict(
                torch.load('checkpoint_{}/binarizer_{}_{:08d}.pth'.format(args.bits, s, epoch)))
            decoder.load_state_dict(
                torch.load('checkpoint_{}/decoder_{}_{:08d}.pth'.format(args.bits, s, epoch)))


        def gasuss_noise(image, mean=0, var=0.001):
            '''
                添加高斯噪声
                mean : 均值
                var : 方差
            '''
            # image = np.array(image/255, dtype=float)
            noise = np.random.normal(mean, var ** 0.5, image.shape)
            out = image + noise
            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            out = np.clip(out, low_clip, 1.0)
            return out


        def save(index, epoch=True):
            if not os.path.exists('flag_{}/bits_{}/siIter_{}_iter_{}'.format(args.flag, args.bits, args.siIterations,
                                                                             args.iterations)):
                # os.mkdir('flag_{}/checkpoint_{}'.format(args.flag, args.iterations))
                os.makedirs('flag_{}/bits_{}/siIter_{}_iter_{}'.format(args.flag, args.bits, args.siIterations,
                                                                       args.iterations))  # 创建多级目录

            if epoch:
                s = 'epoch'
            else:
                s = 'iter'

            torch.save(encoder.state_dict(), 'flag_{}/bits_{}/siIter_{}_iter_{}/encoder_{}_{:05d}.pth'.format(
                args.flag, args.bits, args.siIterations, args.iterations, s, index))

            torch.save(binarizer.state_dict(),
                       'flag_{}/bits_{}/siIter_{}_iter_{}/binarizer_{}_{:05d}.pth'.format(
                           args.flag, args.bits, args.siIterations, args.iterations, s, index))

            torch.save(decoder.state_dict(), 'flag_{}/bits_{}/siIter_{}_iter_{}/decoder_{}_{:05d}.pth'.format(
                args.flag, args.bits, args.siIterations, args.iterations, s, index))

            torch.save(sideProduce.state_dict(), 'flag_{}/bits_{}/siIter_{}_iter_{}/sideProduce_{}_{:05d}.pth'.format(
                args.flag, args.bits, args.siIterations, args.iterations, s, index))


        def psnr01(img1, img2):
            # mse = np.mean( (img1/255. - img2/255.) ** 2 )
            mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
            if mse < 1.0e-10:
                return 100
            PIXEL_MAX = 1.0
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


        scheduler = LS.MultiStepLR(solver, milestones=[5000, 6000, 7000, 8000, 9000], gamma=0.5)

        last_epoch = 0
        if args.checkpoint:
            resume(args.checkpoint)
            last_epoch = args.checkpoint
            scheduler.last_epoch = last_epoch - 1


        def np_to_torch(img):
            img = np.swapaxes(img, 0, 1)  # 1, w, h, 3
            img = np.swapaxes(img, 0, 2)  # 1, 3, h, w
            return torch.from_numpy(img).float()


        PSNR = np.zeros(args.max_epochs * len(train_loader))
        MS_SSIM = np.zeros(args.max_epochs * len(train_loader))

        for epoch in range(last_epoch + 1, args.max_epochs + 1):

            scheduler.step()

            for batch, (imgAll, keyPrePath, keyNextPath, wzPath) in enumerate(train_loader):
                batch_t0 = time.time()

                imgAll = torch.cat((imgAll[0], imgAll[1]), dim=0)

                imgPre = imgAll[:, 0:1, :, :]
                imgMid = imgAll[:, 1:2, :, :]
                imgNext = imgAll[:, 2:3, :, :]

                imgPreOrg = imgAll[:, 3:4, :, :]
                imgNextOrg = imgAll[:, 4:5, :, :]

                preRec, flow1 = sideInfoPreProcess(imgPreOrg, imgMid)
                nxtRec, flow2 = sideInfoPreProcess(imgNextOrg, imgMid)
                imgPre = sideInfoEnhance(imgPreOrg, imgPre, preRec)  # 优化边信息（前一帧）
                imgNext = sideInfoEnhance(imgNextOrg, imgNext, nxtRec)  # 优化边信息（后一帧）

                data = imgMid

                if args.flag == 1:  # 前
                    dataSide = imgPre
                elif args.flag == 2:  # 中
                    dataSide = imgMid
                elif args.flag == 3:  # 后
                    dataSide = imgNext
                elif args.flag == 4:
                    dataSide = torch.FloatTensor(gasuss_noise(imgMid.numpy()))
                elif args.flag == 5:
                    dataSide = (imgPre + imgNext) / 2  # 1124 合成单通道
                    # dataSide = torch.cat([imgPre, imgNext], dim=1)


                patches = Variable(data.cuda())
                dataSide = Variable(dataSide.cuda())

                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()),
                               Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()))
                encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()),
                               Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()))
                encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()),
                               Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()))

                decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()),
                               Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()))
                decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()),
                               Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()))
                decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()),
                               Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()))
                decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, data.size(2) // 2, data.size(3) // 2).cuda()),
                               Variable(torch.zeros(data.size(0), 128, data.size(2) // 2, data.size(3) // 2).cuda()))

                # 1201 重构的关键帧生成边信息
                wzRes = patches

                solver.zero_grad()

                losses_wz = []
                losses_si = []

                bp_t0 = time.time()

                # 迭代生成边信息 1201
                for iter in range(args.siIterations):
                    dataSideTmp = torch.cat([dataSide, wzRes], dim=1)
                    dataSide = sideProduce(dataSideTmp)

                    res = patches - 0.5
                    dataSide = dataSide - 0.5
                    image = torch.zeros(data.size()) + 0.5
                    for iteration in range(args.iterations):
                        # res = patches - 0.5

                        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                            res, encoder_h_1, encoder_h_2, encoder_h_3)

                        codes = binarizer(encoded)

                        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                            codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, dataSide)

                        image = image + output.data.cpu()  # additive reconstruction

                        res = res - output  # 输入与输出的残差

                        loss_wz = res.abs().mean()

                        losses_wz.append(loss_wz)

                    # 迭代过程中重构的WZ帧辅助边信息的优化  1124
                    wzRes = image.cuda()
                    loss_si = (dataSide - patches).abs().mean()
                    losses_si.append(loss_si)

                # image = image / args.iterations
                index = (epoch - 1) * len(train_loader) + batch
                PSNR[index] = psnr01(data.cpu().numpy(), image.cpu().numpy())
                MS_SSIM[index] = msssim(data.cpu().numpy() * 255, image.cpu().numpy() * 255)

                bp_t1 = time.time()

                loss = sum(losses_wz) / (args.iterations * args.siIterations) + \
                       sum(losses_si) / args.siIterations
                loss.backward()

                solver.step()

                batch_t1 = time.time()
                if index == 0:
                    BPP = (args.bits * args.iterations * args.siIterations) / 256
                    print("BPP : {:4f}".format(BPP))

                print(  # 华为云服务器版本升级修改
                    '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                        format(epoch, batch + 1,
                               len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 -
                               batch_t0))
                print("\t\tindex :{}; PSNR[{}] :{:.6f}; MS_SSIM[{}] :{:.6f}"
                      .format(index, index, PSNR[index], index, MS_SSIM[index]))

    except:
        continue

print("Complete the training")
with open("trainDataLevel.txt", "a+") as f:
    f.writelines("Train:\titer: {}; bits: {};flag: {}; BPP: {:.4f}; PSNR: {:.4f}; MS_SSIM: {:.4f};\n"
                 .format(args.iterations, args.bits, args.flag, BPP, PSNR[index], MS_SSIM[index]))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # 让2个子图的x轴一样，同时创建副坐标轴。
lns1 = ax1.plot(PSNR, color='red', label='PSNR')
lns2 = ax2.plot(MS_SSIM, color='green', label='MS_SSIM')
ax1.set_xlabel('iteration')
ax1.set_ylabel('PSNR')
ax2.set_ylabel('MS_SSIM')
plt.title("BPP = {:.4f}".format(BPP))
filename = "flag = {}, BPP = {:.4f}, bits = {}".format(args.flag, BPP, args.bits)

lns = lns1 + lns2
labels = ['PSNR', 'MS_SSIM']
plt.legend(lns, labels, loc=7)
plt.xlim(0, index)
plt.savefig('./PSNR_Image/' + filename + '.png')
plt.show()
