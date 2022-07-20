from torch.autograd import Variable

import time
import argparse
from evaluation import Gray
from calculate import *
from sideInfoEnhancement import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', required=True, type=str, help='path to model')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--batch-size', '-N', type=int, default=1, help='batch size')
parser.add_argument(
    '--level', type=int, default=5, help='Decoding layer that controls the addition of edge information')
# 1 rnn1、2 rnn2、3 rnn3、4 rnn4、5 conv2_2
parser.add_argument(
    '--bits', type=int, default=16, help='Used to calculate BPP')
parser.add_argument(
    '--flag', type=int, default=5, help='unroll side frame')
# 1 前；2 中；3 后；4 噪声
parser.add_argument(
    '--test', required=True, type=str, default="/home/lvting19/dataset/test_min", help='folder of testing images')
parser.add_argument(
    '--testH265', required=True, type=str, default="/home/lvting19/dataset/test_min",
    help='folder of testingH265 images')
parser.add_argument(
    '--iterations', type=int, default=2, help='unroll iterations')
parser.add_argument(
    '--siIterations', type=int, default=2, help='unroll iterations')
parser.add_argument(
    '--max_batch', type=int, default=300, help='unroll iterations')
args = parser.parse_args()
import network1123, siNet1201

encoder = network1123.EncoderCell().cuda()
binarizer = network1123.Binarizer(args.bits).cuda()
decoder = network1123.DecoderCell(args.flag, args.bits).cuda()
sideProduce = siNet1201.SiNet().cuda()  # 1124 边信息生成网络

encoder.load_state_dict(torch.load(args.model))
binarizer.load_state_dict(torch.load(args.model.replace('encoder', 'binarizer')))
decoder.load_state_dict(torch.load(args.model.replace('encoder', 'decoder')))
sideProduce.load_state_dict(torch.load(args.model.replace('encoder', 'sideProduce')))


def init_lstm_test(batch_size, height, width):
    """
    1204 添加  用函数替换
    """
    encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True))
    encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True))
    encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True))

    decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True))
    decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True))
    decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True))
    decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2), volatile=True),
                   Variable(torch.zeros(batch_size, 128, height // 2, width // 2), volatile=True))

    encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
    encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
    encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    return (encoder_h_1, encoder_h_2, encoder_h_3,
            decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)


def psnr01(img1, img2):
    # mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


import torch.utils.data as data
import dataset1123

test_transform = transforms.Compose([
    # transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])

test_set = dataset1123.ImageFolder(is_train=False, root=args.test, rootH265=args.testH265, transform=test_transform)
test_loader = data.DataLoader(
    dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

PSNR = np.zeros(min(args.max_batch, len(test_loader)))
MS_SSIM = np.zeros(min(args.max_batch, len(test_loader)))

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


encoder_time = 0
decoder_time = 0
for batch, (imgAll, keyPrePath, keyNextPath, wzPath) in enumerate(test_loader):
    batch_t0 = time.time()

    encoder.eval()
    binarizer.eval()
    decoder.eval()
    sideProduce.eval()

    imgPre = imgAll[:, 0:1, :, :]
    imgMid = imgAll[:, 1:2, :, :]
    imgNext = imgAll[:, 2:3, :, :]

    imgPreOrg = imgAll[:, 3:4, :, :]
    imgNextOrg = imgAll[:, 4:5, :, :]

    data = imgMid
    # 在这里添加用H265进行关键帧的编解码

    if args.flag == 1:  # 前
        dataSide = imgPre
    elif args.flag == 2:  # 中
        dataSide = imgMid
    elif args.flag == 3:  # 后
        dataSide = imgNext
    elif args.flag == 4:  # 噪声
        dataSide = torch.FloatTensor(gasuss_noise(imgMid.numpy()))
    elif args.flag == 5:  # 前一帧和后一帧
        # dataSide = torch.cat([imgPre, imgNext], dim=1)
        dataSide = (imgPre + imgNext) / 2  # 1124 合成单通道

    image = Variable(data.cuda(), requires_grad=True)
    dataSide = Variable(dataSide.cuda(), requires_grad=True)

    (encoder_h_1, encoder_h_2, encoder_h_3,
     decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm_test(
        batch_size=image.size(0), height=image.size(2), width=image.size(3))

    # 1201 重构的关键帧生成边信息
    wzRes = image

    for iter in range(args.siIterations):
        si_t0 = time.time()
        dataSideTmp = torch.cat([dataSide, wzRes], dim=1)
        dataSide = sideProduce(dataSideTmp)
        si_t1 = time.time()

        res = image - 0.5
        dataSide = dataSide - 0.5
        out_img = torch.zeros(data.size()) + 0.5  # 确定输出的尺寸 1204添加

        bp_t0 = time.time()

        for iters in range(args.iterations):
            # Encode.
            begin = time.time()
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            # Binarize.
            code = binarizer(encoded)

            mid = time.time()
            encoder_time = encoder_time + mid - begin
            # Decode.
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, dataSide)

            # codes.append(code.data.cpu().numpy())
            out_img = out_img + output.data.cpu()
            res = res - output  # Variable

            end = time.time()
            decoder_time = decoder_time + end - mid
        decoder_time = decoder_time + si_t1 - si_t0
        wzRes = out_img.cuda()
    bp_t1 = time.time()

    PSNR[batch] = psnr01(out_img.cpu().numpy(), data.cpu().numpy())
    MS_SSIM[batch] = msssim(out_img.cpu().numpy() * 255, data.cpu().numpy() * 255)

    batch_t1 = time.time()
    print('Batch: {:02d}; Loss: {:.06f}; PSNR: {:.04f}; MS_SSIM: {:.4f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'
          .format(batch + 1, res.data.abs().mean(), PSNR[batch], MS_SSIM[batch],
                  bp_t1 - bp_t0, batch_t1 - batch_t0))

PSNR_Aver = np.mean(PSNR)
MS_SSIM_Aver = np.mean(MS_SSIM)
BPP = (args.bits * args.iterations * args.siIterations) / 256

print("PSNR_Aver: {:.4f}; MS_SSIM_Aver: {:.4f};".format(PSNR_Aver, MS_SSIM_Aver))
print("BPP: {}; 编码时间：{}; 解码时间: {}".format(BPP, encoder_time, decoder_time))
with open("testData.txt", "a+") as f:
    f.writelines(
        "Test: Dataset: {}; flag: {}; BPP: {:.4f}; bits: {}; siiter: {}; iter: {}; PSNR_Aver: {:.4f}; MS_SSIM_Aver: {:.4f}; Encoding_Time: {}; Decoding_Time: {}\n"
        .format(args.test.split("/")[6], args.flag, BPP, args.bits, args.siIterations, args.iterations, PSNR_Aver,
                MS_SSIM_Aver, encoder_time, decoder_time))
