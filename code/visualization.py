import os
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

savepath = '/content/features'
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(width, height, x, savename):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        # plt.imshow(img, cmap='gray')
        plt.imshow(img)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()


class visul_hrnet(nn.Module):

    def __init__(self, model):
        super(visul_hrnet, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.base(x)
        x = self.model.stage1(x)

        x_list = []
        for i in range(self.model.stage2_cfg['NUM_BRANCHES']):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        draw_features(8, 2, y_list[0].cpu().numpy(), "{}/{}.png".format(savepath, "stage2-" + str(0)))
        draw_features(8, 4, y_list[1].cpu().numpy(), "{}/{}.png".format(savepath, "stage2-" + str(1)))

        x_list = []
        for i in range(self.model.stage3_cfg['NUM_BRANCHES']):
            if self.model.transition2[i] is not None:
                x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        draw_features(8, 2, y_list[0].cpu().numpy(), "{}/{}.png".format(savepath, "stage3-" + str(0)))
        draw_features(8, 4, y_list[1].cpu().numpy(), "{}/{}.png".format(savepath, "stage3-" + str(1)))
        draw_features(8, 8, y_list[2].cpu().numpy(), "{}/{}.png".format(savepath, "stage3-" + str(2)))

        x_list = []
        for i in range(self.model.stage4_cfg['NUM_BRANCHES']):
            if self.model.transition3[i] is not None:
                x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)

        draw_features(8, 2, y_list[0].cpu().numpy(), "{}/{}.png".format(savepath, "stage4-" + str(0)))
        draw_features(8, 4, y_list[1].cpu().numpy(), "{}/{}.png".format(savepath, "stage4-" + str(1)))
        draw_features(8, 8, y_list[2].cpu().numpy(), "{}/{}.png".format(savepath, "stage4-" + str(2)))
        draw_features(8, 16, y_list[3].cpu().numpy(), "{}/{}.png".format(savepath, "stage4-" + str(3)))



class visul_resnet(nn.Module):
    def __init__(self, model):
        super(visul_resnet, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        draw_features(8, 8, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

        x = self.model.layer2(x)
        draw_features(8, 8, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

        x = self.model.layer3(x)
        draw_features(8, 8, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

        x = self.model.layer4(x)
        draw_features(8, 8, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))