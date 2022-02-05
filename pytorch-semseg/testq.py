import os
import torch
import argparse
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    pass
    # print(
    #     "Failed to import pydensecrf,\
    #        CRF post-processing will not work"
    # )
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [
        7,
        8,
        11,
        12,
        13,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
    ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_segmap( mask):
  # Put all void classes to zero
  for _voidc in void_classes:
      mask[mask == _voidc] = 250
  for _validc in valid_classes:
      mask[mask == _validc]  =class_map[_validc]
  return mask

n_classes=19
def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

colors = [  # [  0,   0,   0],
[128, 64, 128],
[244, 35, 232],
[70, 70, 70],
[102, 102, 156],
[190, 153, 153],
[153, 153, 153],
[250, 170, 30],
[220, 220, 0],
[107, 142, 35],
[152, 251, 152],
[0, 130, 180],
[220, 20, 60],
[255, 0, 0],
[0, 0, 142],
[0, 0, 70],
[0, 60, 100],
[0, 80, 100],
[0, 0, 230],
[119, 11, 32],
]
class_map = dict(zip(valid_classes, range(19)))
label_colours = dict(zip(range(19), colors))

def test(model_path,img_path,out_path,image_name):
    dataset='cityscapes'
    img_norm=True
    dcrf=False
    n_classes=19
    img_size=(256,512) # 512 , 1024
    mean=[0.0, 0.0, 0.0]
   
    model_file_name = os.path.split(model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    img = misc.imread(img_path)
    print('image shape',img.shape)
    #data_loader = get_loader(args.dataset)
    #loader = data_loader(root='content/data/', is_transform=True, img_norm=args.img_norm, test_mode=True)

    resized_img = misc.imresize(img, (img_size[0], img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
        img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
    else:
        img = misc.imresize(img, (img_size[0], img_size[1]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean
    if img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=dataset)
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)

    if dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = out_path[:-4] + "_drf.png"
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = misc.imresize(pred, orig_size, "nearest", mode="F")
    #return pred
    decoded = decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    misc.imsave(out_path+image_name+'.png', decoded)
    print("Segmentation Mask Saved at: {}".format(out_path))


