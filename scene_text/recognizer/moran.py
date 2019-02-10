from collections import OrderedDict
import logging
import os

import cv2
from PIL import Image
import torch
from torch.autograd import Variable

from .MORAN_v2.tools import utils
from .MORAN_v2.tools import dataset
from .MORAN_v2.models.moran import MORAN

log = logging.getLogger(__name__)

class MORANRecognizer:

    def __init__(self
        , model_path = os.path.join(os.path.dirname(__file__), 'MORAN_v2/demo.pth')):

        alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

        self.cuda_flag = False
        if torch.cuda.is_available():
            self.cuda_flag = True
            self.MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=self.cuda_flag)
            self.MORAN = self.MORAN.cuda()
        else:
            self.MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=self.cuda_flag)

        if not os.path.isfile(model_path):
            log.info('loading model from Google Drive URL')
            from scene_text.util import download_file_from_google_drive
            download_file_from_google_drive('1IDvT51MXKSseDq3X57uPjOzeSYI09zip',
                model_path)

        log.info('loading pretrained model from %s' % model_path)
        if self.cuda_flag:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            MORAN_state_dict_rename[name] = v
        self.MORAN.load_state_dict(MORAN_state_dict_rename)

        for p in self.MORAN.parameters():
            p.requires_grad = False
        self.MORAN.eval()

        self.converter = utils.strLabelConverterForAttention(alphabet, ':')
        self.transformer = dataset.resizeNormalize((100, 32))


    def recognize(self, cv2_img):
        cv2_im = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image = pil_im.convert('L')
        image = self.transformer(image)

        if self.cuda_flag:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        text = Variable(text)
        length = Variable(length)

        max_iter = 20
        t, l = self.converter.encode('0'*max_iter)
        utils.loadData(text, t)
        utils.loadData(length, l)
        output = self.MORAN(image, length, text, text, test=True, debug=True)

        preds, preds_reverse = output[0]
        demo = output[1]

        _, preds = preds.max(1)
        _, preds_reverse = preds_reverse.max(1)

        sim_preds = self.converter.decode(preds.data, length.data)
        sim_preds = sim_preds.strip().split('$')[0]
        sim_preds_reverse = self.converter.decode(preds_reverse.data, length.data)
        sim_preds_reverse = sim_preds_reverse.strip().split('$')[0]
        # cv2.imshow("demo", demo)
        # cv2.waitKey()
        return  {'ltr': sim_preds, 'rtl': sim_preds_reverse }
