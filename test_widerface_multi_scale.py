from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_eresnet
from layers.functions.prior_box import PriorBox
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', action="store_true", default=False, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--test_scales', default=[0.5, 1.0, 1.5, 2.0], type=float, nargs='+', help='multi-scale multipliers')
parser.add_argument('--do_flip', action="store_true", default=True, help='whether to do flip testing')
args = parser.parse_args()


# --------------------------------------------------------------------
# ⭐ RetinaFace 맞춤형 Bounding Box Voting 함수 (Landmark 포함 가중 평균)
# --------------------------------------------------------------------
def bbox_vote(det, nms_threshold=0.4):
    if det.shape[0] == 0:
        return det
    
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets_out = []
    
    while det.shape[0] > 0:
        # IOU 계산
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # 겹치는 박스들(Merge Index) 추출
        merge_index = np.where(o >= nms_threshold)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        # 겹치는 게 자기 자신뿐이면 그대로 저장
        if merge_index.shape[0] <= 1:
            dets_out.append(det_accu[0])
            continue

        # Score 가중치 추출
        weights = det_accu[:, 4:5]
        sum_weights = np.sum(weights)

        # 박스(0~3)와 랜드마크(5~14) 좌표 분리 후 합침
        coords = np.hstack((det_accu[:, 0:4], det_accu[:, 5:15]))
        
        # 가중 평균(Weighted Average) 계산
        avg_coords = np.sum(coords * weights, axis=0) / sum_weights
        max_score = np.max(det_accu[:, 4])

        # 새로운 15차원 배열로 병합된 결과 재조립
        merged_box = np.zeros((15,), dtype=np.float32)
        merged_box[0:4] = avg_coords[0:4]
        merged_box[4] = max_score
        merged_box[5:15] = avg_coords[4:14]

        dets_out.append(merged_box)

    dets_out = np.array(dets_out)
    # 점수순 정렬 후 최대 750개 반환 (WIDER FACE 공식)
    order = dets_out[:, 4].argsort()[::-1]
    dets_out = dets_out[order]
    return dets_out[:750, :]


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def predict_one_scale(img_raw, resize_factor, net, cfg, device, _t):
    img = np.float32(img_raw)
    if resize_factor != 1:
        img = cv2.resize(img, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)
    _t['forward_pass'].toc()

    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    prior_data = priors.data
    
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize_factor
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]]).to(device)
    landms = landms * scale1 / resize_factor
    landms = landms.cpu().numpy()
    _t['misc'].toc()

    return boxes, scores, landms


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "eresnet":
        cfg = cfg_eresnet
    else:
        pass 
    
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        im_shape = img_raw.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        
        target_size = 1024
        max_size = 1920
        base_resize = float(target_size) / float(im_size_min)
        if np.round(base_resize * im_size_max) > max_size:
            base_resize = float(max_size) / float(im_size_max)
            
        if args.origin_size:
            base_resize = 1.0

        all_boxes = []
        all_scores = []
        all_landms = []

        # 1. Multi-scale Testing Loop
        for scale_mult in args.test_scales:
            current_resize = base_resize * scale_mult
            
            boxes, scores, landms = predict_one_scale(img_raw, current_resize, net, cfg, device, _t)
            
            w = boxes[:, 2] - boxes[:, 0] + 1
            h = boxes[:, 3] - boxes[:, 1] + 1
            
            if scale_mult < 1.0:
                keep = np.where(np.maximum(w, h) > 30)[0]
            elif scale_mult > 1.0:
                keep = np.where(np.minimum(w, h) < 100)[0]
            else:
                keep = np.arange(len(boxes))
                
            boxes = boxes[keep]
            scores = scores[keep]
            landms = landms[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_landms.append(landms)

            # 2. Flip Image Test
            if args.do_flip:
                img_flip = cv2.flip(img_raw, 1)
                boxes_f, scores_f, landms_f = predict_one_scale(img_flip, current_resize, net, cfg, device, _t)
                
                im_width = im_shape[1]
                
                oldx1 = boxes_f[:, 0].copy()
                oldx2 = boxes_f[:, 2].copy()
                boxes_f[:, 0] = im_width - oldx2 - 1
                boxes_f[:, 2] = im_width - oldx1 - 1
                
                landms_f_restored = landms_f.copy()
                landms_f_restored[:, 0] = im_width - landms_f[:, 2] - 1  
                landms_f_restored[:, 1] = landms_f[:, 3]                 
                landms_f_restored[:, 2] = im_width - landms_f[:, 0] - 1  
                landms_f_restored[:, 3] = landms_f[:, 1]                 
                landms_f_restored[:, 4] = im_width - landms_f[:, 4] - 1  
                landms_f_restored[:, 6] = im_width - landms_f[:, 8] - 1  
                landms_f_restored[:, 7] = landms_f[:, 9]                 
                landms_f_restored[:, 8] = im_width - landms_f[:, 6] - 1  
                landms_f_restored[:, 9] = landms_f[:, 7]                 

                w_f = boxes_f[:, 2] - boxes_f[:, 0] + 1
                h_f = boxes_f[:, 3] - boxes_f[:, 1] + 1
                
                if scale_mult < 1.0:
                    keep_f = np.where(np.maximum(w_f, h_f) > 30)[0]
                elif scale_mult > 1.0:
                    keep_f = np.where(np.minimum(w_f, h_f) < 100)[0]
                else:
                    keep_f = np.arange(len(boxes_f))
                    
                boxes_f = boxes_f[keep_f]
                scores_f = scores_f[keep_f]
                landms_f_restored = landms_f_restored[keep_f]

                all_boxes.append(boxes_f)
                all_scores.append(scores_f)
                all_landms.append(landms_f_restored)

        # --------------------------------------------------------------------
        # ⭐ NMS 대신 Bbox Voting 적용 파트
        # --------------------------------------------------------------------
        _t['misc'].tic()
        
        has_boxes = any(len(b) > 0 for b in all_boxes)
        if has_boxes:
            boxes = np.vstack(all_boxes)
            scores = np.hstack(all_scores)
            landms = np.vstack(all_landms)

            # 기본 Thresold 통과한 박스들만 추림
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            if len(boxes) > 0:
                # Bbox(4), Score(1), Landmark(10) 모두 합쳐서 (N, 15) 차원 배열 생성
                dets = np.hstack((boxes, scores[:, np.newaxis], landms)).astype(np.float32, copy=False)
                
                # NMS 대신 Voting 함수 호출!
                dets = bbox_vote(dets, args.nms_threshold)
            else:
                dets = np.empty((0, 15))
        else:
            dets = np.empty((0, 15))
            
        _t['misc'].toc()

        # Save Txt and Images
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            if len(bboxs) == 0:
                fd.write(bboxs_num)
            else:
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

    total_time = _t['forward_pass'].total_time + _t['misc'].total_time
    fps = num_images / total_time
    print(f'Total FPS : {fps:.4f}')
    forward_time = num_images / _t['forward_pass'].total_time
    print(f'Forward Pass FPS : {forward_time:.4f}')