'''
Domain Adaptation Network based on DACS
'''
from mmseg.registry import MODELS
import logging
from typing import List, Dict, Union,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper
import numpy as np
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from typing import Iterable
def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything



@MODELS.register_module()
class DACS_encoder_decoder(BaseSegmentor):

    def __init__(self,**cfg):
        super().__init__(
            data_preprocessor=cfg.get('data_preprocessor',None),
            init_cfg=cfg.get('init_cfg',None)
        )
        
        self.backbone = MODELS.build(cfg['backbone'])
        if cfg.get('neck',None) is not None:
            self.neck = MODELS.build(cfg['neck'])
        self._init_decode_head(cfg.get('decode_head',None))
        self._init_auxiliary_head(cfg.get('auxiliary_head',None))

        self.train_cfg = cfg.get('train_cfg',None)
        self.test_cfg = cfg.get('test_cfg',None)

        '''
        init ema teacher model
        '''
        ema_cfg = cfg.copy()
        ema_cfg['type'] = 'EncoderDecoder'
        self.ema_model = MODELS.build(ema_cfg)
        for name,param in self.ema_model.named_parameters():
            param.requires_grad = False
    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels


    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,iter = None) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        losses = dict()
        source_data = data['source_data']
        target_data = data['target_data']
        source_data = self.data_preprocessor(source_data, True)
        target_data = self.data_preprocessor(target_data, True)

        loss_source = self.loss(**source_data)
        losses.update(loss_source)
        #get pseudo label
        if iter > 4000:
            with torch.no_grad():
                batch_img_metas = [
                    data_sample.metainfo for data_sample in target_data['data_samples']
                ]
                ema_seg_logits = self.ema_model.encode_decode(target_data['inputs'],batch_img_metas)
                ema_probability = ema_seg_logits.softmax(dim=1)
                ema_probs,ema_psudo_ann = torch.max(ema_probability,dim=1)
                # print(ema_psudo_ann)
            #dacs training
            Mixmask = self.generate_class_mask(src_ann=source_data['data_samples'])
            mix_data = self.mix_src_tgt(src_img=source_data['inputs'],src_ann=source_data['data_samples'],tgt_img=target_data['inputs'],tgt_pseudo_ann=ema_psudo_ann,mask=Mixmask)
            loss_mix = self.loss(**mix_data)
            '''
            对loss_mix中的所有损失乘0.5
            '''
            for key in loss_mix:
                loss_mix[key] = 0.5*loss_mix[key]
            '''
            loss_mix中的所有key加上'mix_'前缀
            '''
            loss_mix = {f'mix_{key}': value for key, value in loss_mix.items()}
            losses.update(loss_mix)

        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        self.update_ema(iter)
        return log_vars
    def update_ema(self, iter):
        '''
        更新ema模型参数
        '''
        alpha_teacher = 0.99
        if iter is None:
            iter = 0
        alpha_teacher = min(1 - 1 / (iter + 1), alpha_teacher)
        for ema_param, param in zip(self.ema_model.backbone.parameters(), self.backbone.parameters()):
            ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
        for ema_param, param in zip(self.ema_model.decode_head.parameters(), self.decode_head.parameters()):
            ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
    
    def generate_class_mask_single(label, classes):
        label, classes = torch.broadcast_tensors(label.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = label.eq(classes).sum(0)
        return N #512,512
    def generate_class_mask(self,src_ann=None):#选择源域标签中一半的类别
        
        gt_sem_seg = torch.stack([data_sample.gt_sem_seg.data for data_sample in src_ann]).squeeze(1)

        bs = gt_sem_seg.shape[0]
        mask = []
        for i in range(bs):
            classes = torch.unique(gt_sem_seg[i])
            n_cls = classes.shape[0]
            class_choiced = (classes[torch.Tensor(np.random.choice(n_cls, int((n_cls+n_cls%2)/2),replace=False)).long()]).cuda() #除以2减1
            mask_i = self.generate_class_mask_single(gt_sem_seg[i],class_choiced)
            mask.append(mask_i)
        
        return mask
    def generate_class_mask_single(self,label, classes):
        label, classes = torch.broadcast_tensors(label.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = label.eq(classes).sum(0)
        return N #512,512
    def mix_src_tgt(self,src_img = None,src_ann = None,tgt_img = None,tgt_pseudo_ann = None,mask = None):
        bs = len(mask)
        src_ann_copy = src_ann.copy()
        src_ann = torch.stack([data_sample.gt_sem_seg.data for data_sample in src_ann]).squeeze(1)

        mix_img = torch.zeros_like(src_img)
        mix_ann = torch.zeros_like(src_ann)
        for i in range(bs):
            mix_img[i] = mask[i] * src_img[i] + (1-mask[i])*tgt_img[i]
            mix_ann[i] = mask[i] * src_ann[i] + (1-mask[i])*tgt_pseudo_ann[i]#src_ann, 4,512,512 ->4,6,512,512
        mix_data = {}
        mix_data['inputs'] = mix_img
        mix_data['data_samples'] = src_ann_copy
        for i in range(bs):
            mix_data['data_samples'][i].gt_sem_seg.data = mix_ann[i].unsqueeze(0)
        return mix_data
    # forzen backbone
    # def extract_feat(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     with torch.no_grad():
    #         x = self.backbone(inputs)
    #         x = detach_everything(x)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)#4,3,512,512->[[4,1024,128,128],[4,1024,64,64],[4,1024,32,32],[4,1024,16,16]]
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses




    def loss(self, inputs: Tensor, data_samples: SampleList,iter = None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)#[4,64,128,128|4,128,64,64|4,256,64,64|4,512,64,64]
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
