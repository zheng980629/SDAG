import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from scipy.stats import pearsonr
from torchvision.transforms import Resize


@MODEL_REGISTRY.register()
class SemanticAwareSelfSupervisedModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SemanticAwareSelfSupervisedModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.transform =  Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt['recon_opt']['loss_weight'] > 0:
            self.cri_recon = build_loss(train_opt['recon_opt']).to(self.device)
        else:
            self.cri_recon = None

        if train_opt['margin_opt']['loss_weight'] > 0:
            self.cri_margin = build_loss(train_opt['margin_opt']).to(self.device)
        else:
            self.cri_margin = None

        if train_opt['PearsonScore_opt']['loss_weight'] > 0:
            self.cri_pearsonScore = build_loss(train_opt['PearsonScore_opt']).to(self.device)
        else:
            self.cri_pearsonScore = None
        
        if train_opt['lqRecon_opt']['loss_weight'] > 0:
            self.cri_lqRecon = build_loss(train_opt['lqRecon_opt']).to(self.device)
        else:
            self.cri_lqRecon = None

            

        # if train_opt.get('perceptual_opt'):
        #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        # else:
        #     self.cri_perceptual = None

        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.sigma = data['sigma'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.recon_L = self.net_g(self.transform(self.lq))
        self.recon_L = self.recon_L * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

        self.lq_gamma = self.lq ** (1 / 0.3)
        self.recon_L_gamma = self.net_g(self.transform(self.lq_gamma))
        self.recon_L_gamma = self.recon_L_gamma * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

        self.lq_noise = self.lq + torch.normal(0, 1 / 255.0, self.lq.shape).cuda() * 50
        self.recon_L_noise = self.net_g(self.transform(self.lq_noise))
        self.recon_L_noise = self.recon_L_noise * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_recon:
            l_recon = self.cri_recon(self.recon_L, self.gt)
            l_total += l_recon
            loss_dict['l_recon'] = l_recon
        
        if self.cri_margin:
            cri_margin_gamma = self.cri_margin(self.recon_L, self.lq, self.recon_L_gamma, self.lq_gamma)
            l_total += cri_margin_gamma
            loss_dict['cri_margin_gamma'] = cri_margin_gamma

        if self.cri_margin:
            cri_margin_noise = self.cri_margin(self.recon_L, self.lq, self.recon_L_noise, self.lq_noise)
            l_total += cri_margin_noise
            loss_dict['cri_margin_noise'] = cri_margin_noise

        if self.cri_pearsonScore:
            cri_pearsonScore_gamma = self.cri_pearsonScore(self.recon_L, self.lq, self.recon_L_gamma, self.lq_gamma)
            l_total += cri_pearsonScore_gamma
            loss_dict['cri_pearsonScore_gamma'] = cri_pearsonScore_gamma

        if self.cri_pearsonScore:
            cri_pearsonScore_noise = self.cri_pearsonScore(self.recon_L, self.lq, self.recon_L_noise, self.lq_noise)
            l_total += cri_pearsonScore
            loss_dict['cri_pearsonScore'] = cri_pearsonScore

        if self.cri_lqRecon:
            cri_lqRecon_gamma = self.cri_lqRecon(self.recon_L_gamma, self.lq_gamma)
            l_total += cri_lqRecon_gamma
            loss_dict['cri_lqRecon_gamma'] = cri_lqRecon_gamma

        if self.cri_lqRecon:
            cri_lqRecon_noise = self.cri_lqRecon(self.recon_L_noise, self.lq_noise)
            l_total += cri_lqRecon_noise
            loss_dict['cri_lqRecon_noise'] = cri_lqRecon_noise

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output_recon_L = self.net_g_ema(self.transform(self.lq))
                self.output_recon_L = self.output_recon_L * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

                self.lq_gamma = self.lq ** (1 / 0.3)
                self.output_recon_gamma = self.net_g_ema(self.transform(self.lq_gamma))
                self.output_recon_gamma = self.output_recon_gamma * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

                self.lq_noise = self.lq + torch.normal(0, 1 / 255.0, self.lq.shape).cuda() * 50
                self.output_recon_noise = self.net_g_ema(self.transform(self.lq_noise))
                self.output_recon_noise = self.output_recon_noise * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)
        
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output_recon_L = self.net_g(self.transform(self.lq))
                self.output_recon_L = self.output_recon_L * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)
            
                self.lq_gamma = self.lq ** (1 / 0.3)
                self.output_recon_gamma = self.net_g(self.transform(self.lq_gamma))
                self.output_recon_gamma = self.output_recon_gamma * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

                self.lq_noise = self.lq + torch.normal(0, 1 / 255.0, self.lq.shape).cuda() * 50
                self.output_recon_noise = self.net_g(self.transform(self.lq_noise))
                self.output_recon_noise = self.output_recon_noise * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.229, 0.224, 0.225])),0),2),3).to(self.device) + \
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array([0.485, 0.456, 0.406])),0),2),3).to(self.device)

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # [0-255]
            lq_img = tensor2img([visuals['lq']])
            lq_gamma_img = tensor2img([visuals['lq_gamma']])
            lq_noise_img = tensor2img([visuals['lq_noise']])
            recon_L_img = tensor2img(visuals['recon_L'])
            recon_L_gamma_img = tensor2img(visuals['recon_gamma'])
            recon_L_noise_img = tensor2img(visuals['recon_noise'])

            metric_data['img'] = recon_L_img
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = lq_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_lq = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_lq.png')
                    save_img_path_recon_L = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_recon_L.png')
                    save_img_path_lq_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_lq_gamma.png')
                    save_img_path_recon_L_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_recon_L_gamma.png')
                    save_img_path_lq_noise = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_lq_noise.png')
                    save_img_path_recon_L_noise = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_recon_L_noise.png')
                    # save_img_path_noiseMap = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_noiseMap.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path_lq = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq.png')
                        save_img_path_recon_L = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L.png')
                        save_img_path_lq_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq_gamma.png')
                        save_img_path_recon_L_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L_gamma.png')
                        save_img_path_lq_noise = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq_noise.png')
                        save_img_path_recon_L_noise = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L_noise.png')
                        # save_img_path_noiseMap = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_noiseMap.png')
                    else:
                        save_img_path_lq = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq.png')
                        save_img_path_recon_L = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L.png')
                        save_img_path_lq_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq_gamma.png')
                        save_img_path_recon_L_gamma = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L_gamma.png')
                        save_img_path_lq_noise = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq_noise.png')
                        save_img_path_recon_L_noise = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_recon_L_noise.png')
                        # save_img_path_noiseMap = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_noiseMap.png')
                imwrite(lq_img, save_img_path_lq)
                imwrite(recon_L_img, save_img_path_recon_L)
                imwrite(lq_gamma_img, save_img_path_lq_gamma)
                imwrite(recon_L_gamma_img, save_img_path_recon_L_gamma)
                imwrite(lq_noise_img, save_img_path_lq_noise)
                imwrite(recon_L_noise_img, save_img_path_recon_L_noise)
                # imwrite(gt_img, save_img_path_gt)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['recon_L'] = self.output_recon_L
        out_dict['recon_gamma'] = self.output_recon_gamma
        out_dict['recon_noise'] = self.output_recon_noise
        out_dict['lq_gamma'] = self.lq_gamma
        out_dict['lq_noise'] = self.lq_noise
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
