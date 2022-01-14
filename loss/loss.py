from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReconstructionLoss(nn.Module): #supposed to be small
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr): # mean of all images' pixelwise MAEs (mean absolute errors)  in one batch
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        loss = F.mse_loss(sr_relu5_1, hr_relu5_1) #sr_relu5_1 = self.vgg19((sr + 1.) / 2.) in trainer.py ------ (sr+1)/2 changes range of possible values to 0 to 1, as vgg19 needs it
        return loss


class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        ### S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*2, S.size(-1)*2), mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*4, S.size(-1)*4), mode='bicubic'))
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture  = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        elif (self.type == 'l2'):
            loss_texture  = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        
        return loss_texture


class AdversarialLoss(nn.Module):
    def __init__(self, logger, device, args, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1, 
        lr_dis_fkt=2, train_crop_size=32):#crop size default value,but defined in
        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.args = args
        self.lr_dis_fkt = lr_dis_fkt
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = device #torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(train_crop_size*4).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.AdamW(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=self.args.lr_base*lr_dis_fkt
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

            
    def forward(self, fake, real,  learningrate=1e-4, no_backward = False):
        for g in self.optimizer.param_groups:
            g['lr'] = learningrate*self.lr_dis_fkt

        fake_detach = fake.detach()
        #self.optimizer.param_groups[0]['lr']
        ###Training of Discrimiator
        if not (self.args.eval or self.args.test or no_backward):
            for _ in range(self.gan_k): 
                self.optimizer.zero_grad()
                d_fake = self.discriminator(fake_detach)
                d_real = self.discriminator(real)
                if (self.gan_type.find('WGAN') >= 0):
                    loss_d = (d_fake - d_real).mean() # E[ D(x^) - D(x) ] -----x^~generator_Distribution ,     x~real_Distribution
                    if self.gan_type.find('GP') >= 0:
                        epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                        epsilon = epsilon.expand(real.size())
                        hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                        
                        hat.requires_grad = True 
                        d_hat = self.discriminator(hat)
                        gradients = torch.autograd.grad(
                            outputs=d_hat.sum(), inputs=hat,
                            retain_graph=True, create_graph=True, only_inputs=True
                        )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_norm = gradients.norm(2, dim=1)
                        gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                        loss_d += gradient_penalty

                elif (self.gan_type == 'GAN'):
                    valid_score = torch.ones(real.size(0), 1).to(self.device)
                    fake_score = torch.zeros(real.size(0), 1).to(self.device)
                    real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                    fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                    loss_d = (real_loss + fake_loss) / 2.

                # Discriminator update
                #if not no_backward:
                loss_d.backward()
                self.optimizer.step()
                
        ###calculate generator (TTSR) loss
        d_fake_for_g = self.discriminator(fake) 
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g
  
    def get_state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict
    
    def update_state_dict(self, discr_path=None, discr_optim_path=None):
        discr_state_dict_save = {k:v for k,v in torch.load(discr_path, map_location=self.device).items()}
        discr_state_dict = self.discriminator.state_dict()
        discr_state_dict.update(discr_state_dict_save)            
        self.discriminator.load_state_dict(discr_state_dict)
        if (discr_optim_path):
            discr_optim_state_dict_save = {k:v for k,v in torch.load(discr_optim_path, map_location=self.device).items()}
            discr_optim_state_dict = self.optimizer.state_dict()
            discr_optim_state_dict.update(discr_optim_state_dict_save)            
            self.optimizer.load_state_dict(discr_optim_state_dict)
        



def get_loss_dict(args, logger, device):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8): #If Reconstruction loss weight smaller than 1e-8 --> System Error
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type='l1')
    if (abs(args.per_w - 0) > 1e-8):
        loss['per_loss'] = PerceptualLoss()#If Reconstruction loss weight smaller than 1e-8 --> System Error
    if (abs(args.tpl_w - 0) > 1e-8):
        loss['tpl_loss'] = TPerceptualLoss(use_S=args.tpl_use_S, type=args.tpl_type)#If Reconstruction loss weight smaller than 1e-8 --> System Error
    if (abs(args.adv_w - 0) > 1e-8):
        loss['adv_loss'] = AdversarialLoss(logger=logger,args=args, device=device, use_cpu=args.cpu, num_gpu=args.num_gpu, 
            gan_type=args.GAN_type, gan_k=args.GAN_k, lr_dis_fkt=args.lr_rate_dis_fkt,
            train_crop_size=args.train_crop_size)  # Reconstruction loss weight smaller than 1e-8 --> System Error
    return loss