import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import higher


def disable_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class AverageMeter:

    def __init__(self, name):
        self.value = 0
        self.total_count = 0

    def update(self, value, count):
        old_count = self.total_count
        new_count = new_count + count

        self.value = self.value * old_count / new_count + value * count / new_count
        self.total_count = new_count

    def reset(self):
        self.value = 0
        self.count = 0

class ERM(nn.Module):
    def __init__(self, model, loss_fn, device, hparams, init_optim=True, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        self.optimizer_name = hparams['optimizer']
        self.learning_rate = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']

        if init_optim:
            params = self.model.parameters()
            self.init_optimizers(params)

    def init_optimizers(self, params):
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, params),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay)

    def predict(self, x):
        return self.model(x)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, logits, labels):
        # Evaluate
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
        return accuracy

    def learn(self, images, labels, group_ids=None, mask=None, mask_p=1.0):
        
        self.train()

        # Forward
        logits = self.predict(images) # batch, num_class

        # Apply the mask
        if mask:
            random_mask = torch.randn(len(labels)).float().ge(mask_p).float().to(self.device)
            loss = self.loss_fn(logits, labels) * random_mask
            loss = torch.sum(loss) / torch.sum(random_mask)
        else:
            loss = self.loss_fn(logits, labels)

        self.update(loss)
        stats = { 'objective': loss.detach().item(), }
        return logits, stats


class DRNNLossComputer:
    """ Adapted from public DRNN repo"""
    def __init__(self, criterion, is_robust, n_groups, alpha=None, gamma=0.1, adj=None,
                 min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False, device='cpu'):

        self.criterion = criterion
        self.is_robust = is_robust
        self.step_size = step_size

        self.btl = btl # set to false
        self.device = device

        self.n_groups = n_groups

        # quantities mintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups

        # The following 4 variables are not used
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.normalize_loss = normalize_loss

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count) # this one is actually used
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss

        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count


class DRNN(ERM):

    def __init__(self, model, loss_fn, device, n_groups,  hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.device = device
        self.n_groups = n_groups
        self.loss_computer = DRNNLossComputer(loss_fn, is_robust=True,
                                         n_groups=n_groups,
                                         step_size=hparams['robust_step_size'],
                                         device=device)

    def learn(self, images, labels, group_ids):
        self.train()

        # Forward and backward
        logits = self.predict(images)
        loss = self.loss_computer.loss(logits, labels, group_ids.to(self.device), is_training=True)

        self.update(loss)

        stats = {
                'objective': loss.detach().item()
                }

        return logits, stats


class ARM_CML(ERM):

    def __init__(self, model, loss_fn, device, context_net, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.context_net = context_net
        self.support_size = hparams['support_size']
        self.n_context_channels = hparams['n_context_channels']
        self.adapt_bn = hparams['adapt_bn']
        self.normalize = hparams['normalize']
        self.online = hparams['online']
        self.T = hparams['T']
        self.zero_context = hparams['zero_context']

        params = list(self.model.parameters()) + list(self.context_net.parameters())

        self.init_optimizers(params)


    def predict(self, x):
        batch_size, c, h, w = x.shape
        # print("x.shape: batch_size, c, h, w", x.shape)

        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size

        if self.adapt_bn:
            out = []
            for i in range(meta_batch_size):
                x_i = x[i*support_size:(i+1)*support_size]
                context_i = self.context_net(x_i)
                context_i = context_i.mean(dim=0).expand(support_size, -1, -1, -1)
                x_i = torch.cat([x_i, context_i], dim=1)
                out.append(self.model(x_i))
            return torch.cat(out)
        else:            
            context = self.context_net(x) # Shape: batch_size, channels, H, W
            
            if self.zero_context:
                zero_context = context.clone()
                tmp = []
                for i, c_ in enumerate(zero_context):
                    if i > 0 and (i+1) % 5 == 0:
                        tmp.append(torch.zeros_like(c_))
                        continue
                    tmp.append(c_)
                zero_context = torch.stack(tmp).to(self.device)
                zero_context = zero_context.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))

            context = context.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))

            if self.model.training is False and self.online:               

                if self.zero_context:

                    num_order = []
                    sum = 0
                    for i in range(1, support_size+1):
                        sum += 1
                        if i % 5 == 0:
                            sum -= 1
                        num_order.append(sum)

                    length_tensor_zero = torch.tensor(num_order).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device) 
                    
                    zero_context = zero_context.cumsum(dim=1)
                    zero_context = torch.div(zero_context, length_tensor_zero.detach())
                    zero_context = zero_context.reshape((meta_batch_size * support_size, self.n_context_channels, h, w)) # meta_batch_size * support_size, context_size           

                context = context.cumsum(dim=1)
                length_tensor = torch.arange(1, support_size+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)                                                  
                context = torch.div(context, length_tensor.detach())           
                context = context.reshape((meta_batch_size * support_size, self.n_context_channels, h, w)) # meta_batch_size * support_size, context_size           

            else:
                context = context.mean(dim=1) # Shape: meta_batch_size, self.n_context_channels
                context = torch.repeat_interleave(context, repeats=support_size, dim=0) # meta_batch_size * support_size, context_size            

            if self.zero_context:
                # zero_context = context.clone()
                # zero_context[4::5, :, :, :] = 0
                x = torch.cat([x, zero_context], dim=1)
            else:
                x = torch.cat([x, context], dim=1)

            if self.model.training is False and self.model.__class__.__name__[-3:] == 'UNC':
                self.model.eval()
                enable_dropout(self.model)
                outputs = []
                for _ in range(self.T):
                    outputs.append(self.model(x))
                probs = F.softmax(torch.stack(outputs, dim=0), dim=-1).mean(dim=0)
                return probs

            return self.model(x)

    def predict_with_context(self, x, context=None, train=False):
        batch_size, c, h, w = x.shape

        meta_batch_size, support_size = 1, batch_size * batch_size
  
        #context - Shape: batch_size * batch_size, channels, H, W    
        # support_size * support_size, context_size
        self.eval()
        self.context_net.eval()
        self.model.eval()
        x = torch.cat([x, context], dim=1)
        with torch.no_grad():
            return self.model(x), None, None

class ARM_CUSUM(ERM):

    def __init__(self, model, loss_fn, device, context_net, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.context_net = context_net      
        self.adapt_bn = hparams['adapt_bn']
        self.affine_on = hparams['affine_on']
        self.T = hparams['T']
        self.beta = nn.Parameter(torch.tensor(hparams['beta']), requires_grad=True)

        self.context_norm = None
        if hparams['norm_type'] == 'layer':
            _, h, w = hparams['input_shape']
            self.context_norm = nn.LayerNorm((hparams['n_context_channels'], h, w),
                                             elementwise_affine=bool(self.affine_on),
                                             device=device
                                             )
        elif hparams['norm_type'] == 'instance':
            self.context_norm = nn.InstanceNorm3d(hparams['n_context_channels'], 
                                                  affine=False,
                                                  device=device)
            
        self.support_size = hparams['support_size']
        self.n_context_channels = hparams['n_context_channels']
        self.normalize = hparams['normalize']
        self.norm_type = hparams['norm_type']
        self.input_shape = hparams['input_shape']
        self.zero_context = hparams['zero_context']

        params = list(self.model.parameters()) + list(self.context_net.parameters())
        if self.affine_on:
            params += list(self.context_norm.parameters())

        self.init_optimizers(params)


    def predict(self, x):
        # get current input size
        batch_size, c, h, w = x.shape

        # get number of groups and support size
        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
            
        # compute sample-wise context (batch_size, channels, H, W)
        ctx = self.context_net(x)

        # reshape ctx
        ctx = ctx.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))

        # normalize
        ctx = self.context_norm(ctx)

        # accumulate ctx (meta_batch_size, support_size, channels, h, w)
        ctx = ctx.cumsum(dim=1)

        # normalize
        ctx = self.context_norm(ctx)

        # reshape ctx (meta_batch_size * support_size, channels, h, w)
        ctx = ctx.reshape((meta_batch_size * support_size, self.n_context_channels, h, w))  

        # do prediction based on context
        x = torch.cat([x, ctx], dim=1)
        return self.model(x)

class ARM_UNC(ERM):

    def __init__(self, model, loss_fn, device, context_net, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.context_net = context_net
        self.affine_on = hparams['affine_on']
        self.n_context_channels = hparams['n_context_channels']
        _, h, w = hparams['input_shape']
        self.context_norm = nn.LayerNorm((hparams['n_context_channels'], h, w), elementwise_affine=bool(self.affine_on), device=device)

        self.support_size = hparams['support_size']
        self.eps = 1e-10
        self.beta = torch.zeros(1, requires_grad=True, device=device)
        self.tau = torch.zeros(1, requires_grad=True, device=device)
        self.bald = hparams['bald']
        self.adapt_bn = hparams['adapt_bn']
        self.is_offline = hparams['is_offline']
        self.zero_init = hparams['zero_init']
        self.zero_context = hparams['zero_context']

        if self.zero_init:
            self.context_init = torch.zeros(1, hparams['n_context_channels'], hparams['input_shape'][-2], hparams['input_shape'][-1], requires_grad=True, device=device)
        else:
            context_init = np.random.randn(1, hparams['n_context_channels'], hparams['input_shape'][-2], hparams['input_shape'][-1]) * np.sqrt(2.0/hparams['n_context_channels'])
            self.context_init = torch.tensor(context_init, requires_grad=True, device=device, dtype=torch.float)

        self.T = hparams['T']
        self.debug = hparams['debug']
        self.cxt_self_include = hparams['cxt_self_include']
      
        params = list(self.model.parameters())
        params += list(self.context_net.parameters())
        params += list(self.context_norm.parameters())
        # params += list(self.context_init)
        params.append(self.context_init)
        params.append(self.beta)
        params.append(self.tau)

        self.init_optimizers(params)

    def get_BALD_acquisition(self, y_T):

        expected_entropy = - torch.mean(torch.sum(y_T * torch.log(y_T + self.eps), dim=-1), dim=0) 
        expected_p = torch.mean(y_T, dim=0)
        entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + self.eps), dim=-1)
        return (entropy_expected_p - expected_entropy)

    # def predict(self, x, train=False):
    #     # get current input size
    #     batch_size, c, h, w = x.shape

    #     # get number of groups and support size
    #     if batch_size % self.support_size == 0:
    #         meta_batch_size, support_size = batch_size // self.support_size, self.support_size
    #     else:
    #         meta_batch_size, support_size = 1, batch_size

    #     # set training mode
    #     if train:
    #         # set all modules training mode
    #         self.train()
    #         enable_dropout(self.model)
    #     else:
    #         # set all module eval
    #         self.eval()
    #         # but turn on dropout
    #         enable_dropout(self.model)

    #     # compute sample-wise context (batch_size, channels, H, W)
    #     ctx = self.context_net(x)

    #     # reshape x and ctx
    #     x = x.reshape(meta_batch_size, support_size, c, h, w)
    #     ctx = ctx.reshape(meta_batch_size, support_size, self.n_context_channels, h, w)

    #     # accumulate ctx by uncertainty weights
    #     u_list, ent_list = [], []
    #     if self.cxt_self_include:
    #         ctx_list = [ctx.transpose(0, 1)[0]]
    #     else:
    #         ctx_list = [self.context_init.expand(meta_batch_size, -1, -1, -1)]
        
    #     # for each input data
    #     for idx, (x_t, ctx_t) in enumerate(zip(x.transpose(0, 1), ctx.transpose(0, 1))):

    #         x_ctx_t = torch.cat([x_t, self.context_norm(ctx_list[-1])], dim=1)

    #         # compute uncertainty
    #         with torch.no_grad():
    #             out_prob = []
    #             for _ in range(self.T):
    #                 out_prob.append(self.model(x_ctx_t) * torch.exp(self.tau))

    #         out_prob = F.softmax(torch.stack(out_prob, dim=0), dim=-1)            
    #         out_prob = torch.mean(out_prob, dim=0)
    #         entropy = torch.sum(-out_prob * torch.log2(out_prob + self.eps), dim=-1)

    #         u = torch.exp(-torch.exp(self.beta) * entropy)

    #         u_list.append(u); ent_list.append(entropy)
    #         ctx_list.append(u.reshape(-1, 1, 1, 1) * ctx_t + ctx_list[-1])

    #     # stack results
    #     u_list = torch.stack(u_list, dim=0).squeeze().tolist()
    #     ent_list = torch.stack(ent_list, dim=0).squeeze().tolist()

    #     # reshape context list (meta_batch_size, support_size, self.n_context_channels, h, w)
    #     self.last_ctx = ctx_list[-1] # support_size, self.n_context_channels, h, w
    #     ctx_list = torch.stack(ctx_list[:-1], dim=0).transpose(0, 1)

    #     # reshape input / context (meta_batch_size * support_size, self.n_context_channels, h, w)
    #     ctx_list = ctx_list.reshape(-1, self.n_context_channels, h, w)
    #     x = x.reshape(-1, c, h, w)

    #     x_ctx = torch.cat([x, self.context_norm(ctx_list)], dim=1)
    #     if train:
    #         self.train()
    #         return self.model(x_ctx)  * torch.exp(self.tau)
    #     else:
    #         out_prob = []
    #         for _ in range(self.T):
    #             out_prob.append(F.softmax(self.model(x_ctx) * torch.exp(self.tau), dim=-1))
    #         return torch.stack(out_prob, dim=0).mean(dim=0), u_list, ent_list

    def predict(self, x, train=False):
        # get current input size
        batch_size, c, h, w = x.shape

        # get number of groups and support size
        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size

        # set training mode
        if train:
            # set all modules training mode
            self.train()
            enable_dropout(self.model)
        else:
            # set all module eval
            self.eval()
            # but turn on dropout
            enable_dropout(self.model)

        if self.adapt_bn:
            self.train()

            out = []
            for i in range(meta_batch_size):
                x_i = x[i*support_size:(i+1)*support_size]
                context_i = self.context_net(x_i)

                # accumulate ctx by uncertainty weights
                ctx_list = [self.context_init]

                # for each input data
                for idx, (x_t, ctx_t) in enumerate(zip(x_i, context_i)):
                    x_ctx_t = torch.cat([x_t.unsqueeze(0), self.context_norm(ctx_list[-1])], dim=1)

                    # compute uncertainty
                    with torch.no_grad():
                        out_prob = []
                        for _ in range(self.T):
                            out_prob.append(self.model(x_ctx_t) * torch.exp(self.tau))
                    out_prob = F.softmax(torch.stack(out_prob, dim=0), dim=-1)            
                    out_prob = torch.mean(out_prob, dim=0)
                    entropy = torch.sum(-out_prob * torch.log2(out_prob + self.eps), dim=-1)
                    u = torch.exp(-torch.exp(self.beta) * entropy)
                    ctx_list.append(u.reshape(-1, 1, 1, 1) * ctx_t.unsqueeze(0) + ctx_list[-1])

                # reshape context list (support_size, self.n_context_channels, h, w)
                ctx_list = torch.stack(ctx_list[:-1], dim=0)
                ctx_list = ctx_list.reshape(-1, self.n_context_channels, h, w)

                x_ctx = torch.cat([x_i, self.context_norm(ctx_list)], dim=1)

                if train:
                    self.train()
                    out.append( self.model(x_ctx) * torch.exp(self.tau) )                    
                else:
                    out_prob = []
                    with torch.no_grad():
                        for _ in range(self.T):
                            out_prob.append(F.softmax(self.model(x_ctx) * torch.exp(self.tau), dim=-1))
                    out.append( torch.stack(out_prob, dim=0).mean(dim=0) )
            if train:
                return torch.cat(out)      
            else: 
                return torch.cat(out), None, None
        
        else:
            # compute sample-wise context (batch_size, channels, H, W)
            ctx = self.context_net(x)

            # reshape x and ctx
            x = x.reshape(meta_batch_size, support_size, c, h, w)
            ctx = ctx.reshape(meta_batch_size, support_size, self.n_context_channels, h, w)

            # accumulate ctx by uncertainty weights
            u_list, ent_list = [], []
            # if self.cxt_self_include:
            #     ctx_list = [ctx.transpose(0, 1)[0]]
            # else:
            ctx_list = [self.context_init.expand(meta_batch_size, -1, -1, -1)]
            
            # for each input data
            for idx, (x_t, ctx_t) in enumerate(zip(x.transpose(0, 1), ctx.transpose(0, 1))):

                x_ctx_t = torch.cat([x_t, self.context_norm(ctx_list[-1])], dim=1)

                # compute uncertainty
                with torch.no_grad():
                    out_prob = []
                    for _ in range(self.T):
                        out_prob.append(self.model(x_ctx_t) * torch.exp(self.tau))

                out_prob = F.softmax(torch.stack(out_prob, dim=0), dim=-1)            
                out_prob = torch.mean(out_prob, dim=0)
                entropy = torch.sum(-out_prob * torch.log2(out_prob + self.eps), dim=-1)

                u = torch.exp(-torch.exp(self.beta) * entropy)

                u_list.append(u); ent_list.append(entropy)
                ctx_list.append(u.reshape(-1, 1, 1, 1) * ctx_t + ctx_list[-1])

            # stack results
            u_list = torch.stack(u_list, dim=0).squeeze().tolist()
            ent_list = torch.stack(ent_list, dim=0).squeeze().tolist()

            # reshape context list (meta_batch_size, support_size, self.n_context_channels, h, w)
            self.last_ctx = ctx_list[-1] # support_size, self.n_context_channels, h, w            
            if self.is_offline:
                ctx_list = self.last_ctx.unsqueeze(0).repeat_interleave(repeats=support_size, dim=0).transpose(0, 1)
            else:
                ctx_list = torch.stack(ctx_list[:-1] if not self.cxt_self_include else ctx_list[1:], dim=0).transpose(0, 1) # [meta_batch_size, support_size, c, h, w]

            # reshape input / context (meta_batch_size * support_size, self.n_context_channels, h, w)
            ctx_list = ctx_list.reshape(-1, self.n_context_channels, h, w)
            x = x.reshape(-1, c, h, w)

            x_ctx = torch.cat([x, self.context_norm(ctx_list)], dim=1)
            if train:
                self.train()
                return self.model(x_ctx)  * torch.exp(self.tau)
            else:
                out_prob = []
                for _ in range(self.T):
                    out_prob.append(F.softmax(self.model(x_ctx) * torch.exp(self.tau), dim=-1))
                return torch.stack(out_prob, dim=0).mean(dim=0), u_list, ent_list

    def learn(self, images, labels, group_ids=None, mask=None, mask_p=1.0):
        self.train()
        # Forward
        logits = self.predict(images, train=True)

        loss = self.loss_fn(logits, labels)

        self.update(loss)
        stats = { 'objective': loss.detach().item(), }
        return logits, stats



class ARM_LL(ERM):

    def __init__(self, model, loss_fn, device, learned_loss_net, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.support_size = hparams['support_size']
        self.learned_loss_net = learned_loss_net
        self.n_inner_iter = 1
        self.inner_lr = 1e-1
        self.inner_opt = torch.optim.SGD(model.parameters(),
                                        lr=self.inner_lr)

        params = list(self.model.parameters()) + list(self.learned_loss_net.parameters())
        self.init_optimizers(params)

    def predict(self, x, labels=None, backprop_loss=False):

        self.train() # see this thread for why this is done https://github.com/facebookresearch/higher/issues/55

        n_domains = math.ceil(len(x) / self.support_size)

        logits = []
        loss = []
        for domain_id in range(n_domains):
            start = domain_id*self.support_size
            end = start + self.support_size
            end = min(len(x), end) # in case final domain has fewer than support size samples

            domain_x = x[start:end]

            with higher.innerloop_ctx(
                self.model, self.inner_opt, copy_initial_weights=False) as (fnet, diffopt):

                # Inner loop
                for _ in range(self.n_inner_iter):
                    spt_logits = fnet(domain_x)
                    spt_loss = self.learned_loss_net(spt_logits)
                    diffopt.step(spt_loss)

                # Evaluate
                domain_logits = fnet(domain_x)
                logits.append(domain_logits)

                if backprop_loss and labels is not None:
                    domain_labels = labels[start:end]
                    domain_loss = self.loss_fn(domain_logits, domain_labels)
                    domain_loss.backward()
                    loss.append(domain_loss.to('cpu').detach().item())


        logits = torch.cat(logits)

        if backprop_loss:
            return logits, np.mean(loss)
        else:
            return logits


    def learn(self, x, labels, group_ids=None):
        self.train()
        logits, loss = self.predict(x, labels, backprop_loss=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = {}

        return logits, stats


class ARM_BN(ERM):

    def __init__(self, model, loss_fn, device, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.support_size = hparams['support_size']

    def predict(self, x):
        self.model.train()

        n_domains = math.ceil(len(x) / self.support_size)

        logits = []
        for domain_id in range(n_domains):
            start = domain_id * self.support_size
            end = start + self.support_size
            end = min(len(x), end) # in case final domain has fewer than support size samples
            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)

        logits = torch.cat(logits)

        return logits

class MMD(ERM):

    def my_cdist(cls, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(cls, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = cls.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(cls, x, y):
        Kxx = cls.gaussian_kernel(x, x).mean()
        Kyy = cls.gaussian_kernel(y, y).mean()
        Kxy = cls.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def __init__(self, model, loss_fn, device, hparams, num_classes):

        featurizer = model
        classifier = nn.Linear(model.num_features, num_classes)
        model = nn.Sequential(featurizer, classifier).to(device)
        super().__init__(model, loss_fn, device, hparams)

        self.featurizer = featurizer.to(device)
        self.classifier = classifier.to(device)

        self.gamma = hparams['gamma']
        self.support_size = hparams['support_size']

    def learn(self, images, labels, group_ids):
        self.train()

        features = self.featurizer(images)
        logits = self.classifier(features)
        loss = self.loss_fn(logits, labels)
        penalty = 0

        num_domains = math.ceil(len(images) / self.support_size) # meta batch size

        for i in range(num_domains):
            for j in range(i, num_domains):
                penalty_add = self.mmd(features[i*self.support_size:(i+1)*self.support_size], features[j*self.support_size:(j+1)*self.support_size])
                penalty += penalty_add
        penalty /= (num_domains * (num_domains - 1) / 2)
        objective = loss + self.gamma * penalty
        self.update(objective)

        stats = {'loss': loss,
                 'objective': objective}


        return logits, stats


class DANN(ERM):
    def __init__(self, model, loss_fn, device, hparams, num_domains, num_classes):

        featurizer = model
        classifier = nn.Linear(model.num_features, num_classes)

        model = nn.Sequential(featurizer, classifier).to(device)

        super().__init__(model, loss_fn, device, hparams, init_optim=False)

        self.discriminator = nn.Sequential(
                nn.Linear(featurizer.num_features, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_domains),
              ).to(device)

        self.num_domains = num_domains
        self.featurizer = featurizer.to(device)
        self.classifier = classifier.to(device)

        self.support_size = hparams['support_size']
        self.d_steps_per_g_step = hparams['d_steps_per_g_step']
        self.lambd = hparams['lambd']

        self.step_count = 0

        self.init_optimizers(None)


    def init_optimizers(self, params):
        if self.optimizer_name == 'adam': # This is used for MNIST.
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate, weight_decay=self.weight_decay)
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay,
                                              momentum=0.9)
            self.disc_optimizer = torch.optim.SGD(self.discriminator.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay,
                                                momentum=0.9)

    def learn(self, images, labels, group_ids):
        self.train()

        # Forward
        features = self.featurizer(images)

        domain_preds = self.discriminator(features)
        group_ids = group_ids.to(self.device)
        disc_loss = self.loss_fn(domain_preds, group_ids)

        # Backprop
        if self.step_count % (self.d_steps_per_g_step + 1) < self.d_steps_per_g_step:

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            self.step_count += 1
            return None, None
        else:
            logits = self.classifier(features)

            # Evaluate
            loss = self.loss_fn(logits, labels)
            gen_loss = loss - self.lambd * disc_loss
            self.optimizer.zero_grad()
            gen_loss.backward()
            self.optimizer.step()

            stats = {}
            self.step_count += 1
            return logits, stats


