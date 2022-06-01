import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()


    def forward(self, logits_per_image, logits_per_text):
        # Set ground truth - see clip paper page 4
        # ground_truth = torch.arange(BATCH_SIZE).to(self.device)
        ground_truth = (
            torch.arange(len(logits_per_image)).type_as(logits_per_image).long()
        )

        return (
            self.loss_img(logits_per_image, ground_truth)
            + self.loss_txt(logits_per_text, ground_truth)
        ) / 2

class SIMCLRLoss(nn.Module):
    """
    Code from facebook - SLIP Paper
    (This project is under the CC-BY-NC 4.0 license.)
    https://github.com/facebookresearch/SLIP/blob/main/losses.py

    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, device, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.device = device
        #self.labels = None
        #self.masks = None
        #self.last_local_batch_size = None

    def forward(self, aug1_embed, aug2_embed):
        #q_a = outputs['aug1_embed']
        #q_b = outputs['aug2_embed']
        q_a = aug1_embed
        q_b = aug2_embed

        #q_a = F.normalize(q_a, dim=-1, p=2)
        #q_b = F.normalize(q_b, dim=-1, p=2)

        #local_batch_size = q_a.size(0)

        k_a, k_b = q_a, q_b

        #k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        #if local_batch_size != self.last_local_batch_size:
        #    self.labels = local_batch_size * utils.get_rank() + torch.arange(
        #        local_batch_size, device=q_a.device
        #    )
        #    total_batch_size = local_batch_size * utils.get_world_size()
        #    self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
        #    self.last_local_batch_size = local_batch_size
        
        labels = (
            torch.arange(len(q_a)).type_as(q_a).long()
        )
        labels = labels.to(self.device)
        # masks = F.one_hot(labels, len(q_a)) * 1e9
        masks = torch.eye(len(q_a)) * 1e9
        masks = masks.to(self.device)
        
        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        logit1 = torch.cat([logits_ab, logits_aa], dim=1)
        logit2 = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = F.cross_entropy(logit1, labels)
        loss_b = F.cross_entropy(logit2, labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        #with torch.no_grad():
        #    pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
        #    correct = pred.eq(labels).sum()
        #    acc = 100 * correct / len(q_a)

        #return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}
        return loss