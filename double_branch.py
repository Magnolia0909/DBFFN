import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from vit_pytorch import vit

class DBranch(nn.Module):
    def __init__(self, args):
        super(DBranch, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.visual_extractor_02 = args.visual_extractor_02
        self.visual_extractor_02_image_size = args.visual_extractor_02_image_size
        self.visual_extractor_02_patch_size = args.visual_extractor_02_patch_size
        self.visual_extractor_02_num_classes = args.visual_extractor_02_num_classes
        self.visual_extractor_02_dim = args.visual_extractor_02_dim
        self.visual_extractor_02_depth = args.visual_extractor_02_depth
        self.visual_extractor_02_heads = args.visual_extractor_02_heads
        self.visual_extractor_02_dropout = args.visual_extractor_02_dropout
        self.visual_extractor_02_mlp_dim = args.visual_extractor_02_mlp_dim
        self.visual_extractor_02_emd_dropout = args.visual_extractor_02_emd_dropout

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        # PiT && ViT
        # model_02 = getattr(mae, self.visual_extractor_02)(image_size=self.visual_extractor_02_image_size,
        #                                                     patch_size=self.visual_extractor_02_patch_size,
        #                                                     num_classes=self.visual_extractor_02_num_classes,
        #                                                     dim=self.visual_extractor_02_dim,
        #                                                     depth=self.visual_extractor_02_depth,
        #                                                     heads=self.visual_extractor_02_heads,
        #                                                     mlp_dim=self.visual_extractor_02_mlp_dim,
        #                                                     dropout=self.visual_extractor_02_dropout,
        #                                                     emb_dropout=self.visual_extractor_02_emd_dropout
        #                                                    )

        # ViT && ViT_with_patch_merger
        model_02 = getattr(vit, self.visual_extractor_02)(image_size=self.visual_extractor_02_image_size,
                                                            patch_size=self.visual_extractor_02_patch_size,
                                                            num_classes=self.visual_extractor_02_num_classes,
                                                            dim=self.visual_extractor_02_dim,
                                                            depth=self.visual_extractor_02_depth,
                                                            heads=self.visual_extractor_02_heads,
                                                            mlp_dim=self.visual_extractor_02_mlp_dim,
                                                            dropout=self.visual_extractor_02_dropout,
                                                            emb_dropout=self.visual_extractor_02_emd_dropout)

        # model_02.classifier = None

        modules = list(model.children())[:-2]
        modules_02 = list(model_02.children())[:-1]
        # num_ftrs = model .classifier.in_features
        # # model.classifier = nn.Sequential(
        # #     # nn.Linear(num_ftrs, 14),
        # #     nn.Sigmoid()
        # # )
        # modules = list(model.children())

        self.model = nn.Sequential(*modules)
        self.model_02 = nn.Sequential(*modules_02)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.concat = nn.Sequential(
            nn.Linear(self.visual_extractor_02_mlp_dim*2, self.visual_extractor_02_mlp_dim),
            nn.LeakyReLU(),
        )

        self.W = nn.Linear (self.visual_extractor_02_mlp_dim, 2*self.visual_extractor_02_mlp_dim)
        self.U = nn.Linear(self.visual_extractor_02_mlp_dim, 2*self.visual_extractor_02_mlp_dim)

    def patch_reshape(self, size, x):
        x = x.cuda()
        _, _, x_dim = x.shape
        result = nn.Linear(x_dim, size).cuda()
        x = result(x)
        return x
    def avg_reshape(self, size, x):
        x = x.cuda()
        _, x_dim = x.shape
        result = nn.Linear(x_dim, size).cuda()
        x = result(x)
        return x

    def forward(self, images):

        patch_feats = self.model(images)
        patch_feats_02 = self.model_02(images)

        # Resnet
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        p_batch, p_middle, p_feat = patch_feats.shape
        avg, avg_feat = avg_feats.shape
        #
        # avg_feats_02 = self.avg_fnt(patch_feats_02).squeeze().reshape(-1, patch_feats_02.size(1))
        # avg_feats_02 = avg_feats_02.repeat(1, 2)
        # batch_size_02, feat_size_02, _, _ = patch_feats_02.shape
        # patch_feats_02 = patch_feats.reshape(batch_size_02, -1, res_size)

        # Densenet + Resnet
        # batch_size_02, feat_size_02, _, _ = patch_feats_02.shape
        # patch_feats_02 = patch_feats.reshape(batch_size_02, -1, p_feat )

        # ViT
        # patch_feats_02 = patch_feats_02.reshape(8, -1, 2048)
        # avg_feats_02 = patch_feats_02

        # ViT + Resnet
        patch_feats_02 = self.patch_reshape(p_middle, patch_feats_02).permute(0,2,1)
        patch_feats_02 = self.patch_reshape(p_feat, patch_feats_02)
        avg_feats_02 = patch_feats_02.reshape(-1,p_feat).permute(1,0)
        avg_feats_02 = self.avg_reshape(avg, avg_feats_02).permute(1,0)

       # Densenet + Resnet
       #  avg_feats_02 =  self.avg_fnt(patch_feats).squeeze().reshape(avg, -1)
       #  avg_feats_02 = self.avg_reshape(avg_feat, avg_feats_02)

        # Densenet + Resnet && ViT + Resnet
        # cat_patch_feat = torch.cat((patch_feats, patch_feats_02), 2)
        # cat_avg_feat = torch.cat((avg_feats, avg_feats_02), 1)
        # cat_patch_feat = self.concat(cat_patch_feat)
        # cat_avg_feat = self.concat(cat_avg_feat)
        # return cat_patch_feat, cat_avg_feat

        # patch_feats_gate = patch_feats  + patch_feats_02
        # gates = self.W(patch_feats) + self.U(torch.tanh(patch_feats_02))
        # gates = torch.split(gates, split_size_or_sections=self.visual_extractor_02_mlp_dim, dim=2)
        # input_gate, forget_gate = gates
        # input_gate = F.log_softmax(input_gate)
        # forget_gate = F.log_softmax(forget_gate)
        # patch_feats_gate = input_gate * torch.tanh(patch_feats_gate) + forget_gate * patch_feats_02
        #
        # avg_feats_gate = avg_feats + avg_feats_02
        # avg_gates = self.W(avg_feats) + self.U(torch.tanh(avg_feats_02))
        # avg_gates = torch.split(avg_gates, split_size_or_sections=self.visual_extractor_02_mlp_dim, dim=1)
        # avg_input_gate, avg_forget_gate = avg_gates
        # avg_input_gate = F.log_softmax(avg_input_gate)
        # avg_forget_gate = F.log_softmax(avg_forget_gate)
        # avg_feats_gate = avg_input_gate * torch.tanh(avg_feats_gate) + avg_forget_gate * avg_feats_02
        # # print("patch_feat02", patch_feats_02.size())
        #
        # return 0.8*patch_feats+0.1*patch_feats_02+0.1*patch_feats_gate,\
        #        0.8*avg_feats+0.1*avg_feats_02+0.1*avg_feats_gate

        return patch_feats  + patch_feats_02, avg_feats + avg_feats_02
        # return patch_feats, avg_feats