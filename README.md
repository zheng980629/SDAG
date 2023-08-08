# Learning Semantic Degradation-Aware Guidance for Recognition-Driven Unsupervised Low-Light Image Enhancement (AAAI2023)

[Paper link](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=aL_WRTkAAAAJ&citation_for_view=aL_WRTkAAAAJ:_FxGoFyzp5QC)

## How to train SDAG
 ```
python train.py -opt /SDAG/options/train/SemanticAwareRecon.yml
```

## How to integrate the SDAG into an existing enhancer

2. Define the SDAG network and load the pre-trained model.
```
SemanticNet = SemanticAwareNet(channels=64).cuda()
SemanticNet.load_state_dict(torch.load('/gdata1/zhengns/checkpoint/PersonalizedEnhancement/experiments/SemanticAware_margin05_lqRecon_NetC_selfSupervised/models/net_g_78000.pth')['params'])

```

2. Self-reconstructing the enhanced image from the unsupervised enhancer for measuring degradations in the semantic level.
```
semanticPrior = SemanticNet(enhanced_image) # 
```

3. Employing SDAG as a regularization term.
```
loss_semantic = F.l1_loss(enhanced_image, semanticPrior, reduction='mean')
```