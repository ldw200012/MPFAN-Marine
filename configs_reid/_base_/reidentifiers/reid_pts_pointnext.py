num_classes  = 10 * 2

output_feat_size = 64 # (original) = 64

hidden_size = output_feat_size * 2
hidden_size_match = output_feat_size * 2
# 2 for avg and maxpool and 2 for concatenating the features of each object

model = dict(
    losses_to_use=dict(kl=False,match=False,cls=False,shape=False,fp=False,triplet=False),
    alpha=dict(kl=1,match=1,cls=1,shape=1,fp=1,vis=1,triplet=1,),

    triplet_margin=10,
    triplet_p=2,
    triplet_sample_num=128,
    use_o=False,

    type='ReIDNet',
    numpoints=[128,64,32],
    output_feat_size=output_feat_size,
    num_classes=num_classes,
    use_dgcnn=False,
    
    backbone=dict(type='PointNeXt'),

    match_head=[dict(type='LinearRes', n_in=hidden_size_match, n_out=hidden_size_match, norm='GN',ng=8),
                dict(type='Linear', in_features=hidden_size_match, out_features=1)],
    cls_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=16),
              dict(type='Linear', in_features=hidden_size, out_features=num_classes)],
    fp_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=16),
              dict(type='Linear', in_features=hidden_size, out_features=1)],
    shape_head=[
        dict(type='Conv1d', in_channels=hidden_size, out_channels=1024, kernel_size=output_feat_size//2),
        dict(type='BatchNorm1d', num_features=1024),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=1024, out_channels=2048, kernel_size=output_feat_size//4),
        dict(type='BatchNorm1d', num_features=2048),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=2048, out_channels=2048, kernel_size=output_feat_size//4),
    ],
    # shape_head=dict(),
    downsample=None,
    cross_stage1=dict(type='corss_attention',d_model=output_feat_size,nhead=2,attention='linear'),
    cross_stage2=dict(type='corss_attention',d_model=output_feat_size,nhead=2,attention='linear'),
    local_stage1=dict(type='local_self_attention',d_model=output_feat_size,nhead=2,attention='linear',knum=48,pos_size=output_feat_size),
    local_stage2=dict(type='local_self_attention',d_model=output_feat_size,nhead=2,attention='linear',knum=48,pos_size=output_feat_size),
)