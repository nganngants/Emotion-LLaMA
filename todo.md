Note: Needed following features: FaceMAE_feats, VideoMAE_feats, and Audio_feats, which correspond to the features from the Local Encoder, Temporal Encoder, and Audio Encoder mentioned in the paper. Local Encoder uses the MAE model pre-trained on MER2023. Temporal Encoder uses the VideoMAE model pre-trained on MER2023. MAE-DFER (which we didn't use) is a model pre-trained on VoxCeleb2 that can extract good dynamic facial expression features, which corresponds to the Temporal Encoder in our work. The Audio Encoder uses the HuBERT-chinese model. Since most of the datasets involved in our work are in Chinese, we chose encoders that were mostly pre-trained on Chinese datasets.

https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth

* [ ] FaceMAE_feats: MAE model https://github.com/zeroQiaoba/MERTools/blob/master/MER2023/feature_extraction/visual/extract_manet_embedding.py
* [ ] VideoMAE_feats: VideoMAE model
* [ ] Audio_feats: HuBERT model (english)