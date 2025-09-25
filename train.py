from rfdetr import RFDETRSmall, RFDETRNano
from rfdetr.config import RFDETRSmallConfig, RFDETRNanoConfig

if __name__ == "__main__":
    config = RFDETRNanoConfig()
    
    # output size != 4 ==>
    exclude_keys = [f"transformer.enc_out_bbox_embed.{i}.layers.2.weight" for i in range(config.group_detr)]
    exclude_keys += [f"transformer.enc_out_bbox_embed.{i}.layers.2.bias" for i in range(config.group_detr)]
    exclude_keys += ["transformer.decoder.ref_point_head.layers.0.weight" , "bbox_embed.layers.2.weight", "bbox_embed.layers.2.bias"]
    
    # dec_n_points != 2 ==>
    exclude_keys += [f"transformer.decoder.layers.{i}.cross_attn.{b}.{c}" for i in range(config.dec_layers) for b in ["sampling_offsets", "attention_weights"] for c in ["weight", "bias"]]
    model = RFDETRNano(lite_refpoint_refine=False, num_boxes_per_query=2, pretrain_exclude_keys=exclude_keys, dec_n_points=4)

    model.train(
        dataset_dir="../playdarts/dataset_dual_corners",
        # eval=True,
        epochs=15,
        batch_size=20,
        grad_accum_steps=1,
        lr=5e-4,
        output_dir="outputs/output_dual_4_points_weighted_loss_more_data",
        wandb=False,
        num_workers=2,
        lr_drop=8,
        # resume="outputs/output_dual_4_points_weighted_loss/checkpoint.pth",
    )