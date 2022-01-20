.PHONY: vidvrd_sg_eval
vidvrd_sg_eval:
	python tools/test_net_rel.py \
		--dataset vidvrd \
		--cfg configs/vidvrd/vidvrd_res101xi3d50_pred_boxes_flip_dc5_2d_new.yaml \
		--load_ckpt checkpoints/model_step12999.pth \
		--output_dir results/vidvrd \
		--do_eval
