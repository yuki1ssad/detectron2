# Task 1 class : 0-20
# train
# CUDA_VISIBLE_DEVICES=0,1 python projects_my/Featurized-QueryRCNN/train_net.py --num-gpus 2 --resume --config-file projects_my/Featurized-QueryRCNN/configs/OW/t1_train.yaml OUTPUT_DIR "projects_my/Featurized-QueryRCNN/output/t1"

# Task 2 class : 20-40
# CUDA_VISIBLE_DEVICES=0,1 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t2_train.yaml OUTPUT_DIR "myOutput/t2" MODEL.WEIGHTS "myOutput/t1/model_final.pth"
# CUDA_VISIBLE_DEVICES=0,1 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t2_ft.yaml OUTPUT_DIR "myOutput/t2_ft" MODEL.WEIGHTS "myOutput/t2/model_final.pth"


# # Task 3 class : 40-60
# CUDA_VISIBLE_DEVICES=0,1 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t3_train.yaml OUTPUT_DIR "myOutput/t3" MODEL.WEIGHTS "myOutput/t2_ft/model_final.pth"
# CUDA_VISIBLE_DEVICES=2,3 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --dist-url tcp://127.0.0.1:53027 --config-file projects/QueryRCNN_OW/configs_owod/t3_ft.yaml OUTPUT_DIR "myOutput_baseline_ota_uch_contrast/t3_ft_openNum20" MODEL.WEIGHTS "myOutput_baseline_ota_uch_contrast/t3/model_final.pth" OWOD.OPEN_SET_NUM_PERITER 20

# # Task 4 class : 60-80
# CUDA_VISIBLE_DEVICES=0,1 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t4_train.yaml OUTPUT_DIR "myOutput/t4" MODEL.WEIGHTS "myOutput/t3_ft/model_final.pth"
# CUDA_VISIBLE_DEVICES=0,1 python projects/QueryRCNN_OW/train_net.py --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t4_ft.yaml OUTPUT_DIR "myOutput/t4_ft" MODEL.WEIGHTS "myOutput/t4/model_final.pth"

# CUDA_VISIBLE_DEVICES=0,1 python -m debugpy --listen 5555 --wait-for-client projects/QueryRCNN_OW/train_net.py --num-gpus 2 --eval-only --resume --config-file projects/QueryRCNN_OW/configs_owod/t4_ft.yaml OUTPUT_DIR "output_fqrcnn_ow/t4_ft_0.72iter_t4_3.6iter/"







# # debug
# CUDA_VISIBLE_DEVICES=0,1 python -m debugpy --listen 5555 --wait-for-client projects/QueryRCNN_OW/train_net.py --num-gpus 2 --dist-url tcp://127.0.0.1:53025 --config-file projects/QueryRCNN_OW/configs_owod/t1_train.yaml OUTPUT_DIR "./output_fqrcnn_ow/debug/test"
# CUDA_VISIBLE_DEVICES=0,1 python -m debugpy --listen 5555 --wait-for-client projects/QueryRCNN_OW/train_net.py --eval-only --num-gpus 2 --config-file projects/QueryRCNN_OW/configs_owod/t1_train.yaml OUTPUT_DIR "./output_fqrcnn_ow/debug" SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS "output_fqrcnn_ow/t1_5.4iter/simOTA_nearest_uch_mem40_contrast_rep1/model_final.pth"
# python -m debugpy --listen 5555 --wait-for-client
# --dist-url tcp://127.0.0.1:53025

