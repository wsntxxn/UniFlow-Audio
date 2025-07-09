if [ ! -d data/vggsound/logs ]; then
    mkdir -p data/vggsound/logs
fi

CUDA_VISIBLE_DEVICES=0 python data_preprocess/vggsound/prepare_clip.py --part_index 1 > data/vggsound/logs/prepare_clip_part01.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python data_preprocess/vggsound/prepare_clip.py --part_index 2 > data/vggsound/logs/prepare_clip_part02.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python data_preprocess/vggsound/prepare_clip.py --part_index 3 > data/vggsound/logs/prepare_clip_part03.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python data_preprocess/vggsound/prepare_clip.py --part_index 4 > data/vggsound/logs/prepare_clip_part04.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python data_preprocess/vggsound/prepare_clip.py --part_index 5 > data/vggsound/logs/prepare_clip_part05.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python data_preprocess/vggsound/prepare_clip.py --part_index 6 > data/vggsound/logs/prepare_clip_part06.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python data_preprocess/vggsound/prepare_clip.py --part_index 7 > data/vggsound/logs/prepare_clip_part07.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python data_preprocess/vggsound/prepare_clip.py --part_index 8 > data/vggsound/logs/prepare_clip_part08.log 2>&1 &
wait
echo "All parts of VGGSound CLIP features prepared successfully."
