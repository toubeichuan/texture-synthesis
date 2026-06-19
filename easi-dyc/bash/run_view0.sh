OBJ_NAME="labubu"
TEXT_PROMPT="American Shorthair,yellow-green eyes, fluffy paws, A cat with black and silver stripes, white background,photorealistic, high quality texture"
CAM_DIST=0.8

STYLE_IMAGE="round_bird/labubu.jpg"
IP_STRENGTH=1.0
CN_STRENGTH=1.0

python scripts/generate_texture.py \
    --input_dir "data/meshes/${OBJ_NAME}" \
    --output_dir "outputs_labubu" \
    --obj_file "${OBJ_NAME}.obj" \
    --prompt "${TEXT_PROMPT}" \
    --style_img "data/texture_images/${STYLE_IMAGE}" \
    --style_img_bg_color 255 255 255 \
    --ip_adapter_path "./ip_adapter" \
    --ip_adapter_strength $IP_STRENGTH \
    --ip_adapter_n_tokens 16 \
    --controlnet_cond "canny" \
    --controlnet_strength $CN_STRENGTH \
    --use_cc_edges True \
    --use_depth_edges True \
    --use_normal_edges True \
    --add_view_to_prompt \
    --ddim_steps 50 \
    --guidance_scale 10 \
    --new_strength 1 \
    --update_strength 0.4 \
    --view_threshold 0.1 \
    --blend 0 \
    --dist $CAM_DIST \
    --num_viewpoints 1 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 20 \
    --update_mode heuristic \
    --seed 42 \
    --post_process \
    --tex_resolution "1k" \
    --use_objaverse \
    --override_rendering_0 "data/texture_images/round_bird/ipadapter_result_00.png" \
    --skip_view0_new_stage True
