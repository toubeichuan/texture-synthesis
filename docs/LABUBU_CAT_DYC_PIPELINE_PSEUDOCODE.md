# Labubu to Cat DYC Pipeline Pseudocode

```text
function RUN_LABUBU_CAT_PIPELINE(config):
    timestamp = local_time_to_minute("%Y%m%d_%H%M")
    run_name = "labubu_cat_dyc_" + timestamp

    paths = expand_templates(config, run_name)
    assert paths.texture_root does not exist
    assert paths.scene_dir does not exist
    assert paths.model_dir does not exist
    assert paths.render_dir does not exist
    create paths.run_root
    save resolved_config and run_manifest immediately

    # Stage 1: DYC texture-view generation, always from original inputs.
    assert config.texture.implementation == "easi-dyc"
    assert input == easi-dyc/data/meshes/cat/cat.obj
    assert reference == easi-dyc/data/texture_images/round_bird/labubu.jpg
    assert reference_mask == labubu_mask.jpg
    run in texture-synthesis environment:
        cd easi-dyc
        python scripts/generate_texture_frozenview0_labubu.py \
            --input_dir data/meshes/cat \
            --obj_file cat.obj \
            --style_img data/texture_images/round_bird/labubu.jpg \
            --style_mask data/texture_images/round_bird/labubu_mask.jpg \
            --output_dir <timestamped texture_root> \
            --use_shapenet --num_viewpoints 1 --update_steps 0 \
            --ip_adapter_strength 0.6 --controlnet_cond canny ...
    texture_run_dir = find exactly one new args.json under texture_root
    assert numeric generate/inpainted images are exactly 0.png..9.png

    # Stage 2: Build camera data independently from the DYC preset.
    # Do not use the update-stage viewpoints.json because locked-view updates
    # do not describe the ten principal generation cameras.
    run export_camera_poses.py:
        --preset shapenet --dist 0.8
        --out <run_root>/camera/camera_poses.json
    run camera_poses_to_transforms.py:
        --input camera_poses.json
        --out transforms_train.json
        --camera-angle-x 1.413716694
        --file-prefix ./train --file-mode index
        --rotate-x-deg 90
    assert transforms contain 10 unique poses

    # Stage 3: Prepare mask/similarity-filtered supervision.
    for index in 0..9:
        image = generate/inpainted/<index>.png
        old = generate/mask/<index>_old.png
        update = generate/mask/<index>_update.png
        new = generate/mask/<index>_new.png
        similarity = generate/similarity/<index>.png

        foreground = binary(old OR update OR new)
        valid = foreground AND (similarity >= 0.1 * 255)
        valid_ratio = area(valid) / max(area(foreground), 1)
        if valid_ratio < 0.65:
            reject this image and its matching camera frame
        else:
            valid = dilate(valid, 2 pixels)
            valid = erode(valid, 1 pixel)
            cleaned = composite(image, white_background, valid)
            save cleaned as scene/train/<new_index>.png
            copy matching transform frame and renumber file_path

    assert remaining images == remaining camera frames
    assert every retained camera pose is unique
    assert at least the configured minimum number of views remains

    # Stage 4: Use original geometry, not DYC's final textured mesh.
    original_mesh = load easi-dyc/data/meshes/cat/cat.obj
    center = (bounds_max + bounds_min) / 2
    max_extent = max(bounds_max - bounds_min)
    original_mesh.vertices = (vertices - center) / max_extent
    save geometry-only mesh as scene/mesh.obj
    # This matches DYC's render-space normalization while preserving the
    # original faces. Raw, unnormalized cat.obj must not be copied directly.

    write scene/transforms_train.json from retained frames
    write test/val transforms according to smoke-test policy
    write dataset manifest with hashes of source image, source mesh and outputs

    # Stage 5: Required black box. Rendering cannot happen without a model.
    TRAIN_GS_WITHOUT_EXPOSING_INTERNAL_PROGRESS(
        scene_dir=scene,
        model_dir=timestamped model_dir,
        gs_type="gs_mesh",
        num_splats=5,
        iterations=30000
    )

    # Stage 6: Render only the checkpoint produced by this run.
    assert checkpoint belongs to run_name
    render train and test cameras into timestamped render_dir
    write final manifest and visualization index
    return all timestamped paths
```

The training call is intentionally treated as an opaque required step. Omitting
it entirely would make Gaussian rendering impossible because there would be no
checkpoint to load.
