import mitsuba as mi
import numpy as np
import drjit as dr
import os
from os.path import realpath, join

def mse(image, image_ref):
    """Compute the mean squared error of an image."""
    return dr.mean(dr.square(image - image_ref))

def scalar_rgb_example():
    mi.set_variant('scalar_rgb')
    # print(help(mi.traverse))
    scene = mi.load_file('scenes/living-room-2/scene.xml')

    img = mi.render(scene, spp=1024)
    bitmap = mi.Bitmap(img)
    bitmap.write('output.exr')

    # cam2 = mi.load_dict({
    #     'type': 'perspective',
    #     'to_world': mi.ScalarTransform4f.look_at(
    #         target=[-4.0, -0.0, 0.0],
    #         origin=[3.0, 0.5, 5.0],
    #         up=[0.0, 1.0, 0.0]
    #     ),
    #     'filmet': {
    #         'type': 'hdrfilm',
    #         'width': 360,
    #         'height': 640,
    #     }
    # })

    # img2 = mi.render(scene, sensor=cam2, spp=64)
    # bitmap2 = mi.Bitmap(img2)
    # bitmap2.write('output2.exr')

def llvm_ad_rgb_example():
    mi.set_variant('llvm_ad_rgb')
    scene = mi.load_file('scenes/cornell-box/scene.xml')
    # image_ref = mi.render(scene, spp=1024)
    # image_ref_data = np.array(image_ref)
    # bitmap_ref = mi.Bitmap(image_ref_data)
    # # bitmap_ref = mi.util.convert_to_bitmap(image_data)
    # bitmap_ref.write('output_ref.exr')

    scene_params = mi.traverse(scene)
    # print(scene_params) 
    param_key = 'LeftWallBSDF.brdf_0.reflectance.value'
    param_ref = mi.Color3f(scene_params[param_key])
    scene_params[param_key] = mi.Color3f(0.01, 0.0, 0.9)
    scene_params.update()

    image = mi.render(scene, spp=1024)
    image_data = np.array(image)
    bitmap = mi.Bitmap(image_data)
    bitmap.write('output_mod.exr')

def gradient_based_optimize_scene():
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_file('scenes/cornell-box/scene.xml')
    image_ref = mi.render(scene, spp=16)
    # dr.eval(image_ref)
    # image_ref = dr.detach(image_ref)

    scene_params = mi.traverse(scene)
    # print(scene_params) 
    param_key = 'LeftWallBSDF.brdf_0.reflectance.value'
    param_ref = mi.Color3f(scene_params[param_key])
    scene_params[param_key] = mi.Color3f(0.01, 0.0, 0.9)
    # scene_params[param_key] = dr.detach(scene_params[param_key])
    scene_params.update()
    dr.enable_grad(scene_params[param_key])

    opt = mi.ad.Adam(lr=0.1)
    opt[param_key] = scene_params[param_key]
    # scene_params.update()

    images = []
    for it in range(50):
        image = mi.render(scene, scene_params, spp=2)
        loss = mse(image, image_ref)

        dr.backward(loss)
        opt.step()

        opt[param_key] = dr.clip(opt[param_key], 0.0, 1.0)
    
        scene_params.update(opt)

        # dr.eval(loss, opt[param_key])

        err_ref = dr.sum(dr.square(param_ref - scene_params[param_key]))

        print(f"Iteration {it:02d}: Param Error = {err_ref[0]:6f}", end='\r')
        images.append(np.array(image))

    for i, img in enumerate(images):
        bitmap = mi.Bitmap(img)
        bitmap.write(f'opt/optimize_{i:02d}.exr')

    print('\nOptimization complete!')

def caustic_design():
    mi.set_variant('cuda_ad_rgb')
    mi.set_log_level(mi.LogLevel.Warn)

    config = {
        'render_resolution': (128, 128),
        'heightmap_resolution': (512, 512),
        'n_upsampling_steps': 4,
        'spp': 32,
        'max_iterations': 1000,
        'learning_rate': 3e-5,
        'reference': 'scenes/caustic-design/target.exr',
    }
    output_dir = realpath(join('outputs', 'sunday'))
    os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    # Tutorial 2
    scalar_rgb_example()

    # Tutorial 3
    # llvm_ad_rgb_example()
    # gradient_based_optimize_scene()

    # Tutorial 4
    # caustic_design()