from nerf.utils import *
from main import parse_args, run_test_model, get_model, get_device, install_additional_packages, start, get_device_name

import gradio as gr
from zipfile import ZipFile
import shutil

options = parse_args()
device = get_device()

if get_device_name() != 'cpu':
    install_additional_packages()

def zip_files(workspace: str):
    dir_to_zip = os.path.join(workspace, 'results')
    dir_to_zip = dir_to_zip if os.path.exists(dir_to_zip) else workspace
    return shutil.make_archive(workspace, 'zip', dir_to_zip)
    # zip_filename = f"{workspace}.zip"
    # with ZipFile(zip_filename, "w") as zip_ref:
    #     for folder_name, subfolders, filenames in os.walk(dir_to_zip):
    #         for filename in filenames:
    #             file_path = os.path.join(folder_name, filename)
    #             zip_ref.write(file_path, arcname=os.path.relpath(file_path, dir_to_zip))
    #
    #     zip_ref.close()
    # return zip_filename

# define UI
with gr.Blocks(css=".gradio-container {max-width: 512px; margin: auto;}") as demo:

    # title
    gr.Markdown('[Stable-DreamFusion](https://github.com/kelyamany/stable-dreamfusion) Text-to-3D')

    # inputs
    text = gr.Textbox(label="Prompt", placeholder='Description of what you want to see ...', max_lines=1, value="a plywood chair")
    negative = gr.Textbox(label="N-Prompt", placeholder='Description of what you dont want to include ...', max_lines=1, value="")
    workspace = gr.Textbox(label="Workspace", max_lines=1, value="workspace")
    nerf_backbone = gr.Radio(choices=['Vanilla', 'Instant-NGP'], label="NeRF Backbone", value='Instant-NGP')
    iters = gr.Slider(label="Iters", minimum=100, maximum=20000, value=5000, step=100)
    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)

    save_mesh = gr.Checkbox(label="Save Mesh", value=True)
    # mcubes_resolution = gr.Number(label="Mesh Resolution", value=256, type=int)
    # decimate_target = gr.Number(label="Mesh Decimation", value=1e5, type=int)

    guidance = gr.Radio(choices=['stable-diffusion', 'clip'], label="Guidance", value='stable-diffusion')

    generate_button = gr.Button('Generate')
    test_button = gr.Button('Test')

    # outputs
    image = gr.Image(label="image", visible=True)
    # video = gr.Video(label="video", visible=save_mesh.value)
    mesh = gr.Model3D(label="3D Model", visible=False)
    zip_file = gr.File(label="Results", visible=True)
    logs = gr.Textbox(label="logging")

    # gradio main func
    def parse_inputs(text, negative, workspace, nerf_backbone, iters, seed, save_mesh, guidance): #, mcubes_resolution, decimate_target
        # inputs
        options.text = text
        options.negative = negative
        options.workspace = workspace
        options.iters = iters
        options.seed = seed

        options.save_mesh = save_mesh
        # options.mcubes_resolution = mcubes_resolution
        # options.decimate_target = decimate_target

        options.guidance = guidance

        if nerf_backbone == 'Instant-NGP':
            options.O = True
        else:
            options.O2 = True

        return options

    def submit(text, negative, workspace, nerf_backbone, iters, seed, save_mesh,guidance_choice): #, mcubes_resolution, decimate_target):

        opt = parse_inputs(text, negative, workspace, nerf_backbone, iters, seed, save_mesh, guidance_choice)#, mcubes_resolution, decimate_target

        start_t = time.time()
        start(opt)
        end_t = time.time()

        img_files = glob.glob(os.path.join(opt.workspace, 'results', '*.png'))
        # video_files = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
        # assert video_files is not None, "cannot retrieve videos!"
        # video_files.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime
        #
        mesh_files = glob.glob(os.path.join(opt.workspace, 'results', '*.obj'))

        yield {
            image: gr.update(value=img_files[-1] if img_files is not None else None, visible=True),
            # video: gr.update(value=video_files[-1], visible=True),
            mesh: gr.update(value=mesh_files[-1] if mesh_files is not None else None, visible=opt.save_mesh),
            zip_file: gr.update(value=zip_files(opt.workspace), visible=True),
            logs: f"Generation Finished in {(end_t - start_t)/ 60:.4f} minutes!",
        }

    def run_test_mode(workspace):
        opt = options
        opt.workspace = workspace

        model = get_model(opt)

        start_t = time.time()
        run_test_model(opt, model, device)
        end_t = time.time()

        # img_files = glob.glob(os.path.join(opt.workspace, 'results', '*.png'))
        # assert img_files is not None, "cannot retrieve videos!"
        # img_files.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime
        #
        # video_files = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
        # assert video_files is not None, "cannot retrieve videos!"
        # video_files.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime

        yield {
            # image: gr.update(value=img_files[-1], visible=True),
            # video: gr.update(value=video_files[-1], visible=True),
            zip_file: gr.update(value=zip_files(opt.workspace), visible=True),
            logs: f"Generation Finished in {(end_t - start_t) / 60:.4f} minutes!",
        }


    generate_button.click(
        submit, 
        # [text, negative, workspace, nerf_backbone, iters, seed, save_mesh, mcubes_resolution, decimate_target, guidance],
        [text, negative, workspace, nerf_backbone, iters, seed, save_mesh,
         guidance],
        [image, mesh, zip_file, logs]
    )

    test_button.click(
        run_test_mode,
        [workspace],
        [zip_file, logs]
    )

# concurrency_count: only allow ONE running progress, else GPU will OOM.
demo.queue(concurrency_count=1)

demo.launch(share=options.need_share)
