import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *
from main import parse_args, run

import gradio as gr
import gc


options = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define UI

with gr.Blocks(css=".gradio-container {max-width: 512px; margin: auto;}") as demo:

    # title
    gr.Markdown('[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) Text-to-3D')

    # inputs
    options.text = gr.Textbox(label="Prompt", max_lines=1, value="a plywood chair")
    options.negative = gr.Textbox(label="Exclude", max_lines=1, value="")
    options.workspace = gr.Textbox(label="Workspace", max_lines=1, value="workspace")
    nerf_backbone = gr.Radio(choices=['Instant-NGP', 'Vanilla'], label="NeRF Backbone", value='Instant-NGP')
    options.iters = gr.Slider(label="Iters", minimum=1000, maximum=20000, value=5000, step=100)
    options.seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)

    options.save_mesh = gr.Checkbox(label="Save Mesh", value=True)
    options.mcubes_resolution = gr.Number(label="Mesh Resolution", value=256, type=int)
    options.decimate_target = gr.Number(label="Mesh Decimation", value=1e5, type=int)

    options.guidance = gr.Radio(choices=['stable-diffusion', 'clip'], label="Guidance", value='stable-diffusion')


    if nerf_backbone == 'Instant-NGP':
        options.O = True
    else:
        options.O2 = True

    button = gr.Button('Generate')

    # outputs
    image = gr.Image(label="image", visible=True)
    video = gr.Video(label="video", visible=False)
    # Download link for a .obj file
    # mesh = gr.File(label="Mesh File", type="file")
    # material = gr.File(label="Material File", type="file")
    logs = gr.Textbox(label="logging")

    opt = options
    # gradio main func
    def submit():

        print(f'[INFO] loading options..')
        if opt.backbone == 'vanilla':
            from nerf.network import NeRFNetwork
        elif opt.backbone == 'grid':
            from nerf.network_grid import NeRFNetwork
        else:
            raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

        print(f'[INFO] loading models..')

        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device, opt.sd_version, opt.hf_key)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

        print(f'[INFO] everything loaded!')

        seed_everything(opt.seed)

        # simply reload everything...
        model = NeRFNetwork(opt)
        
        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-15)
        elif opt.optim == 'adamw':
            optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        print(f'[INFO] everything loaded!')

        # train (every ep only contain 8 steps, so we can get some vis every ~10s)
        STEPS = 8
        max_epochs = np.ceil(opt.iters / STEPS).astype(np.int32)

        # we have to get the explicit training loop out here to yield progressive results...
        loader = iter(valid_loader)

        start_t = time.time()

        for epoch in range(max_epochs):

            trainer.train_gui(train_loader, step=STEPS)
            
            # manual test and get intermediate results
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(valid_loader)
                data = next(loader)

            trainer.model.eval()

            if trainer.ema is not None:
                trainer.ema.store()
                trainer.ema.copy_to()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=trainer.fp16):
                    preds, preds_depth = trainer.test_step(data, perturb=False)

            if trainer.ema is not None:
                trainer.ema.restore()

            pred = preds[0].detach().cpu().numpy()
            # pred_depth = preds_depth[0].detach().cpu().numpy()

            pred = (pred * 255).astype(np.uint8)

            yield {
                image: gr.update(value=pred, visible=True),
                video: gr.update(visible=False),
                logs: f"training iters: {epoch * STEPS} / {opt.iters}, lr: {trainer.optimizer.param_groups[0]['lr']:.6f}",
            }
        

        # test
        trainer.test(test_loader)

        results = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
        assert results is not None, "cannot retrieve results!"
        results.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime
        
        end_t = time.time()
        
        yield {
            image: gr.update(visible=False),
            video: gr.update(value=results[-1], visible=True),
            logs: f"Generation Finished in {(end_t - start_t)/ 60:.4f} minutes!",
        }

    
    button.click(
        submit, 
        [],
        [image, video, logs]
    )

# concurrency_count: only allow ONE running progress, else GPU will OOM.
demo.queue(concurrency_count=1)

demo.launch(share=options.need_share)
