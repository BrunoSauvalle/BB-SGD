
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from argparse import ArgumentParser
from tqdm import tqdm

import datasets

# recommended autograd options for speed optimization
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--input_path',
                        default=os.getcwd(),
                        help="Path to the input frames sequence directory")
    parser.add_argument('--output_dir_path',
                        default=os.getcwd(),
                        help="Path of the directory where to save the reconstructed background")
    parser.add_argument('--OF_algorithm',
                        default='DIS_FAST',
                        choices=['DIS_FAST','DIS_ULTRAFAST'],
                        help="Algorithm to compute optical flow.")
    parser.add_argument('--tau_1', type=float,default = 0.25, help="hyperparameter tau_1")
    parser.add_argument('--tau_2', type=float, default=255.0/40000, help="hyperparameter tau_2")
    parser.add_argument('--tau_3', type=float, default=240.0 /255, help="hyperparameter tau_3")
    parser.add_argument('--r', type=int, default=75 , help="hyperparameter r")
    parser.add_argument('--gamma', type=float, default=3, help="hyperparameter gamma for global weight")
    parser.add_argument('--beta', type=float, default=6, help="hyperparameter beta : bootstrap coefficient")
    parser.add_argument('--phi', type=float, default=2, help="hyperparameter phi for optical flow weights")
    parser.add_argument('--keep_motionless_frames', dest='motionless_frames_suppression', action='store_false')
    parser.set_defaults(motionless_frames_suppression=True)
    parser.add_argument('--n_iterations', type=int, default=3000, help="number of optimization iterations")

    return parser


def background_loss(input_images, optical_flows,reconstructed_backgrounds,args, short_video):
    """ background loss used during training
        input images format is tensor shape N,C,H,W RGB+optical flow range 0-255
         """

    bs, nc, h, w = input_images.size() # batch size, number of channels, image height, image width

    pixel_losses = torch.sum(torch.nn.functional.smooth_l1_loss(input_images,
                                                reconstructed_backgrounds,  reduction='none', beta=3.0), dim=1)

    with torch.no_grad(): # computation of various weights
        soft_masks = torch.tanh(pixel_losses*(1/(255*args.tau_1))) # range 0-1
        bootstrap_weight_logit = -args.beta*torch.nn.functional.avg_pool2d(soft_masks, 2 * (w // args.r) + 1,
                                                         stride=1, padding= w // args.r, count_include_pad=False) # range 0-1 BSxHxW
        if short_video:
            pixel_weights = torch.exp(bootstrap_weight_logit)
        else:
            global_error_weight_logit = -args.gamma*((torch.sum(pixel_losses, dim=(1,2))/(255*h * w)).reshape(bs, 1, 1).expand(bs, h, w))
            optical_flow_weight_logit = -(args.phi / 255.0) * optical_flows
            pixel_weights = torch.exp(bootstrap_weight_logit +  optical_flow_weight_logit + global_error_weight_logit )

        normalized_pixel_weights = (pixel_weights*(1/(255*bs * h * w))).detach()

    loss = torch.sum(pixel_losses * normalized_pixel_weights)

    return loss


def background_training_loop(args,background_model, background_optimizer, scheduler,
                        dataset,batch_size,  device, number_of_epochs):
    """ training loop for static background model"""

    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  num_workers=0,
                                                  drop_last=False, pin_memory=False,
                                                  shuffle=True, persistent_workers=False)

    if device == torch.device('cpu'):
        float_tensor_type = torch.FloatTensor
    else:
        float_tensor_type = torch.cuda.FloatTensor

    if len(dataset) > 10:
        short_video = False
    else:
        short_video = True


    for epoch in tqdm(range(number_of_epochs)):

        for j, data in enumerate(traindataloader, 0):

            input_images = data[0].type(float_tensor_type)
            optical_flows = data[1].type(float_tensor_type)
            current_batch_size = input_images.size()[0]

            background_optimizer.zero_grad(set_to_none=True)
            reconstructed_backgrounds = (255*background_model).expand(current_batch_size,-1,-1,-1)

            loss = background_loss(input_images,optical_flows, reconstructed_backgrounds,args,
                                            short_video)
            loss.backward()
            background_optimizer.step()

        scheduler.step()

    return (background_model[0]).detach().to('cpu') # shape NCHW range 0-1

def compute_static_background_from_sequence(input_path=None,args=None):
    """returns reconstructed background as a torch tensor, CHW shape, range 0-1"""

    if args == None:
        parser = create_parser()
        args = parser.parse_args("") # loads default arguments if no args is provided
    if input_path != None:
        args.input_path = input_path # overrides default input path if input path is provided

    assert os.path.exists(args.input_path)

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else :
        print('warning : no GPU detected, model will be slow')
        device = torch.device("cpu")

    # loads dataset and computes optical flows
    dataset = datasets.Image_dataset_with_optical_flow(args,device= device)

    torch.backends.cudnn.benchmark = True

    # model initialization

    background_model = torch.rand((1, dataset.nc, dataset.image_height, dataset.image_width), device = device,requires_grad=True)

    background_optimizer = optim.Adam([background_model], lr=3e-2,
                                      betas=(0.90, 0.999), weight_decay=0)
    batch_size = 64
    n_iterations=args.n_iterations
    n_iterations_per_epoch = 1+len(dataset)//batch_size
    number_of_epochs = 2+n_iterations//n_iterations_per_epoch

    scheduler = torch.optim.lr_scheduler.StepLR(background_optimizer, step_size=max(2,(3 * number_of_epochs) // 4), gamma=0.1)

    print(f'Optimizing loss function... ')
    reconstructed_background = background_training_loop(args,background_model, background_optimizer, scheduler,
                        dataset,batch_size,  device, number_of_epochs)

    return reconstructed_background # NCHW format range 0-1

if __name__ == "__main__":
        parser = create_parser()
        args = parser.parse_args()
        reconstructed_background = compute_static_background_from_sequence(args=args)
        output_path = '%s/reconstructed_background.png' % args.output_dir_path
        vutils.save_image(reconstructed_background,
                          output_path)
        print(f'background reconstruction saved as {output_path} ')





