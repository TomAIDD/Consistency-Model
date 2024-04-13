
import os
import argparse
from tqdm import tqdm
from cm.cm import ConsistencyModel
from cm.toy_tasks.data_generator import DataGenerator
from cm.visualization.vis_utils import plot_main_figure
from cm.utils import save_argparse, LoadFromFile

"""
Continuous training of the consistency model on a toy task.
For better performance, one can pre-training the model with the karras diffusion objective
and then use the weights as initialization for the consistency model.
"""
def get_args():
    parser = argparse.ArgumentParser(description="CT_continuous")
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep second

    parser.add_argument(
        "--device",
        default='cpu',
        type=str,
        help="Device to use for training",
    )

    parser.add_argument(
        "--n-sampling-steps",
        default=20,
        type=int,
        help="Number of sampling steps for the multi-step sampler",
    )

    parser.add_argument(
        "--task",
        default='three_gmm_1D',
        type=str,
        help="Toy task to train on",
        choices=['three_gmm_1D', 'uneven_two_gmm_1D', 'two_gmm_1D', 'single_gaussian_1D'],
    )
    
    parser.add_argument(
        "--use-pretraining",
        action='store_true',
        help="Use pretraining with the karras diffusion objective",
    )

    parser.add_argument(
        "--train-epochs", 
        default=2000, 
        type=int, 
        help="Number of training epochs"
    
    )

    parser.add_argument(
        "--log-dir", 
        default='logs',
        type=str, 
        help="Directory to save logs"
    )

    args = parser.parse_args()

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args

def main():
    args = get_args()

    device = args.device
    use_pretraining = args.use_pretraining
    n_sampling_steps = args.n_sampling_steps
    
    os.makedirs(os.path.join(args.log_dir, 'plots'), exist_ok=True)

    cm = ConsistencyModel(
        lr=1e-4,
        sampler_type='onestep',
        sigma_data=0.5,
        sigma_min=0.01,
        sigma_max=1,
        conditioned=False,
        device='cuda',
        rho=7,
        t_steps_min=200,
        t_steps=500,
        ema_rate=0.999,
        n_sampling_steps=n_sampling_steps,
        use_karras_noise_conditioning=True,    
    )
    train_epochs = args.train_epochs
    # chose one of the following toy tasks: 'three_gmm_1D' 'uneven_two_gmm_1D' 'two_gmm_1D' 'single_gaussian_1D'
    data_manager = DataGenerator(args.task) 
    samples, cond = data_manager.generate_samples(10000)
    samples = samples.reshape(-1, 1).to(device)
    pbar = tqdm(range(train_epochs))
    
    # Pretraining with karras diffusion objective if desired
    if use_pretraining:
        for i in range(train_epochs):
            cond = cond.reshape(-1, 1).to(device)        
            loss = cm.diffusion_train_step(samples, cond, i, train_epochs)
            pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
            pbar.update(1)  

        # plot the results of the pretraining diffusion model to compare with the consistency model
        plot_main_figure(
            data_manager.compute_log_prob, 
            cm, 
            1000, 
            train_epochs, 
            sampling_method='euler', 
            x_range=[-4, 4], 
            save_path = os.path.join(args.log_dir, 'plots'),
            n_sampling_steps=n_sampling_steps,
        )
        
        cm.update_target_network()  # the model contains the pretrained parameter, and this line updates the pretrained parameter to the target network
        pbar = tqdm(range(train_epochs))
    
    # Continuous training for the consistency model
    for i in range(train_epochs):
        cond = cond.reshape(-1, 1).to(device)        
        loss = cm.continuous_train_step(samples, cond, i, train_epochs)
        pbar.set_description(f"Step {i}, Loss: {loss:.8f}")
        pbar.update(1)
    
    # Plotting the results of the training
    # We do this for the one-step and the multi-step sampler to compare the results
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='onestep', 
        x_range=[-4, 4], 
        save_path = os.path.join(args.log_dir, 'plots')
    )
    plot_main_figure(
        data_manager.compute_log_prob, 
        cm, 
        100, 
        train_epochs, 
        sampling_method='multistep', 
        n_sampling_steps=n_sampling_steps,
        x_range=[-4, 4], 
        save_path = os.path.join(args.log_dir, 'plots')
    )
    print('done')


if __name__ == "__main__":
    main()