import torch
import torch.nn as nn
import torch.nn.functional as f
import wandb
import torchsummary
import logging
from tqdm.autonotebook import tqdm
import time
from datetime import date
import os
from utils.logging_handlers import *
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, optimizer, loss, epoch, ckpt_name, save_dir):
    """
        Saving torch checkpoint
    """
    checkpoint_path = os.path.join(save_dir, ckpt_name)
    checkpoint_dict = {}
    checkpoint_dict["epoch"] = epoch
    checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint_dict["model_state_dict"] = model.state_dict()
    checkpoint_dict["loss"] = loss
    
    torch.save(checkpoint_dict, checkpoint_path)
    
    
def load_checkpoint(model, optimizer, checkpoint_path):
    """
        Load torch checkpoint
    """
    checkpoint_dict = torch.load(checkpoint_path)
    epoch = checkpoint_dict["epoch"]
    
    optimizer_state_dict = checkpoint_dict["optimizer_state_dict"]
    optimizer.load_state_dict(optimizer_state_dict)
    
    model_state_dict = checkpoint_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    loss = checkpoint_dict["loss"]
    return epoch, loss


def calculate_final_metric(metrics, weights):
    """
        Calculate the final metrics from the metrics results and the assigned weights:
        - Params:
            metrics: dictionary of all metric name and their values
            weights: the assigned weights to all the metrics
        
        - Returns:
            final_metric:
    """
    assert len(metrics) == len(weights), "[ERROR]: The number of assigned weights is different from the number of metrics"
    
    final_metric = 0.0
    for i, metric in enumerate(metrics.keys()):
        final_metric += weights[i] * metrics[metric]
        
    return final_metric
    

def train_one_epoch(model, dataloader, optimizer, loss_func, metrics, device):
    """
        Train the model for one epoch:
        - Params:
            model: (nn.Module) The deep learning model
            dataloader:     Customized dataloader
            optimizer:      Customized optimizer
            loss_func:      Customized loss function
            metrics: (list) List of applied metrics (Follow TorchVision format)
        - Returns:
    """
    model.train()
    
    # Main training loop
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader)):
        image, target = data
        image, target = image.to(device), target.to(device)
        
        # Feed the input through the model
        output = model(image)
        loss = loss_func(output, target)
        running_loss += loss.item()
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        for metric in metrics:
            metric.update(output, target)
            
    # Calculate the final metrices
    metrics_res = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        metrics_res[metric_name] = metric.compute()
        metric.reset()

    return (running_loss/len(dataloader)), metrics_res


def validate(model, dataloader, loss_func, metrics, device):
    """
        Validate the model
        - Params:
            model:
            dataloader:
            loss_func:
            metrics:
            device:
        - Returns:
    """
    model.eval()
    running_loss = 0.0
    
    # Loop through the batches
    for i, data in tqdm(enumerate(dataloader)):
        image, target = data
        image, target = image.to(device), target.to(device)
        
        # Feed the input through the model
        output = model(image)
        loss = loss_func(output, target)
        running_loss += loss.item()
        
        # Calculate metrics
        for metric in metrics:
            metric.update(output, target)
            
    # Calculate the final metrices
    metrics_res = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        metrics_res[metric_name] = metric.compute()
        metric.reset()
    
    return (running_loss/len(dataloader)), metrics_res


def train_epochs(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        loss_func, 
        metrics,
        metric_weights, 
        device, 
        num_epochs, 
        log_rate=None, 
        save_rate=None,
        save_dir=None, 
        logging_level=logging.DEBUG, 
        use_wandb=False, 
        use_tensorboard=False, 
        resume_training=False, 
        checkpoint_path=None,
        interval=0,
        lr_scheduler=None
    ):
    """
        Train the model for multiple epochs
        - Params:
            model:              DeepLearning Model
            train_loader:       Training dataloader
            val_loader:         Validation dataloader
            optimizer:          Optimizer
            loss_func:          Loss function
            metrics:            List of applied metrics
            metric_weights:     Weights assigned to all metrics, needed to specify the best checkpoint   
            device:             Device, cpu or cuda
            num_epochs:         Total number of epochs used for training the model
            log_rate:           Log validating information after several epochs
            save_rate:          Saving checkpoints after several epochs
            logging_level:      Local logging level [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
            use_wandb:          Whether to use wandb (Weights & Biases) for logging, suitable for training on servers. Required account
            use_tensorboard:    Whether to use tensorboard for logging, suitable for local training
            resume_training:    Resume from saved checkpoint path
            checkpoint_path:    Checkpoint_path, required if resume training from previous session
            interval:           Time rest between each epochs (seconds)
            lr_scheduler:       Learning rate scheduler
    """
    
    # Create logger
    logger = logging.getLogger()
    
    # Set logger level
    logger.setLevel(logging_level)
    logger.addHandler(TqdmLoggingHandler())
    
    # Check requirements:
    if save_rate is not None:
        assert save_dir is not None, "[ERROR]: Please specify the checkpoint directory to use save mode! "
        
    # Resume training from previous checkpoint
    if resume_training:
        assert checkpoint_path is not None, "[ERROR]: Please specity the checkpoint path to resume training from checkpoint!"
        assert os.path.exists(checkpoint_path), "[ERROR]: Checkpoint path does not exists!"
        logger.debug("[INFO]: Resume training from checkpoint")
        epoch, train_loss = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        epoch = 0
    
    # Create tensorboard writer if use_tensorboard
    if use_tensorboard:
        writer = SummaryWriter()      
    
    # Initialize best score, load model to device
    best_score = 0.0
    model.float()
    model.to(device)
    
    # Display model info
    img, label = next(iter(train_loader))
    print(img.shape)
    img_shape = img.shape
    torchsummary.summary(model, img_shape[1:], device=device) # Remove batchsize channel
    
    
    # Main training loop
    while epoch < num_epochs:
        epoch += 1
        logger.debug("[INFO]: Epoch: {}/{}".format(epoch, num_epochs))
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_func, metrics, device)
        
        # Logging using logger
        logger.debug("[INFO]: Training Results:")
        logger.debug("Training loss: {}".format(train_loss))
        for metric in train_metrics.keys():
            logger.debug("Train_{}: {}".format(metric, train_metrics[metric]))
        
        # Logging using tensorboard to log training results
        if use_tensorboard:
            writer.add_scalar("Train Loss", train_loss, epoch)
            for metric in train_metrics.keys():
                writer.add_scalar("Train {}".format(metric), train_metrics[metric], epoch)
            writer.flush()
        
        # Logging using wandb to log training results
        if use_wandb:
            wandb.log(
                {"Train Loss": train_loss})
            
            for metric in train_metrics.keys():
                wandb.log(
                    {"Train {}".format(metric): train_metrics[metric]}
                )
                
        logger.info("[INFO]: Validating ...")
        val_loss, val_metrics = validate(model, val_loader, loss_func, metrics, device)
        
        # Save the best checkpoint
        final_score = calculate_final_metric(val_metrics, metric_weights)
        if final_score > best_score:
            best_score = final_score
            save_checkpoint(model, optimizer, val_loss, epoch, "best.ckpt", save_dir)
        
        # Logging using tensorboard to log training results
        if use_tensorboard:
            writer.add_scalar("Val Loss", val_loss, epoch)
            for metric in train_metrics.keys():
                writer.add_scalar("Val {}".format(metric), val_metrics[metric], epoch)
            writer.flush()
        
        # Logging using wandb
        if use_wandb:
            wandb.log(
                {"Val Loss": val_loss})
            
            for metric in train_metrics.keys():
                wandb.log(
                    {"Val {}".format(metric): val_metrics[metric]}
                )
        
        if (log_rate != None) and (epoch % log_rate == 0):
            logger.debug("[INFO]: Validation Results:")
            logger.debug("Validation loss: {}".format(val_loss))
            for metric in val_metrics.keys():
                logger.debug("Val_{}: {}".format(metric, val_metrics[metric]))
                
        if (save_rate != None) and (epoch % save_rate == 0):
            checkpoint_name = "Epoch_{}.ckpt".format(epoch)
            save_checkpoint(model, optimizer, train_loss, epoch, checkpoint_name, save_dir)

        if interval != 0:
            logger.debug("[INFO]: Sleeping for {} secs ...".format(interval))
            time.sleep(interval)
            
    # Summarizing result:
    best_epoch, best_loss = load_checkpoint(model, optimizer, os.path.join(save_dir, "best.ckpt"))
    logger.debug("[INFO]: ------------------- Training completed! -------------------------")
    logger.debug("[INFO]: Best checkpoint: epoch {}".format(best_epoch))
    best_loss, best_metrics = validate(model, val_loader, loss_func, metrics, device)
    logger.debug("Best loss: {}".format(best_loss))
    for metric in val_metrics.keys():
        logger.debug("Best_{}: {}".format(metric, best_metrics[metric]))
    
            
    # Free resources
    if use_tensorboard:
        writer.close()
    if use_wandb:
        wandb.finish()