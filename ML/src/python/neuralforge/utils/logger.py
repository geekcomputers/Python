import os
import sys
import logging
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, log_dir: str, name: str = "neuralforge"):
        self.log_dir = log_dir
        self.name = name
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info(f"Logger initialized. Logging to: {log_file}")
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        if step is not None:
            message = f"Step {step}: "
        else:
            message = "Metrics: "
        
        metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                       for k, v in metrics.items()]
        message += ", ".join(metric_strs)
        
        self.info(message)
    
    def log_model_summary(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("=" * 50)
        self.info("Model Summary")
        self.info("=" * 50)
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        self.info("=" * 50)
    
    def separator(self, char: str = "=", length: int = 80):
        self.info(char * length)

class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Skipping TensorBoard logging.")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_graph(self, model, input_to_model):
        if self.enabled:
            self.writer.add_graph(model, input_to_model)
    
    def close(self):
        if self.enabled:
            self.writer.close()
