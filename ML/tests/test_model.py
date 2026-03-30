import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from src.python.neuralforge.data.datasets import get_dataset, get_num_classes, get_class_names
from src.python.neuralforge.models.resnet import ResNet18

class ModelTester:
    def __init__(self, model_path='./models/best_model.pt', dataset='cifar10', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset
        
        print("=" * 60)
        print("  NeuralForge - Interactive Model Testing")
        print("=" * 60)
        print(f"Device: {self.device}")
        
        num_classes = get_num_classes(dataset)
        self.model = self.create_model(num_classes)
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']}")
        else:
            print(f"Warning: No model found at {model_path}, using untrained model")
        
        self.model.eval()
        
        test_dataset = get_dataset(dataset, root='./data', train=False, download=True)
        self.dataset = test_dataset.dataset
        self.classes = get_class_names(dataset)
        
        if dataset in ['mnist', 'fashion_mnist']:
            self.image_size = 28
        elif dataset in ['cifar10', 'cifar100']:
            self.image_size = 32
        elif dataset == 'stl10':
            self.image_size = 96
        else:
            self.image_size = 224
        
        print(f"Dataset: {dataset} ({len(self.dataset)} test samples)")
        print(f"Classes: {len(self.classes)}")
        print("=" * 60)
    
    def create_model(self, num_classes):
        model = ResNet18(num_classes=num_classes)
        return model.to(self.device)
    
    def predict_image(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            top5_prob, top5_idx = torch.topk(probabilities, min(5, len(self.classes)), dim=1)
            
            return predicted.item(), confidence.item(), top5_idx[0].cpu().numpy(), top5_prob[0].cpu().numpy()
    
    def test_random_samples(self, num_samples=10):
        print(f"\nTesting {num_samples} random samples...")
        print("-" * 60)
        
        correct = 0
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices, 1):
            image, label = self.dataset[idx]
            pred_class, confidence, top5_idx, top5_prob = self.predict_image(image)
            
            true_label = self.classes[label]
            pred_label = self.classes[pred_class]
            
            is_correct = pred_class == label
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            print(f"{i:2d}. {status} True: {true_label:15s} | Pred: {pred_label:15s} | Conf: {confidence:.2%}")
            
            if not is_correct:
                print(f"    Top-5: ", end="")
                for j, (idx, prob) in enumerate(zip(top5_idx, top5_prob)):
                    print(f"{self.classes[idx]}({prob:.1%})", end=" ")
                print()
        
        accuracy = correct / num_samples
        print("-" * 60)
        print(f"Accuracy: {accuracy:.1%} ({correct}/{num_samples})")
    
    def test_specific_sample(self, index):
        if index < 0 or index >= len(self.dataset):
            print(f"Error: Index must be between 0 and {len(self.dataset)-1}")
            return
        
        image, label = self.dataset[index]
        pred_class, confidence, top5_idx, top5_prob = self.predict_image(image)
        
        print(f"\nSample #{index}")
        print("-" * 60)
        print(f"True Label:      {self.classes[label]}")
        print(f"Predicted:       {self.classes[pred_class]}")
        print(f"Confidence:      {confidence:.2%}")
        print(f"Status:          {'✓ Correct' if pred_class == label else '✗ Wrong'}")
        print("\nTop-5 Predictions:")
        for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
            print(f"  {i}. {self.classes[idx]:15s} {prob:.2%}")
    
    def test_class_accuracy(self):
        print("\nCalculating per-class accuracy...")
        print("-" * 60)
        
        class_correct = [0] * len(self.classes)
        class_total = [0] * len(self.classes)
        
        with torch.no_grad():
            for i, (image, label) in enumerate(self.dataset):
                pred_class, _, _, _ = self.predict_image(image)
                class_total[label] += 1
                if pred_class == label:
                    class_correct[label] += 1
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(self.dataset)} samples...", end='\r')
        
        print(" " * 60, end='\r')
        print("Per-class Accuracy:")
        
        overall_correct = sum(class_correct)
        overall_total = sum(class_total)
        
        for i, class_name in enumerate(self.classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print(f"  {class_name:15s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})")
        
        print("-" * 60)
        print(f"Overall Accuracy: {100.0 * overall_correct / overall_total:.2f}% ({overall_correct}/{overall_total})")
    
    def test_custom_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(image)
            pred_class, confidence, top5_idx, top5_prob = self.predict_image(image_tensor)
            
            print(f"\nCustom Image: {image_path}")
            print("-" * 60)
            print(f"Predicted:       {self.classes[pred_class]}")
            print(f"Confidence:      {confidence:.2%}")
            print("\nTop-5 Predictions:")
            for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
                print(f"  {i}. {self.classes[idx]:15s} {prob:.2%}")
        
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("  Interactive Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  random [N]       - Test N random samples (default: 10)")
        print("  sample <index>   - Test specific sample by index")
        print("  image <path>     - Test custom image file")
        print("  accuracy         - Calculate full test set accuracy")
        print("  help             - Show this help")
        print("  exit             - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input(">>> ").strip().lower()
                
                if not command:
                    continue
                
                if command == 'exit' or command == 'quit':
                    print("Exiting...")
                    break
                
                elif command == 'help':
                    self.interactive_mode()
                    return
                
                elif command.startswith('random'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 10
                    self.test_random_samples(n)
                
                elif command.startswith('sample'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: sample <index>")
                    else:
                        idx = int(parts[1])
                        self.test_specific_sample(idx)
                
                elif command.startswith('image'):
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: image <path>")
                    else:
                        self.test_custom_image(parts[1])
                
                elif command == 'accuracy':
                    self.test_class_accuracy()
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained NeuralForge model')
    
    default_model = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
    parser.add_argument('--model', type=str, default=default_model, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'stl10',
                               'tiny_imagenet', 'imagenet', 'food101', 'caltech256', 'oxford_pets'],
                       help='Dataset to test on')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'random', 'accuracy'],
                       help='Testing mode')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for random mode')
    args = parser.parse_args()
    
    tester = ModelTester(model_path=args.model, dataset=args.dataset, device=args.device)
    
    if args.mode == 'interactive':
        tester.interactive_mode()
    elif args.mode == 'random':
        tester.test_random_samples(args.samples)
    elif args.mode == 'accuracy':
        tester.test_class_accuracy()

if __name__ == '__main__':
    main()
