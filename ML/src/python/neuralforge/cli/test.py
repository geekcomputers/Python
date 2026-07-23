import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from neuralforge.data.datasets import get_dataset, get_num_classes
from neuralforge.models.resnet import ResNet18

def main():
    parser = argparse.ArgumentParser(
        description='NeuralForge - Test trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralforge-test --model models/best_model.pt --dataset cifar10 --mode random
  neuralforge-test --dataset mnist --mode accuracy
  neuralforge-test --dataset stl10 --image cat.jpg
        """
    )
    
    default_model = './models/best_model.pt'
    parser.add_argument('--model', type=str, default=default_model, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'accuracy', 'interactive'])
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for random mode')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NeuralForge - Model Testing")
    print("=" * 60)
    print(f"Device: {args.device}")
    
    dataset_aliases = {
        'cifar-10': 'cifar10', 'stl-10': 'stl10', 'fashion-mnist': 'fashion_mnist',
        'tiny-imagenet': 'tiny_imagenet', 'food-101': 'food101',
    }
    dataset_name = dataset_aliases.get(args.dataset.lower(), args.dataset.lower())
    
    num_classes = get_num_classes(dataset_name)
    model = ResNet18(num_classes=num_classes)
    model = model.to(args.device)
    
    if os.path.exists(args.model):
        print(f"Loading model from: {args.model}")
        checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        print(f"Warning: No model found at {args.model}")
        return
    
    model.eval()
    
    test_dataset = get_dataset(dataset_name, root='./data', train=False, download=True)
    classes = getattr(test_dataset, 'classes', [str(i) for i in range(num_classes)])
    
    print(f"Dataset: {dataset_name} ({len(test_dataset.dataset)} test samples)")
    print("=" * 60)
    
    if args.image:
        image = Image.open(args.image).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, min(5, num_classes), dim=1)
        
        print(f"\nPrediction for {args.image}:")
        print(f"Main: {classes[top5_idx[0][0].item()]} ({top5_prob[0][0].item()*100:.2f}%)")
        print("\nTop-5:")
        for i, (idx, prob) in enumerate(zip(top5_idx[0], top5_prob[0]), 1):
            print(f"  {i}. {classes[idx.item()]:15s} {prob.item()*100:.2f}%")
    
    elif args.mode == 'random':
        print(f"\nTesting {args.samples} random samples...")
        print("-" * 60)
        
        correct = 0
        indices = np.random.choice(len(test_dataset.dataset), args.samples, replace=False)
        
        for i, idx in enumerate(indices, 1):
            image, label = test_dataset.dataset[idx]
            
            with torch.no_grad():
                image = image.unsqueeze(0).to(args.device)
                outputs = model(image)
                pred_class = outputs.argmax(1).item()
                confidence = F.softmax(outputs, dim=1)[0][pred_class].item() * 100
            
            is_correct = pred_class == label
            correct += is_correct
            status = "✓" if is_correct else "✗"
            
            print(f"{i:2d}. {status} True: {classes[label]:15s} | Pred: {classes[pred_class]:15s} | Conf: {confidence:.1f}%")
        
        print("-" * 60)
        print(f"Accuracy: {correct/args.samples:.1%} ({correct}/{args.samples})")
    
    elif args.mode == 'accuracy':
        print("\nCalculating full test accuracy...")
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image, label in test_dataset.dataset:
                image = image.unsqueeze(0).to(args.device)
                outputs = model(image)
                pred_class = outputs.argmax(1).item()
                total += 1
                if pred_class == label:
                    correct += 1
                
                if total % 100 == 0:
                    print(f"Processed {total}/{len(test_dataset.dataset)}...", end='\r')
        
        print(f"\nOverall Accuracy: {100.0 * correct / total:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
