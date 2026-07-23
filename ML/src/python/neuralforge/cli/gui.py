import sys
import os

def main():
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("Error: PyQt6 not installed")
        print("Install with: pip install neuralforge[gui]")
        print("Or: pip install PyQt6")
        sys.exit(1)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    sys.path.insert(0, root_dir)
    
    from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                 QPushButton, QLabel, QLineEdit, QFileDialog, 
                                 QProgressBar, QTextEdit, QGroupBox)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QFont
    
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    
    from neuralforge.data.datasets import get_dataset, get_num_classes
    from neuralforge.models.resnet import ResNet18
    
    class PredictionThread(QThread):
        finished = pyqtSignal(list, list, str)
        error = pyqtSignal(str)
        
        def __init__(self, model, image_path, classes, device):
            super().__init__()
            self.model = model
            self.image_path = image_path
            self.classes = classes
            self.device = device
        
        def run(self):
            try:
                image = Image.open(self.image_path).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    top5_prob, top5_idx = torch.topk(probabilities, min(5, len(self.classes)), dim=1)
                    
                    predictions = []
                    confidences = []
                    
                    for idx, prob in zip(top5_idx[0].cpu().numpy(), top5_prob[0].cpu().numpy()):
                        predictions.append(self.classes[idx])
                        confidences.append(float(prob) * 100)
                    
                    main_prediction = predictions[0]
                    
                    self.finished.emit(predictions, confidences, main_prediction)
            
            except Exception as e:
                self.error.emit(str(e))
    
    class NeuralForgeGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.model = None
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.classes = []
            self.dataset_name = 'cifar10'
            
            self.init_ui()
            self.apply_stylesheet()
        
        def init_ui(self):
            self.setWindowTitle('NeuralForge - Model Tester')
            self.setGeometry(100, 100, 1200, 800)
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QHBoxLayout()
            central_widget.setLayout(main_layout)
            
            left_panel = self.create_left_panel()
            right_panel = self.create_right_panel()
            
            main_layout.addWidget(left_panel, 1)
            main_layout.addWidget(right_panel, 1)
        
        def create_left_panel(self):
            panel = QWidget()
            layout = QVBoxLayout()
            panel.setLayout(layout)
            
            title = QLabel('🚀 NeuralForge Model Tester')
            title.setFont(QFont('Arial', 20, QFont.Weight.Bold))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title)
            
            model_group = QGroupBox('Model Selection')
            model_layout = QVBoxLayout()
            
            model_path_layout = QHBoxLayout()
            self.model_path_input = QLineEdit()
            self.model_path_input.setPlaceholderText('Path to model file (.pt)')
            model_path_layout.addWidget(self.model_path_input)
            
            browse_btn = QPushButton('Browse')
            browse_btn.clicked.connect(self.browse_model)
            model_path_layout.addWidget(browse_btn)
            
            default_btn = QPushButton('Use Default')
            default_btn.clicked.connect(self.use_default_model)
            model_path_layout.addWidget(default_btn)
            
            model_layout.addLayout(model_path_layout)
            
            dataset_layout = QHBoxLayout()
            dataset_label = QLabel('Dataset:')
            self.dataset_input = QLineEdit('cifar10')
            self.dataset_input.setPlaceholderText('cifar10, mnist, stl10, tiny_imagenet, etc.')
            self.dataset_input.setToolTip('Supported: cifar10, cifar100, mnist, fashion_mnist, stl10,\ntiny_imagenet, imagenet, food101, caltech256, oxford_pets')
            dataset_layout.addWidget(dataset_label)
            dataset_layout.addWidget(self.dataset_input)
            model_layout.addLayout(dataset_layout)
            
            self.load_model_btn = QPushButton('Load Model')
            self.load_model_btn.clicked.connect(self.load_model)
            model_layout.addWidget(self.load_model_btn)
            
            self.model_status = QLabel('No model loaded')
            self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            model_layout.addWidget(self.model_status)
            
            model_group.setLayout(model_layout)
            layout.addWidget(model_group)
            
            image_group = QGroupBox('Image Selection')
            image_layout = QVBoxLayout()
            
            image_path_layout = QHBoxLayout()
            self.image_path_input = QLineEdit()
            self.image_path_input.setPlaceholderText('Path to image file')
            image_path_layout.addWidget(self.image_path_input)
            
            browse_image_btn = QPushButton('Browse')
            browse_image_btn.clicked.connect(self.browse_image)
            image_path_layout.addWidget(browse_image_btn)
            
            image_layout.addLayout(image_path_layout)
            
            self.image_preview = QLabel()
            self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_preview.setMinimumHeight(300)
            self.image_preview.setStyleSheet('border: 2px dashed #666; border-radius: 10px;')
            self.image_preview.setText('No image selected')
            image_layout.addWidget(self.image_preview)
            
            self.predict_btn = QPushButton('🔍 Predict')
            self.predict_btn.clicked.connect(self.predict_image)
            self.predict_btn.setEnabled(False)
            image_layout.addWidget(self.predict_btn)
            
            image_group.setLayout(image_layout)
            layout.addWidget(image_group)
            
            layout.addStretch()
            
            return panel
        
        def create_right_panel(self):
            panel = QWidget()
            layout = QVBoxLayout()
            panel.setLayout(layout)
            
            results_group = QGroupBox('Prediction Results')
            results_layout = QVBoxLayout()
            
            self.main_prediction = QLabel('No prediction yet')
            self.main_prediction.setFont(QFont('Arial', 24, QFont.Weight.Bold))
            self.main_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_prediction.setStyleSheet('color: #4CAF50; padding: 20px;')
            results_layout.addWidget(self.main_prediction)
            
            self.confidence_label = QLabel('')
            self.confidence_label.setFont(QFont('Arial', 16))
            self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            results_layout.addWidget(self.confidence_label)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            results_layout.addWidget(self.progress_bar)
            
            results_group.setLayout(results_layout)
            layout.addWidget(results_group)
            
            top5_group = QGroupBox('Top-5 Predictions')
            top5_layout = QVBoxLayout()
            
            self.top5_display = QTextEdit()
            self.top5_display.setReadOnly(True)
            self.top5_display.setMinimumHeight(200)
            top5_layout.addWidget(self.top5_display)
            
            top5_group.setLayout(top5_layout)
            layout.addWidget(top5_group)
            
            info_group = QGroupBox('Model Information')
            info_layout = QVBoxLayout()
            
            self.model_info = QTextEdit()
            self.model_info.setReadOnly(True)
            self.model_info.setMaximumHeight(150)
            info_layout.addWidget(self.model_info)
            
            info_group.setLayout(info_layout)
            layout.addWidget(info_group)
            
            layout.addStretch()
            
            return panel
        
        def apply_stylesheet(self):
            qss = """
            QMainWindow {
                background-color: #1e1e1e;
            }
            
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial;
                font-size: 12px;
            }
            
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #4CAF50;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
            }
            
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            
            QLineEdit {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px;
                color: #e0e0e0;
            }
            
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            
            QTextEdit {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 10px;
                color: #e0e0e0;
            }
            
            QLabel {
                color: #e0e0e0;
            }
            
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
            }
            
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            """
            self.setStyleSheet(qss)
        
        def browse_model(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                'Select Model File', 
                './models',
                'Model Files (*.pt *.pth);;All Files (*.*)'
            )
            if file_path:
                self.model_path_input.setText(file_path)
        
        def use_default_model(self):
            default_path = './models/final_model.pt'
            if not os.path.exists(default_path):
                default_path = './models/best_model.pt'
            self.model_path_input.setText(os.path.abspath(default_path))
        
        def browse_image(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                'Select Image File',
                '',
                'Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*.*)'
            )
            if file_path:
                self.image_path_input.setText(file_path)
                self.display_image(file_path)
        
        def display_image(self, image_path):
            try:
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                self.image_preview.setPixmap(scaled_pixmap)
            except Exception as e:
                self.image_preview.setText(f'Error loading image: {e}')
        
        def load_model(self):
            model_path = self.model_path_input.text()
            dataset_input = self.dataset_input.text().lower().strip()
            
            dataset_aliases = {
                'cifar10': 'cifar10', 'cifar-10': 'cifar10', 'cifar_10': 'cifar10',
                'cifar100': 'cifar100', 'cifar-100': 'cifar100', 'cifar_100': 'cifar100',
                'mnist': 'mnist',
                'fashionmnist': 'fashion_mnist', 'fashion-mnist': 'fashion_mnist', 'fashion_mnist': 'fashion_mnist',
                'stl10': 'stl10', 'stl-10': 'stl10', 'stl_10': 'stl10',
                'tinyimagenet': 'tiny_imagenet', 'tiny-imagenet': 'tiny_imagenet', 'tiny_imagenet': 'tiny_imagenet',
                'imagenet': 'imagenet',
                'food101': 'food101', 'food-101': 'food101', 'food_101': 'food101',
                'caltech256': 'caltech256', 'caltech-256': 'caltech256', 'caltech_256': 'caltech256',
                'oxfordpets': 'oxford_pets', 'oxford-pets': 'oxford_pets', 'oxford_pets': 'oxford_pets',
            }
            
            self.dataset_name = dataset_aliases.get(dataset_input, dataset_input)
            
            if not model_path:
                self.model_status.setText('Please select a model file')
                self.model_status.setStyleSheet('color: #f44336;')
                return
            
            if not os.path.exists(model_path):
                self.model_status.setText('Model file not found')
                self.model_status.setStyleSheet('color: #f44336;')
                return
            
            try:
                self.model_status.setText('Loading model...')
                self.model_status.setStyleSheet('color: #FFC107;')
                QApplication.processEvents()
                
                num_classes = get_num_classes(self.dataset_name)
                self.model = ResNet18(num_classes=num_classes)
                self.model = self.model.to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                try:
                    dataset = get_dataset(self.dataset_name, train=False, download=False)
                    self.classes = getattr(dataset, 'classes', [str(i) for i in range(num_classes)])
                except:
                    from neuralforge.data.datasets import get_class_names
                    self.classes = get_class_names(self.dataset_name)
                
                self.model_status.setText(f'✓ Model loaded successfully')
                self.model_status.setStyleSheet('color: #4CAF50;')
                
                self.predict_btn.setEnabled(True)
                
                total_params = sum(p.numel() for p in self.model.parameters())
                epoch = checkpoint.get('epoch', 'Unknown')
                val_loss = checkpoint.get('best_val_loss', 'Unknown')
                
                val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
                
                info_text = f"""
Model: ResNet18
Dataset: {self.dataset_name.upper()}
Classes: {num_classes}
Parameters: {total_params:,}
Epoch: {epoch}
Best Val Loss: {val_loss_str}
Device: {self.device.upper()}
                """
                self.model_info.setText(info_text.strip())
                
            except Exception as e:
                self.model_status.setText(f'Error: {str(e)}')
                self.model_status.setStyleSheet('color: #f44336;')
        
        def predict_image(self):
            image_path = self.image_path_input.text()
            
            if not image_path or not os.path.exists(image_path):
                self.main_prediction.setText('Please select a valid image')
                self.main_prediction.setStyleSheet('color: #f44336;')
                return
            
            if self.model is None:
                self.main_prediction.setText('Please load a model first')
                self.main_prediction.setStyleSheet('color: #f44336;')
                return
            
            self.predict_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            self.prediction_thread = PredictionThread(self.model, image_path, self.classes, self.device)
            self.prediction_thread.finished.connect(self.display_results)
            self.prediction_thread.error.connect(self.display_error)
            self.prediction_thread.start()
        
        def display_results(self, predictions, confidences, main_prediction):
            self.progress_bar.setVisible(False)
            self.predict_btn.setEnabled(True)
            
            self.main_prediction.setText(f'🎯 {main_prediction}')
            self.main_prediction.setStyleSheet('color: #4CAF50; padding: 20px; font-size: 28px;')
            
            self.confidence_label.setText(f'Confidence: {confidences[0]:.2f}%')
            
            top5_text = '<h3>Top-5 Predictions:</h3><hr>'
            for i, (pred, conf) in enumerate(zip(predictions, confidences), 1):
                bar_width = int(conf * 3)
                bar = '█' * bar_width
                top5_text += f'<p style="margin: 10px 0;"><b>{i}. {pred}</b><br>'
                top5_text += f'<span style="color: #4CAF50;">{bar}</span> {conf:.2f}%</p>'
            
            self.top5_display.setHtml(top5_text)
        
        def display_error(self, error_msg):
            self.progress_bar.setVisible(False)
            self.predict_btn.setEnabled(True)
            
            self.main_prediction.setText(f'Error: {error_msg}')
            self.main_prediction.setStyleSheet('color: #f44336;')
    
    app = QApplication(sys.argv)
    window = NeuralForgeGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
