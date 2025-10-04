import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class SceneGraphTransforms:
    """Data transforms for scene graph datasets with images and annotations."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Basic transforms
        self.transforms = []
        
        # Resize
        self.transforms.append(T.Resize(image_size))
        
        # Data augmentation (if enabled)
        if augment:
            self.transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=10)
            ])
        
        # Convert to tensor
        self.transforms.append(T.ToTensor())
        
        # Normalization (ImageNet stats)
        if normalize:
            self.transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            )
        
        self.transform = T.Compose(self.transforms)
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply transforms to a sample containing image and scene graph data.
        
        Args:
            sample: Dict with keys 'image', 'objects', 'relationships', etc.
        
        Returns:
            Transformed sample
        """
        # Transform image
        if 'image' in sample:
            sample['image'] = self.transform(sample['image'])
        
        # Handle bounding boxes if present
        if 'bboxes' in sample:
            sample['bboxes'] = self._transform_bboxes(sample['bboxes'])
        
        return sample
    
    def _transform_bboxes(self, bboxes: List) -> torch.Tensor:
        """Transform bounding boxes to match image transforms."""
        # Convert to tensor and normalize coordinates
        if isinstance(bboxes, list):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        
        # Normalize to [0, 1] if not already
        # This assumes bboxes are in format [x1, y1, x2, y2]
        return bboxes

class RelationshipTransforms:
    """Transforms specific to relationship data in scene graphs."""
    
    def __init__(self, max_objects: int = 50, max_relationships: int = 100):
        self.max_objects = max_objects
        self.max_relationships = max_relationships
    
    def pad_sequences(self, sample: Dict) -> Dict:
        """Pad object and relationship sequences to fixed length."""
        if 'objects' in sample:
            objects = sample['objects']
            if len(objects) > self.max_objects:
                sample['objects'] = objects[:self.max_objects]
            else:
                # Pad with zeros or special tokens
                padding = [0] * (self.max_objects - len(objects))
                sample['objects'] = objects + padding
        
        if 'relationships' in sample:
            relationships = sample['relationships']
            if len(relationships) > self.max_relationships:
                sample['relationships'] = relationships[:self.max_relationships]
            else:
                padding = [0] * (self.max_relationships - len(relationships))
                sample['relationships'] = relationships + padding
        
        return sample

def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> SceneGraphTransforms:
    """Get transforms for training data."""
    return SceneGraphTransforms(
        image_size=image_size,
        normalize=True,
        augment=True
    )

def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> SceneGraphTransforms:
    """Get transforms for validation/test data."""
    return SceneGraphTransforms(
        image_size=image_size,
        normalize=True,
        augment=False
    )