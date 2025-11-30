from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch 

class GradCamVisualizer():

    def __init__(self, model, target_layers):

        self.model = model
        self.target_layers = target_layers
        
        self.device = next(model.parameters()).device
        self.gradcam = GradCAM(model=model, target_layers=target_layers)
        
    def __call__(self, input_tensor, targets):

        # Prepare the tensor 
        prepared_tensor = self._prepare_for_grad_cam(input_tensor)

        # Generate Grad Cam
        grad_cam_mask = self._generate_grad_cam(prepared_tensor, targets)

        # Prepare Tensor for imshow
        img = self._prepare_for_imshow(input_tensor)

        # Overlay Grad Cam on Image 
        return self._overlay_cam_on_image(img, grad_cam_mask)
    
    def get_prediction(self, input_tensor):   # Input tensor must be of shape [Batch, C, H, W]

        self.model.eval()
        with torch.inference_mode():

            pred_logits = self.model(input_tensor)
            prediction = torch.argmax(pred_logits, dim=1)

        return prediction

    def _prepare_for_grad_cam(self,input_tensor):
        return input_tensor.unsqueeze(0)    # Adds the Batch Dimension to a tensor of shape [C,H,W]

    def _prepare_for_imshow(self, input_tensor):
        img = input_tensor.permute(1,2,0).cpu().numpy()   # imshow requires shape [H,W,C]
                                                            
        img = (img - img.min()) / (img.max() - img.min()) # Now we Normalize the image to have values between [0,1]

        return img
    
    def _generate_grad_cam(self, input_tensor, targets):

        if isinstance(targets, list):
            targets = [ClassifierOutputTarget(target) for target in targets]
        else:
            targets = [ClassifierOutputTarget(targets)]

        grayscale_cam = self.gradcam(input_tensor= input_tensor, targets=targets)
        return grayscale_cam[0,:]
    
    def _overlay_cam_on_image(self, img, mask):
        return show_cam_on_image(img, mask, use_rgb=True)
    