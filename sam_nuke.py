import os
import torch
import segment_anything
from segment_anything import sam_model_registry
import logging
from torch import nn
import segment_anything.modeling

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Define the paths and model type
BASE_PATH = "./nuke/Cattery/SegmentAnything"
SAM_MODELS = {
    "vit_b": "../models/sam_vit_b_01ec64.pth",
    # "vit_l": "../models/sam_vit_l_0b3195.pth",
    "vit_h": "../models/sam_vit_h_4b8939.pth",
}


class SamEncoderNuke(nn.Module):
    """
    A wrapper around the SAM model that allows it to be used as a TorchScript model.
    """

    def __init__(self, sam_model: segment_anything.modeling.Sam) -> torch.Tensor:
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x: torch.Tensor):
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
        dtype = x.dtype

        if dtype == torch.float32:
            x = x.half()

        image = x.to(device)
        image = image * 255.0

        output = self.sam_model.encode(image)
        output = output.reshape(1, 1, 1024, 1024)
        return output.contiguous()


class SamDecoderNuke(nn.Module):
    """
    A wrapper around the SAM model that allows it to be used as a TorchScript model.

    The model is designed to be used in Nuke, where the user can provide up to 24 points.

    The reason for using floats for the points is that Nuke 2D and 3D knobs
    lose their links on the 'Inference' node when reopening a saved Nuke script.
    """

    def __init__(self, sam_model: segment_anything.modeling.Sam, debug: int = 0) -> torch.Tensor:
        super().__init__()
        self.sam_model = sam_model
        self.debug = debug

    def forward(self, x: torch.Tensor):
        """
        Predicts as mask end-to-end from provided image and the center of the image.

        Args:
            image_embeddings: (torch.Tensor) The image embeddings from the original image, in shape 1x1xHxW.

        Returns:
            mask: (torch.Tensor) The image mask, in shape 1x1xHxW.
        """
        debug = self.debug
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Extract the relevant row and slice it to contain only the points data
        num_points = 16
        items_per_point = 3  # x, y, label
        row_data = x[0, 0, 1039, : num_points * items_per_point]

        # Extract points and labels
        points_and_labels = row_data.view(num_points, items_per_point)
        points = points_and_labels[:, 0:2].float().to(device)  # Extracts x, y and converts to float
        labels = points_and_labels[:, 2].int().to(device)  # Extracts labels and converts to int
        labels = labels < 1  # 0 mode is additive, 1 mode is subtractive in Nuke

        # Remove Trackers in Nuke out of the image bounds
        mask = torch.all((points[:, :] >= 1) & (points[:, :] <= 1024), dim=1)
        active_points = points[mask]
        labels = labels[mask]

        # If no active points, return a blank mask
        if active_points.size(0) == 0:
            if debug:
                print("No active points found.")
            return torch.zeros(1, 1, 1040, 1024)

        # Nuke coordinates start from bottom left corner
        active_points[:, 1] = 1024 - active_points[:, 1]

        image_embeddings = x.to(device)
        # The top 16 pixels are reserved for the Nuke data
        image_embeddings = image_embeddings[:, :, 0:1024, 0:1024]
        image_embeddings = image_embeddings.reshape(1, 256, 64, 64)

        # Add batch dimension
        point_coords = active_points[None, :, :]
        point_labels = labels[None, :]

        if debug:
            print("\nActive Points:", point_coords)
            print("Labels:", point_labels)

        mask = self.sam_model(image_embeddings, point_coords, point_labels, True)

        # Back to Nuke coordinates (1024x1040)
        mask = torch.nn.functional.pad(mask, (0, 0, 0, 16))

        return mask.contiguous()


def main():
    """Convert SAM to TorchScript and save it."""

    # Trace the models
    for model_type, checkpoint in SAM_MODELS.items():
        print("=" * 80)
        print(f"Tracing {model_type} model...")

        # Trace the encoder and decoder
        trace_encoder(model_type, checkpoint)
        trace_decoder(model_type, checkpoint)

        print(f"Finished tracing {model_type} model.")


def trace_encoder(model_type, checkpoint):
    sam_model = sam_model_registry[model_type](checkpoint)

    sam_encoder_nuke = SamEncoderNuke(sam_model)
    sam_encoder_nuke.eval()
    sam_encoder_nuke.half()
    sam_encoder_nuke.cuda()

    # Test the model
    sam_encoder_nuke(torch.randn([1, 3, 1024, 1024], device="cuda"))  # RGB image, 1024x1024

    # Trace the model
    with torch.jit.optimized_execution(True):
        scripted_model = torch.jit.script(sam_encoder_nuke)

    # Save the TorchScript model
    DESTINATION = f"{BASE_PATH}/sam_{model_type}_encoder.pt"
    scripted_model.save(DESTINATION)
    print(f"Saved TorchScript model to {DESTINATION} - {file_size(DESTINATION)} MB")


def trace_decoder(model_type, checkpoint):
    sam_model = sam_model_registry[model_type](checkpoint)

    sam_decoder_nuke = SamDecoderNuke(sam_model)
    sam_decoder_nuke.eval()
    sam_decoder_nuke.cuda()

    # Test the model
    sam_decoder_nuke(torch.randn([1, 1, 1040, 1024], device="cuda"))  # 1024x1024 mask + 16x1024 for points

    # Remove the image encoder for the decoding only pass - saving disk space.
    # We need to make sure we don't use the image encoder in the forward pass.
    sam_decoder_nuke.sam_model.image_encoder = None

    with torch.jit.optimized_execution(True):
        # torch.jit.enable_onednn_fusion(True)  # Not supported in PyTorch 1.6
        scripted_model = torch.jit.script(sam_decoder_nuke)
        # scripted_model = torch.jit.freeze(scripted_model.eval())  # Not supported in PyTorch 1.6

    # Save the TorchScript model
    DESTINATION = f"{BASE_PATH}/sam_{model_type}_decoder.pt"
    scripted_model.save(DESTINATION)
    print(f"Saved TorchScript model to {DESTINATION} - {file_size(DESTINATION)} MB")


def file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return int(size_in_bytes / (1024 * 1024))


if __name__ == "__main__":
    main()
